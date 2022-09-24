import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class FusionLayer(nn.Module):
    # Matrix-based fusion
    def __init__(self, n, h):
        super(FusionLayer, self).__init__()
        # define the trainable parameter
        self.weights = nn.Parameter(torch.FloatTensor(1, 1, n, h))

    def forward(self, x):
        # assuming x is of size B-n-h-w
        x = x * self.weights  # element-wise multiplication
        return x


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]  # node_embeddings: E
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)  # A~
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, config, feature_dim):
        super(AVWDCRNN, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_dim = feature_dim
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.ln = nn.LayerNorm(self.hidden_dim)

        self.fusionlayer = FusionLayer(self.num_nodes, self.hidden_dim)

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.feature_dim,
                                          self.hidden_dim, self.cheb_k, self.embed_dim))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.num_nodes
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden = self.ln(output_hidden[-1])
        # output_hidden = self.fusionlayer(output_hidden.unsqueeze(dim=1))
        # return output_hidden
        return output_hidden[-1].unsqueeze(dim=1)

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class MultiATGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim

        super().__init__(config, data_feature)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)

        # self.node_embeddings_1 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        # self.node_embeddings_2 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        # self.node_embeddings_3 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        # self.node_embeddings_4 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        # self.node_embeddings_5 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        # self.node_embeddings_6 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        # self.node_embeddings_7 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)

        # self.encoder_1 = AVWDCRNN(config, 1)
        # self.encoder_2 = AVWDCRNN(config, 1)
        # self.encoder_3 = AVWDCRNN(config, 1)
        # self.encoder_4 = AVWDCRNN(config, 1)
        # self.encoder_5 = AVWDCRNN(config, 1)
        # self.encoder_6 = AVWDCRNN(config, 1)
        # self.encoder_7 = AVWDCRNN(config, 1)
        self.encoder = AVWDCRNN(config, 1)

        self.ln = nn.LayerNorm(self.hidden_dim)

        self.end_conv = nn.Conv2d(1, self.output_window, kernel_size=(1, self.hidden_dim), bias=True)

        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        source = batch['X']

        # init_state = self.encoder_1.init_hidden(source.shape[0])
        # output = self.encoder_1(source[:, :, :, 0:1], init_state, self.node_embeddings_1)
        # output_f = output.clone()  # B, 1, N, hidden
        #
        # init_state = self.encoder_2.init_hidden(source.shape[0])
        # output = self.encoder_2(source[:, :, :, 1:2], init_state, self.node_embeddings_2)
        # output_f += output  # B, 1, N, hidden
        #
        # init_state = self.encoder_3.init_hidden(source.shape[0])
        # output = self.encoder_3(source[:, :, :, 2:3], init_state, self.node_embeddings_3)
        # output_f += output  # B, 1, N, hidden
        #
        # init_state = self.encoder_4.init_hidden(source.shape[0])
        # output = self.encoder_4(source[:, :, :, 3:4], init_state, self.node_embeddings_4)
        # output_f += output  # B, 1, N, hidden
        #
        # init_state = self.encoder_5.init_hidden(source.shape[0])
        # output = self.encoder_5(source[:, :, :, 4:5], init_state, self.node_embeddings_5)
        # output_f += output  # B, 1, N, hidden
        #
        # init_state = self.encoder_6.init_hidden(source.shape[0])
        # output = self.encoder_6(source[:, :, :, 5:6], init_state, self.node_embeddings_6)
        # output_f += output  # B, 1, N, hidden
        #
        # init_state = self.encoder_7.init_hidden(source.shape[0])
        # output = self.encoder_7(source[:, :, :, 6:7], init_state, self.node_embeddings_7)
        # output_f += output  # B, 1, N, hidden

        # Shortcut
        init_state = self.encoder.init_hidden(source.shape[0])
        output = self.encoder(source[:, :, :, 0:7].sum(-1, keepdims=True), init_state, self.node_embeddings)
        # x_residual = self.ln(F.relu(output_f + output))
        x_residual = output

        # CNN based predictor
        x_residual = self.end_conv(x_residual)  # B, T*C, N, 1
        x_residual = x_residual.squeeze(-1).reshape(-1, self.output_window, 1, self.num_nodes)
        x_residual = x_residual.permute(0, 1, 3, 2)  # B, T, N, C
        return x_residual

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        y_true = y_true.sum(-1, keepdims=True)
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
