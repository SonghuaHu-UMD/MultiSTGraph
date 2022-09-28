import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from collections import OrderedDict
import numpy as np


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, adj_mx):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.adj_mx = adj_mx

    def forward(self, x, node_embeddings, nodevec1, nodevec2):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]  # node_embeddings: E
        supports = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
        # supports = self.adj_mx
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


class ATGRUCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, adj_mx):
        super(ATGRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, adj_mx)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, adj_mx)

    def forward(self, x, state, node_embeddings, nodevec1, nodevec2):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, nodevec1, nodevec2))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, nodevec1, nodevec2))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class FusionLayer(nn.Module):
    # Matrix-based fusion
    def __init__(self, out_window, n, out_dim):
        super(FusionLayer, self).__init__()
        # define the trainable parameter
        self.weights = nn.Parameter(torch.FloatTensor(1, out_window, n, out_dim))

    def forward(self, x):
        # assuming x is of size B-n-h-w
        x = x * self.weights  # element-wise multiplication
        return x


class ATGRUEncoder(nn.Module):
    def __init__(self, config, feature_final, adj_max):
        super(ATGRUEncoder, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_final = feature_final
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = config.get('output_dim', 1)
        self.adj_mx = adj_max
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.atgru_cells = nn.ModuleList()
        self.atgru_cells.append(ATGRUCell(self.num_nodes, self.feature_final,
                                          self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))
        for _ in range(1, self.num_layers):
            self.atgru_cells.append(ATGRUCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))
        # self.ln = nn.LayerNorm(self.hidden_dim)
        self.end_conv = nn.Conv2d(self.input_window, self.output_window * self.output_dim,
                                  kernel_size=(1, self.hidden_dim), bias=True)
        self.fusionlayer = FusionLayer(self.output_window, self.num_nodes, self.output_dim)

    def forward(self, x, init_state, node_embeddings, nodevec1, nodevec2):
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
                state = self.atgru_cells[i](current_inputs[:, t, :, :], state, node_embeddings, nodevec1, nodevec2)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        current_inputs = F.dropout(current_inputs, p=0.1, training=self.training)
        # current_inputs = self.ln(current_inputs)
        output = self.end_conv(current_inputs)
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes).permute(0, 1, 3, 2)
        output = self.fusionlayer(output)
        return output

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.atgru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class MultiATGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.add_time_in_day = config.get('add_time_in_day', False)
        self.batch_size = config.get('batch_size', 64)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0)
        self.use_curriculum_learning = config.get('use_curriculum_learning', False)

        self.device = config.get('device', torch.device('cpu'))
        self.static = torch.FloatTensor(data_feature.get('static', None)).to(self.device)
        self.adj_max = torch.FloatTensor(self.data_feature.get('adj_mx', None)).to(self.device)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim
        self.embed_dim = config.get('embed_dim', 10)

        # Define adaptive node embedding and graph matrix
        if self.static is not None:
            u, s, v = torch.pca_lowrank(self.static, q=self.embed_dim)
            initemb = torch.matmul(self.static, v)
            self.node_embeddings = nn.Parameter(initemb, requires_grad=True)
        else:
            self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)

        if self.adj_max is not None:
            m, p, n = torch.svd(self.adj_max)
            initemb1 = torch.mm(m[:, :self.embed_dim], torch.diag(p[:self.embed_dim] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:self.embed_dim] ** 0.5), n[:, :self.embed_dim].t())
            self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
            self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
        else:
            self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(self.embed_dim, self.num_nodes), requires_grad=True)

        # dim of different variables
        # Y; time_in_day 1 ; day_in_week 7 ; external: future known 2; future unknown 5
        self.start_dim = config.get('start_dim', 0)
        self.end_dim = config.get('end_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 1)  # all except end_dim-start_dim
        self.output_dim = self.end_dim - self.start_dim
        self.future_unknown = config.get('future_unknown', 5)  # weather
        self.future_known = config.get('future_known', 2)  # holiday, weekend
        self.time_index_dim = 1 if self.add_time_in_day else 0

        # Merge of Multi-time heads
        self.feature_raw = self.end_dim - self.start_dim + self.future_unknown + self.future_known
        self.feature_final = self.feature_raw + self.time_index_dim + self.future_known
        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)

        # Layers
        self.hidden_dim = config.get('rnn_units', 64)
        self.static_fc = nn.Sequential(
            OrderedDict([('embd', nn.Linear(self.static.shape[1], self.hidden_dim, bias=True)), ('relu1', nn.ReLU())]))
        self.encoder_close = ATGRUEncoder(config, self.feature_final, self.adj_max)
        self.encoder_period = ATGRUEncoder(config, self.feature_final, self.adj_max)
        self.encoder_trend = ATGRUEncoder(config, self.feature_final, self.adj_max)
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def add_future_known(self, dataf, batch):
        if self.output_window == self.input_window:
            future_known_var = batch['y'][:, :, :,
                               self.end_dim + self.time_index_dim:self.end_dim + self.time_index_dim + self.future_known]
            dataf = torch.cat((dataf, future_known_var), dim=-1)
        elif self.output_window < self.input_window:
            future_known_var = batch['y'][:, :, :,
                               self.end_dim + self.time_index_dim:self.end_dim + self.time_index_dim + self.future_known]
            future_known_var = F.pad(future_known_var, (0, 0, 0, 0, 0, self.input_window - self.output_window),
                                     'replicate')
            dataf = torch.cat((dataf, future_known_var), dim=-1)
        return dataf

    def forward(self, batch):
        # source: B, T_1, N, D; target: B, T_2, N, D
        source = torch.cat((batch['X'][:, :, :, self.start_dim:self.end_dim],
                            batch['X'][:, :, :, -self.ext_dim:]), dim=-1)

        # GRU encoder: init based on static variables
        init_state = self.encoder_close.init_hidden(source.shape[0])
        if self.static is not None:
            static_embedding = self.static_fc(self.static)
            init_state = static_embedding.expand(init_state.shape[0], init_state.shape[1], -1, -1)

        # Merge three temporal unit: end_dim-start_dim + future_unknown (weather) + future_known (holiday/weekend) = 8
        output = 0.0
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            output_hours = source[:, begin_index:end_index, :, :]
            output_hours = self.add_future_known(output_hours, batch)
            output_hours = self.encoder_close(output_hours, init_state, self.node_embeddings, self.nodevec1,
                                              self.nodevec2)
            output += output_hours
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            output_days = source[:, begin_index:end_index, :, :]
            output_days = self.add_future_known(output_days, batch)
            output_days = self.encoder_period(output_days, init_state, self.node_embeddings, self.nodevec1,
                                              self.nodevec2)
            output += output_days
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            output_trend = source[:, begin_index:end_index, :, :]
            output_trend = self.add_future_known(output_trend, batch)
            output_trend = self.encoder_trend(output_trend, init_state, self.node_embeddings, self.nodevec1,
                                              self.nodevec2)
            output += output_trend
        return output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., self.start_dim:self.end_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted)
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
