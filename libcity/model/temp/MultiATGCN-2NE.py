import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from collections import OrderedDict


class FC(nn.Module):  # is_training: self.training
    def __init__(self, input_dims, units, activations, bn, bn_decay, use_bias=True):
        super(FC, self).__init__()
        self.input_dims = input_dims
        self.units = units
        self.activations = activations
        self.bn = bn
        self.bn_decay = bn_decay
        self.use_bias = use_bias
        self.layers = self._init_layers()

    def _init_layers(self):
        ret = nn.Sequential()
        units, activations = self.units, self.activations
        if isinstance(units, int):
            units, activations = [units], [activations]
        elif isinstance(self.units, tuple):
            units, activations = list(units), list(activations)
        assert type(units) == list
        index = 1
        input_dims = self.input_dims
        for num_unit, activation in zip(units, activations):
            if self.use_bias:
                basic_conv2d = nn.Conv2d(input_dims, num_unit, (1, 1), stride=1, padding=0, bias=True)
                nn.init.constant_(basic_conv2d.bias, 0)
            else:
                basic_conv2d = nn.Conv2d(input_dims, num_unit, (1, 1), stride=1, padding=0, bias=False)
            nn.init.xavier_normal_(basic_conv2d.weight)
            ret.add_module('conv2d' + str(index), basic_conv2d)
            if activation is not None:
                if self.bn:
                    decay = self.bn_decay if self.bn_decay is not None else 0.1
                    basic_batch_norm = nn.BatchNorm2d(num_unit, eps=1e-3, momentum=decay)
                    ret.add_module('batch_norm' + str(index), basic_batch_norm)
                ret.add_module('activation' + str(index), activation())
            input_dims = num_unit
            index += 1
        return ret

    def forward(self, x):
        # x: (N, H, W, C)
        x = x.transpose(1, 3).transpose(2, 3)  # x: (N, C, H, W)
        x = self.layers(x)
        x = x.transpose(2, 3).transpose(1, 3)  # x: (N, H, W, C)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, input_length, num_of_vertices, feature_dim, temporal=True, spatial=True, config=None):
        super(PositionEmbedding, self).__init__()
        self.input_length = input_length
        self.num_of_vertices = num_of_vertices
        self.embedding_size = feature_dim
        self.temporal = temporal
        self.spatial = spatial
        self.temporal_emb = torch.nn.Parameter(
            torch.zeros((1, self.input_length, 1, self.embedding_size)).to(config['device']))
        # shape is (1, T, 1, C)
        nn.init.xavier_uniform_(self.temporal_emb)
        self.spatial_emb = torch.nn.Parameter(
            torch.zeros((1, 1, self.num_of_vertices, self.embedding_size)).to(config['device']))
        # shape is (1, 1, N, C)
        nn.init.xavier_uniform_(self.spatial_emb)

    def forward(self, data):
        if self.temporal:
            data += self.temporal_emb
        if self.spatial:
            data += self.spatial_emb
        return data


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, nodevec1, nodevec2):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = nodevec1.shape[0]  # node_embeddings: E
        supports = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)  # A~
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', nodevec1 + nodevec2.T, self.weights_pool) \
            # + torch.einsum('nd,dkio->nkio', nodevec2.T, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(nodevec1 + nodevec2.T, self.bias_pool) \
            # + torch.matmul(nodevec2.T, self.bias_pool)  # N, dim_out
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

    def forward(self, x, state, nodevec1, nodevec2):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, nodevec1, nodevec2))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, nodevec1, nodevec2))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, config, feature_used):
        super(AVWDCRNN, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_used = feature_used
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.feature_used,
                                          self.hidden_dim, self.cheb_k, self.embed_dim))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim))

    def forward(self, x, init_state, nodevec1, nodevec2):
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
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, nodevec1, nodevec2)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class MultiATGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        self.device = config.get('device', torch.device('cpu'))

        self.static = torch.FloatTensor(data_feature.get('static', None)).to(self.device)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim

        super().__init__(config, data_feature)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        # self.output_dim = self.data_feature.get('output_dim', 1)
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)

        self.ext_dim = self.data_feature.get('ext_dim', 1)
        self.adj_max = torch.FloatTensor(self.data_feature.get('adj_mx', None)).to(self.device)
        self.start_dim = config.get('start_dim', 0)
        self.end_dim = config.get('end_dim', 1)
        self.output_dim = self.end_dim - self.start_dim
        self.feature_used = self.end_dim - self.start_dim + self.ext_dim

        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)

        self.temporal_emb = config.get('temporal_emb', True)
        self.spatial_emb = config.get('spatial_emb', True)
        # self.input_embed = int(self.hidden_dim / 4)
        self.input_embed = self.feature_used + 3

        # self.input_fc = FC(input_dims=self.feature_dim, units=[self.input_embed, self.input_embed],
        #                    activations=[nn.ReLU, None], bn=True, bn_decay=0.1)

        # self.static_fc = nn.Linear(self.static.shape[1], self.input_embed)

        self.static_fc = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.static.shape[1], 32, bias=True)),
            ('relu1', nn.ReLU()),
            ('fc', nn.Linear(32, self.input_embed, bias=True)),
            ('relu2', nn.ReLU()),
        ]))

        self.static_fc2 = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.static.shape[1], 32, bias=True)),
            ('relu1', nn.ReLU()),
            ('fc', nn.Linear(32, self.hidden_dim, bias=True)),
            ('relu2', nn.ReLU()),
        ]))

        self.decoder_fc2 = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(3, 32, bias=True)),
            ('relu1', nn.ReLU()),
            ('fc', nn.Linear(32, self.hidden_dim, bias=True)),
            ('relu2', nn.ReLU()),
        ]))

        self.position_embedding = PositionEmbedding(self.input_window, self.num_nodes, self.feature_dim,
                                                    self.temporal_emb, self.spatial_emb, config)
        #
        # self.static_context_variable_selection = GatedResidualNetwork(
        #     input_size=self.static.shape[1], hidden_size=self.hidden_dim, output_size=self.hidden_dim,
        #     dropout=0.1)

        self.weight_t1 = nn.Parameter(torch.FloatTensor(1, self.len_closeness, 1, self.feature_used))
        self.weight_t2 = nn.Parameter(torch.FloatTensor(1, self.len_period, 1, self.feature_used))
        self.weight_t3 = nn.Parameter(torch.FloatTensor(1, self.len_trend, 1, self.feature_used))
        self.weight_s1 = nn.Parameter(torch.FloatTensor(1, 1, self.num_nodes, self.input_embed))
        self.weight_s2 = nn.Parameter(torch.FloatTensor(1, 1, self.num_nodes, self.hidden_dim))

        if self.adj_max is not None:
            m, p, n = torch.svd(self.adj_max)
            initemb1 = torch.mm(m[:, :self.embed_dim], torch.diag(p[:self.embed_dim] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:self.embed_dim] ** 0.5), n[:, :self.embed_dim].t())
            self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
            self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)

        # self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(config, self.input_embed + 6)
        self.end_conv = nn.Conv2d(self.input_window, self.output_window * self.output_dim,
                                  kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv_decoder = nn.Conv2d(self.input_window, self.output_window * self.output_dim,
                                          kernel_size=(1, 3), bias=True)

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
        source = torch.cat(
            (batch['X'][:, :, :, self.start_dim:self.end_dim], batch['X'][:, :, :, -self.ext_dim:]), dim=-1)

        # Merge three temporal unit
        output = 0.0
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            output_hours = source[:, begin_index:end_index, :, :]
            output += output_hours * self.weight_t1
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            output_days = source[:, begin_index:end_index, :, :]
            output += output_days * self.weight_t2
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            output_weeks = source[:, begin_index:end_index, :, :]
            output += output_weeks * self.weight_t3

        output = torch.cat((output, batch['y'][:, :, :, self.end_dim:self.end_dim + 3]), dim=-1)

        # First convolution
        # output = self.input_fc(output)

        # Static variable enrichment
        if self.static is not None:
            u, s, v = torch.pca_lowrank(self.static, q=6)
            initemb = torch.matmul(self.static, v)
            static_embedding = initemb.expand(output.shape[0], output.shape[1], -1, -1)
            # output += static_embedding * self.weight_s1
            output = torch.cat((output, static_embedding), dim=-1)
        # static_embedding = self.static_fc(self.static)
        # static_embedding = static_embedding.expand(output.shape[0], output.shape[1], -1, -1)
        # output += static_embedding * self.weight_s1

        # output = self.position_embedding(output)

        # GRU encoder
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(output, init_state, self.nodevec1, self.nodevec2)  # B, T, N, hidden
        # output = output[:, -1:, :, :]  # B, 1, N, hidden

        # Static variable enrichment
        # if self.static is not None:
        #     static_embedding = self.static_fc2(self.static)
        #     static_embedding = static_embedding.expand(output.shape[0], output.shape[1], -1, -1)
        #     output += static_embedding * self.weight_s2

        # Decoder time index
        # decoder_time = batch['y'][:, :, :, self.end_dim:self.end_dim + 3]
        # decoder_time = self.end_conv_decoder(decoder_time)

        # CNN based predictor
        output = F.dropout(output, p=0.1, training=self.training)
        output = self.end_conv(output)  # B, T*C, N, 1
        # output += decoder_time
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C
        return output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., self.start_dim:self.end_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted)
        # losses = 0.0
        # for i in range(7):
        #     losses += loss.masked_mae_torch(y_predicted[:, :, :, i], y_true[:, :, :, i], 0)
        # return losses
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
