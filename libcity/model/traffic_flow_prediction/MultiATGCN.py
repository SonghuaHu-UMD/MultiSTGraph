import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=False):
    # L~ = 2L/lambda - I
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    lap = calculate_normalized_laplacian(adj)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        # self.lin1 = nn.Linear(embed_dim, embed_dim)
        # self.lin2 = nn.Linear(embed_dim, embed_dim)
        self.alpha = 3
        self.k = embed_dim

    def forward(self, x, node_emb, node_vec1, node_vec2, supports, adpadj):
        node_num = node_emb.shape[0]  # node_emb: E
        if adpadj == 'gwnet':
            supports = F.softmax(F.relu(torch.mm(node_vec1, node_vec2)), dim=1)
        elif adpadj == 'agcrn':
            supports = F.softmax(F.relu(torch.mm(node_emb, node_emb.T)), dim=1)
        elif adpadj == 'mtgnn':
            nodevec1 = torch.tanh(self.alpha * node_vec1)
            nodevec2 = torch.tanh(self.alpha * node_vec2)
            a = torch.mm(nodevec1, nodevec2) - torch.mm(nodevec2.T, nodevec1.T)
            adj = F.relu(torch.tanh(self.alpha * a))
            mask = torch.zeros(node_num, node_num).to(x.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            supports = adj * mask
        else:
            supports = torch.FloatTensor(supports).to(x.device)
        support_set = [torch.eye(node_num).to(x.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_emb, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_emb, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class ATGRUCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(ATGRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim)
        self.update = AGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_emb, node_vec1, node_vec2, supports, adpadj):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_emb, node_vec1, node_vec2, supports, adpadj))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_emb, node_vec1, node_vec2, supports, adpadj))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ATGRUEncoder(nn.Module):
    def __init__(self, config, feature_final):
        super(ATGRUEncoder, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_final = feature_final
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder'
        self.agru_cells = nn.ModuleList()
        self.agru_cells.append(
            ATGRUCell(self.num_nodes, self.feature_final, self.hidden_dim, self.cheb_k, self.embed_dim))
        for _ in range(1, self.num_layers):
            self.agru_cells.append(
                ATGRUCell(self.num_nodes, self.hidden_dim, self.hidden_dim, self.cheb_k, self.embed_dim))

    def forward(self, x, init_state, node_emb, node_vec1, node_vec2, supports, adpadj):
        assert x.shape[2] == self.num_nodes
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.agru_cells[i](
                    current_inputs[:, t, :, :], state, node_emb, node_vec1, node_vec2, supports, adpadj)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.agru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class MultiATGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.add_time_in_day = config.get('add_time_in_day', False)
        self.batch_size = config.get('batch_size', 64)
        self.device = config.get('device', torch.device('cpu'))
        config['num_nodes'] = self.num_nodes
        self.embed_dim = config.get('embed_dim', 10)

        # Adjacent matrix
        self.adj_mx = torch.FloatTensor(self.data_feature.get('adj_mx', None))
        self.adjtype = config.get('adjtype', "scalap")
        self.supports = self.cal_supports(self.adj_mx, self.adjtype)
        self.adpadj = config.get('adpadj', "gwnet")

        # Define adaptive node embedding and graph matrix
        self.static = torch.FloatTensor(data_feature.get('static', None)).to(self.device)
        self.static_fc0 = nn.Sequential(OrderedDict(
            [('embd', nn.Linear(min(self.num_nodes, 20), self.embed_dim, bias=True)), ('relu1', nn.ReLU())])).to(
            self.device)
        if self.static is not None:
            u, s, v = torch.pca_lowrank(self.static, q=min(self.num_nodes, 20))
            initemb = torch.matmul(self.static, v)
            initemb = self.static_fc0(initemb)
            self.node_emb = nn.Parameter(initemb, requires_grad=True)
        else:
            self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)

        if self.adj_mx is not None:
            m, p, n = torch.svd(self.adj_mx)
            initemb1 = torch.mm(m[:, :self.embed_dim], torch.diag(p[:self.embed_dim] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:self.embed_dim] ** 0.5), n[:, :self.embed_dim].t())
            self.node_vec1 = nn.Parameter(initemb1, requires_grad=True)
            self.node_vec2 = nn.Parameter(initemb2, requires_grad=True)
        else:
            self.node_vec1 = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
            self.node_vec2 = nn.Parameter(torch.randn(self.embed_dim, self.num_nodes), requires_grad=True)

        # Dim of different variables: Y; time_in_day 1 ; day_in_week 7 ; external: future known 2; future unknown 5
        self.start_dim = config.get('start_dim', 0)
        self.end_dim = config.get('end_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 1)  # all except end_dim-start_dim
        self.output_dim = self.end_dim - self.start_dim
        self.future_unknown = config.get('future_unknown', 5)  # weather
        self.future_known = config.get('future_known', 2)  # holiday, weekend
        self.time_index_dim = 1 if self.add_time_in_day else 0
        self.feature_raw = self.end_dim - self.start_dim + self.future_unknown + self.future_known
        self.feature_final = self.feature_raw + self.time_index_dim + self.future_known
        self.hidden_dim = config.get('rnn_units', 64)

        # Merge of multi-time heads
        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)
        self.weight_t1 = nn.Parameter(torch.FloatTensor(1, self.len_closeness, self.num_nodes, self.feature_raw))
        self.weight_t2 = nn.Parameter(torch.FloatTensor(1, self.len_period, self.num_nodes, self.feature_raw))
        self.weight_t3 = nn.Parameter(torch.FloatTensor(1, self.len_trend, self.num_nodes, self.feature_raw))

        # Layers
        self.static_fc = nn.Sequential(
            OrderedDict([('embd', nn.Linear(self.static.shape[1], self.hidden_dim, bias=True)), ('relu1', nn.ReLU())]))
        self.encoder = ATGRUEncoder(config, self.feature_final)
        self.end_conv = nn.Conv2d(self.input_window, self.output_window * self.output_dim,
                                  kernel_size=(1, self.hidden_dim), bias=True)

        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def cal_supports(self, adj_mx, adjtype):
        if adjtype == "scalap":
            supports = calculate_scaled_laplacian(adj_mx)
        elif adjtype == "identity":
            supports = np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)
        else:
            print("adj type not defined")
        return supports

    def forward(self, batch):
        # source: get all features except time index
        source = torch.cat((batch['X'][:, :, :, self.start_dim:self.end_dim],
                            batch['X'][:, :, :, -self.ext_dim + self.time_index_dim:]), dim=-1)

        # Merge three temporal unit: end_dim-start_dim + future_unknown (weather) + future_known (holiday/weekend) = 8
        output = 0.0
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            output_hours = source[:, begin_index:end_index, :, :]
            output += output_hours * self.weight_t1  # element-wise weight
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
        if self.add_time_in_day:  # add back the time_index now: 1
            begin_index = 0
            end_index = begin_index + self.len_closeness
            time_in_day = batch['X'][:, begin_index:end_index, :, self.end_dim:self.end_dim + 1]
            output = torch.cat((output, time_in_day), dim=-1)

        # Add future-known variables: holidays and weekends # 2
        if self.output_window == self.input_window:
            fknown_var = batch['y'][:, :, :, self.end_dim + 1:self.end_dim + 1 + self.future_known]
            output = torch.cat((output, fknown_var), dim=-1)
        elif self.output_window < self.input_window:
            fknown_var = batch['y'][:, :, :, self.end_dim + 1:self.end_dim + 1 + self.future_known]
            fknown_var = F.pad(fknown_var, (0, 0, 0, 0, 0, output.shape[1] - fknown_var.shape[1]), 'replicate')
            output = torch.cat((output, fknown_var), dim=-1)

        # GRU encoder: init based on static variables
        init_state = self.encoder.init_hidden(source.shape[0])
        if self.static is not None:
            static_embedding = self.static_fc(self.static)
            init_state = static_embedding.expand(init_state.shape[0], init_state.shape[1], -1, -1)
        output, output_hidden = self.encoder(output, init_state, self.node_emb, self.node_vec1, self.node_vec2,
                                             self.supports, self.adpadj)

        # CNN based output
        output = F.dropout(output, p=0.1, training=self.training)
        output = self.end_conv(output)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes).permute(0, 1, 3, 2)
        return output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[:, 0:self.output_window, :, self.start_dim:self.end_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted)
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
