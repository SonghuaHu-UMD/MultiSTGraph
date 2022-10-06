import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
from collections import OrderedDict
import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.spatial.distance import cdist


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


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def calculate_adjacency_matrix_dist(dist_mx, weight_adj_epsilon=0.0):
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    dist_mx = np.exp(-np.square(dist_mx / std))
    dist_mx[dist_mx < weight_adj_epsilon] = 0
    return dist_mx


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, adjtype, adpadj):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.adjtype = adjtype
        self.adpadj = adpadj
        if self.adjtype == 'multi' and self.adpadj in ['bidirection', 'unidirection']:
            cheb_ks = 1 + (self.cheb_k - 1) * 4
        elif self.adjtype == 'multi' and self.adpadj == 'none':
            cheb_ks = 1 + (self.cheb_k - 1) * 3
        else:
            cheb_ks = self.cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_ks, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.alpha = 3
        self.k = embed_dim

    def forward(self, x, node_emb, node_vec1, node_vec2, support_static):
        node_num = node_emb.shape[0]  # node_emb: E
        I_mx = torch.eye(node_num).to(x.device)
        # If using adaptive graph
        if self.adpadj == 'bidirection':
            support_adp = [[I_mx, F.softmax(F.relu(torch.mm(node_vec1, node_vec2)), dim=1)]]
        elif self.adpadj == 'unidirection':
            support_adp = [[I_mx, F.softmax(F.relu(torch.mm(node_emb, node_emb.T)), dim=1)]]
        elif self.adpadj == 'none':
            support_adp = None
        # If using multi graph
        if self.adpadj == 'none':
            supports = support_static
        else:
            if self.adjtype == 'multi' and self.adpadj in ['bidirection', 'unidirection']:
                supports = support_adp + support_static
            elif self.adjtype != 'multi' and self.adpadj in ['bidirection', 'unidirection']:
                supports = support_adp
        out = [I_mx]
        for s in range(0, len(supports)):
            support_set = supports[s].copy()
            support_set = [var.to(x.device) for var in support_set]
            for k in range(2, self.cheb_k):
                support_set.append(torch.matmul(2 * support_set[1], support_set[-1]) - support_set[-2])
            out.extend(support_set[1:])
        supports = torch.stack(out, dim=0).to(x.device)
        # supports = self.mlp(supports).view(1, node_num, node_num)
        weights = torch.einsum('nd,dkio->nkio', node_emb, self.weights_pool)
        bias = torch.matmul(node_emb, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv


class ATGRUCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, adjtype, adpadj):
        super(ATGRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + dim_out, 2 * dim_out, cheb_k, embed_dim, adjtype, adpadj)
        self.update = AGCN(dim_in + dim_out, dim_out, cheb_k, embed_dim, adjtype, adpadj)

    def forward(self, x, state, node_emb, node_vec1, node_vec2, supports_static):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_emb, node_vec1, node_vec2, supports_static))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_emb, node_vec1, node_vec2, supports_static))
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
        self.adjtype = config.get('adjtype', 'od')
        self.adpadj = config.get('adpadj', 'bidirection')
        self.cheb_k = config.get('cheb_order', 2)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder'
        self.agru_cells = nn.ModuleList()
        self.agru_cells.append(ATGRUCell(self.num_nodes, self.feature_final, self.hidden_dim, self.cheb_k,
                                         self.embed_dim, self.adjtype, self.adpadj))
        for _ in range(1, self.num_layers):
            self.agru_cells.append(ATGRUCell(self.num_nodes, self.hidden_dim, self.hidden_dim, self.cheb_k,
                                             self.embed_dim, self.adjtype, self.adpadj))

    def forward(self, x, init_state, node_emb, node_vec1, node_vec2, supports):
        assert x.shape[2] == self.num_nodes
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.agru_cells[i](current_inputs[:, t, :, :], state, node_emb, node_vec1, node_vec2, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.agru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)


class OutputLayer(nn.Module):
    def __init__(self, num_of_vertices, input_length, num_of_features, num_of_filters=128, predict_length=12,
                 output_dim=1):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length
        self.output_dim = output_dim
        self.hidden_layer = nn.Linear(self.input_length * self.num_of_features, self.num_of_filters)
        self.ouput_layer = nn.Linear(self.num_of_filters, self.predict_length * self.output_dim)

    def forward(self, data):
        data = torch.transpose(data, 1, 2)
        data = torch.reshape(data, (-1, self.num_of_vertices, self.input_length * self.num_of_features))
        data = torch.relu(self.hidden_layer(data))
        data = self.ouput_layer(data).reshape((-1, self.num_of_vertices, self.predict_length, self.output_dim))
        data = data.permute(0, 2, 1, 3)
        return data


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

        # Adjacent matrix: OD
        adj_mx_od = torch.FloatTensor(self.data_feature.get('adj_mx', None))
        # adj_mx_od = torch.div(adj_mx_od, adj_mx_od.max())
        # adj_mx_od = adj_mx_od.fill_diagonal_(1)
        self.adj_mx_od = adj_mx_od

        # Adjacent matrix: Semantic similarity
        self.static = data_feature.get('static', None)
        if self.static is not None:
            static_euc = cdist(self.static, self.static, metric='euclidean')
            static_euc[static_euc == 0] = 1
            self.adj_mx_cos = torch.FloatTensor(1 / static_euc)
        else:
            self.adj_mx_cos = torch.eye(self.num_nodes)

        # Adjacent matrix: Distance
        geo_coor = data_feature.get('coordinate', None)
        geo_coor[['x', 'y']] = geo_coor['coordinates']. \
            str.replace('[', ',').str.replace(']', ',').str.split(r',', expand=True)[[1, 2]].astype(float)
        geo_mx = pd.concat([geo_coor] * len(geo_coor), ignore_index=True)
        geo_mx[['geo_id_1', 'x_1', 'y_1']] = geo_coor.loc[
            geo_coor.index.repeat(len(geo_coor)), ['geo_id', 'x', 'y']].reset_index(drop=True)
        geo_mx['dist'] = haversine_array(geo_mx['y'], geo_mx['x'], geo_mx['y_1'], geo_mx['x_1'])
        geo_mx = geo_mx.pivot(index='geo_id', columns='geo_id_1', values='dist').values
        self.adj_mx_dis = torch.FloatTensor(calculate_adjacency_matrix_dist(geo_mx, 0.0))

        self.adpadj = config.get('adpadj', "bidirection")
        self.adjtype = config.get('adjtype', "od")
        I_mx = torch.eye(self.num_nodes)
        if self.adjtype == 'multi':
            self.adj_mx = self.adj_mx_od
            self.supports = [[I_mx, torch.FloatTensor(calculate_scaled_laplacian(self.adj_mx_od))],
                             [I_mx, torch.FloatTensor(calculate_scaled_laplacian(self.adj_mx_dis))],
                             [I_mx, torch.FloatTensor(calculate_scaled_laplacian(self.adj_mx_cos))]]
        elif self.adjtype == 'od':
            self.adj_mx = self.adj_mx_od
            self.supports = [[I_mx, torch.FloatTensor(calculate_scaled_laplacian(self.adj_mx))]]
        elif self.adjtype == 'dist':
            self.adj_mx = self.adj_mx_dis
            self.supports = [[I_mx, torch.FloatTensor(calculate_scaled_laplacian(self.adj_mx))]]
        elif self.adjtype == 'cosine':
            self.adj_mx = self.adj_mx_cos
            self.supports = [[I_mx, torch.FloatTensor(calculate_scaled_laplacian(self.adj_mx))]]
        elif self.adjtype == 'identity':
            self.adj_mx = I_mx
            self.supports = [[I_mx, I_mx]]

        # Define adaptive node embedding and graph matrix
        if self.static is not None:
            self.static = torch.FloatTensor(self.static).to(self.device)
            self.static_initial = nn.Sequential(OrderedDict(
                [('embd', nn.Linear(min(self.num_nodes, 20), self.embed_dim, bias=True)), ('relu1', nn.ReLU())])).to(
                self.device)
            u, s, v = torch.pca_lowrank(self.static, q=min(self.num_nodes, 20))
            initemb = torch.matmul(self.static, v)
            initemb = self.static_initial(initemb)
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

        # Dim of different variables: Y; time_in_day 1 ; external: future known 2; future unknown 5
        self.start_dim = config.get('start_dim', 0)
        self.end_dim = config.get('end_dim', 1)
        self.load_dynamic = config.get('load_dynamic', True)
        self.time_index_dim = 1 if self.add_time_in_day else 0
        self.ext_dim = self.data_feature.get('ext_dim', 1)  # self.feature_dim - self.output_dim
        self.output_dim = self.end_dim - self.start_dim
        self.future_known = config.get('future_known', 0)  # holiday, weekend
        self.future_unknown = self.ext_dim - self.time_index_dim - self.future_known  # weather
        if self.load_dynamic == False:
            self.future_known = 0
            self.future_unknown = 0
        self.feature_final = self.end_dim - self.start_dim + self.time_index_dim + self.future_known
        self.hidden_dim = config.get('rnn_units', 64)

        # Merge of multi-time heads
        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)
        self.len_ts = int((self.len_period + self.len_trend + self.len_closeness) / 24)
        self.weight_ts = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(1, 24, self.num_nodes, self.output_dim)) for i in range(self.len_ts)])
        self.weight_tsg = nn.ParameterList([nn.Parameter(torch.FloatTensor(1)) for i in range(self.len_ts)])

        # Layers
        if self.static is not None:
            self.static_fc = nn.Sequential(OrderedDict(
                [('embd', nn.Linear(self.static.shape[1], self.hidden_dim, bias=True)), ('relu1', nn.ReLU())]))
        if self.load_dynamic:
            self.exter_fc = nn.Sequential(
                OrderedDict([('embd', nn.Linear((self.future_unknown + 2 * self.future_known) * self.input_window,
                                                64, bias=True)),
                             ('relu1', nn.ReLU()), ('linear', nn.Linear(64, self.output_window * self.output_dim,
                                                                        bias=True)), ('relu1', nn.ReLU())]))
            # self.exter_fc = nn.Sequential(OrderedDict(
            #     [('embd', nn.Linear(self.future_unknown + 2 * self.future_known, self.hidden_dim, bias=True)),
            #      ('relu1', nn.ReLU())]))
            # self.exter_conv = nn.Conv2d(self.input_window, self.output_window * self.output_dim,
            #                             kernel_size=(1, self.hidden_dim), bias=True)
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

    def forward(self, batch):
        # population inflow
        source = batch['X'][:, :, :, self.start_dim:self.end_dim]

        # three temporal unit: end_dim-start_dim + future_unknown (weather) + future_known (holiday/weekend) = 8
        output = 0.0
        ccount = 0
        if self.len_closeness > 0:
            begin_index = 0
            for kk in range(0, int(self.len_closeness / 24)):
                end_index = begin_index + 24
                output_hours = source[:, begin_index:end_index, :, :]
                begin_index = end_index
                output += self.weight_tsg[ccount] * output_hours * self.weight_ts[ccount]  # element-wise weight
                ccount += 1
        if self.len_period > 0:
            begin_index = self.len_closeness
            for kk in range(0, int(self.len_period / 24)):
                end_index = begin_index + 24
                output_days = source[:, begin_index:end_index, :, :]
                begin_index = end_index
                output += self.weight_tsg[ccount] * output_days * self.weight_ts[ccount]
                ccount += 1
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            for kk in range(0, int(self.len_trend / 24)):
                end_index = begin_index + 24
                output_weeks = source[:, begin_index:end_index, :, :]
                output += self.weight_tsg[ccount] * output_weeks * self.weight_ts[ccount]
                ccount += 1
        if self.add_time_in_day:  # add back the time_index now: 1
            begin_index = 0
            end_index = begin_index + 24
            time_in_day = batch['X'][:, begin_index:end_index, :, self.end_dim:self.end_dim + 1]
            output = torch.cat((output, time_in_day), dim=-1)

        # future-known variables: holidays and weekends # 2
        if self.load_dynamic and self.future_known > 0:
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
        outputs, output_hidden = self.encoder(output, init_state, self.node_emb, self.node_vec1, self.node_vec2,
                                              self.supports)

        # CNN-based output
        outputs = F.dropout(outputs, p=0.1, training=self.training)
        outputs = self.end_conv(outputs)
        outputs = outputs.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes) \
            .permute(0, 1, 3, 2)

        # # External enrich
        # if self.load_dynamic:
        #     # future-known (2, Y) + future-unknown (5, X) + future-known (2, X)
        #     exter_vars = torch.cat(
        #         (batch['X'][:, 0:self.input_window, :, self.end_dim + self.time_index_dim:], fknown_var), dim=-1)
        #     # exter_vars = self.exter_conv(self.exter_fc(exter_vars))
        #     # exter_vars = exter_vars.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes) \
        #     #     .permute(0, 1, 3, 2)
        #     # outputs = outputs + exter_vars
        #     exter_vars = torch.permute(exter_vars, (0, 2, 1, 3)).reshape(exter_vars.shape[0], self.num_nodes, -1)
        #     outputs = outputs + self.exter_fc(exter_vars).reshape(-1, self.num_nodes, self.output_window,
        #                                                           self.output_dim).permute(0, 2, 1, 3)
        return outputs

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., self.start_dim:self.end_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted)
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
