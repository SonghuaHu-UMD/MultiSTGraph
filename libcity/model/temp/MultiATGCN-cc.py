from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


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


class ATGRUEncoder(nn.Module):
    def __init__(self, config, feature_final, adj_max):
        super(ATGRUEncoder, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_final = feature_final
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.cheb_k = config.get('cheb_order', 2)
        self.adj_mx = adj_max
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.atgru_cells = nn.ModuleList()
        self.atgru_cells.append(ATGRUCell(self.num_nodes, self.feature_final,
                                          self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))
        for _ in range(1, self.num_layers):
            self.atgru_cells.append(ATGRUCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))

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
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.atgru_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class ATGRUDecoder(nn.Module):
    def __init__(self, config, feature_final, adj_max):
        super(ATGRUDecoder, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_final = feature_final
        self.hidden_dim = config.get('rnn_units', 64)
        self.atten = config.get('atten', False)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.start_dim = config.get('start_dim', 0)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.end_dim = config.get('end_dim', 1)
        self.output_dim = self.end_dim - self.start_dim
        self.cheb_k = config.get('cheb_order', 2)
        self.adj_mx = adj_max
        self.projection_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.attention_linear = nn.Linear(self.hidden_dim + 4, self.output_window)
        self.attn_combine = nn.Linear(self.hidden_dim + 4, self.hidden_dim)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.atgru_cells = nn.ModuleList()
        if self.atten:
            for _ in range(0, self.num_layers):
                self.atgru_cells.append(ATGRUCell(self.num_nodes, self.hidden_dim,
                                                  self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))
        else:
            self.atgru_cells.append(ATGRUCell(self.num_nodes, self.output_dim + 3,
                                              self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))
            for _ in range(1, self.num_layers):
                self.atgru_cells.append(ATGRUCell(self.num_nodes, self.hidden_dim,
                                                  self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))

    def forward(self, input, hidden_state, encoder_output, node_embeddings, nodevec1, nodevec2):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        hidden_states = []
        # output = input
        if self.atten:
            attention_input = torch.cat((hidden_state[-1], input), -1)
            attention_weights = F.softmax(self.attention_linear(attention_input), -1).unsqueeze(1)
            output = torch.einsum('ijks,iskr->ijkr', attention_weights, encoder_output).squeeze(1)
            output = torch.cat((output, input), -1)
            output = self.attn_combine(output)
            output = F.relu(output)
        else:
            output = input
        for layer_num, dcgru_layer in enumerate(self.atgru_cells):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], node_embeddings, nodevec1, nodevec2)
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output)
        return projected, torch.stack(hidden_states)


class MultiATGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self.num_nodes = data_feature.get('num_nodes', 1)
        config['num_nodes'] = self.num_nodes
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.add_time_in_day = config.get('add_time_in_day', False)
        self.batch_size = config.get('batch_size', 64)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0)
        self.use_curriculum_learning = config.get('use_curriculum_learning', False)
        self.device = config.get('device', torch.device('cpu'))
        self.static = torch.FloatTensor(data_feature.get('static', None)).to(self.device)
        self.adj_max = torch.FloatTensor(self.data_feature.get('adj_mx', None)).to(self.device)
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

        # Dim of different variables
        # Y; time_in_day 1 ; day_in_week 7 ; external: future known 2; future unknown 5
        self.start_dim = config.get('start_dim', 0)
        self.end_dim = config.get('end_dim', 1)
        self.ext_dim = self.data_feature.get('ext_dim', 1)  # all except end_dim-start_dim
        self.output_dim = self.end_dim - self.start_dim
        self.hidden_dim = config.get('rnn_units', 64)
        self.future_unknown = config.get('future_unknown', 5)  # weather
        self.future_known = config.get('future_known', 2)  # holiday, weekend
        self.time_index_dim = 1 if self.add_time_in_day else 0

        # Merge of multi-time heads
        self.feature_raw = self.end_dim - self.start_dim + self.future_unknown + self.future_known
        self.feature_final = self.feature_raw + self.time_index_dim
        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)
        self.weight_t1 = nn.Parameter(torch.FloatTensor(1, self.len_closeness, 1, self.feature_raw))
        self.weight_t2 = nn.Parameter(torch.FloatTensor(1, self.len_period, 1, self.feature_raw))
        self.weight_t3 = nn.Parameter(torch.FloatTensor(1, self.len_trend, 1, self.feature_raw))

        # Layers
        self.static_fc = nn.Sequential(
            OrderedDict([('embd', nn.Linear(self.static.shape[1], self.hidden_dim, bias=True)), ('relu1', nn.ReLU())]))
        self.encoder = ATGRUEncoder(config, self.feature_final, self.adj_max)
        self.decoder_model = ATGRUDecoder(config, self.feature_final, self.adj_max)
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

    def decoder(self, y_prev, y_labels, decoder_hidden, encoder_output, node_embeddings, nodevec1, nodevec2):
        decoder_hidden_state = decoder_hidden
        decoder_input = y_prev
        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state, encoder_output, node_embeddings, nodevec1, nodevec2)
            decoder_input = torch.cat((decoder_output, y_labels[:, t, :, 1:4]), dim=-1)
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.teacher_forcing_ratio:
                    decoder_input = y_labels[:, t, :, 0:4]
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def forward(self, batch):
        # source: B, T_1, N, D; target: B, T_2, N, D
        source = torch.cat((batch['X'][:, :, :, self.start_dim:self.end_dim],
                            batch['X'][:, :, :, -self.ext_dim + self.time_index_dim:]), dim=-1)

        # Merge three temporal unit: end_dim-start_dim + future_unknown (weather) + future_known (holiday/weekend) = 8
        inputs = 0.0
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            output_hours = source[:, begin_index:end_index, :, :]
            inputs += output_hours * self.weight_t1
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            output_days = source[:, begin_index:end_index, :, :]
            inputs += output_days * self.weight_t2
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            output_weeks = source[:, begin_index:end_index, :, :]
            inputs += output_weeks * self.weight_t3
        if self.add_time_in_day:  # 1
            begin_index = 0
            end_index = begin_index + self.len_closeness
            time_in_day = batch['X'][:, begin_index:end_index, :, self.end_dim:self.end_dim + 1]
            inputs = torch.cat((inputs, time_in_day), dim=-1)

        # GRU encoder: init based on static variables
        init_state = self.encoder.init_hidden(source.shape[0])
        if self.static is not None:
            static_embedding = self.static_fc(self.static)
            init_state = static_embedding.expand(init_state.shape[0], init_state.shape[1], -1, -1)
        encoder_output, encoder_hidden = self.encoder(inputs, init_state, self.node_embeddings, self.nodevec1,
                                                      self.nodevec2)
        # output = output[:, -1:, :, :]  # B, 1, N, hidden

        # S2S based decoder
        decoder_hidden = torch.stack(encoder_hidden)
        y_prev = torch.cat([inputs[:, -1, :, 0:3], inputs[:, -1, :, -2:-1]], dim=-1)
        y_labels = batch['y']
        output = self.decoder(y_prev, y_labels, decoder_hidden, encoder_output, self.node_embeddings, self.nodevec1,
                              self.nodevec2)

        # CNN based predictor
        # skip = self.end_conv(encoder_output)  # B, T*C, N, 1
        # skip = skip.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes).permute(0, 1, 3, 2)
        # output = F.dropout(output, p=0.1, training=self.training)
        # output = torch.relu(output + skip)
        return output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., self.start_dim:self.end_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted)
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)

class ATGRUDecoder(nn.Module):
    def __init__(self, config, feature_final, adj_max):
        super(ATGRUDecoder, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_final = feature_final
        self.hidden_dim = config.get('rnn_units', 64)
        self.atten = config.get('atten', False)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 2)
        self.start_dim = config.get('start_dim', 0)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.end_dim = config.get('end_dim', 1)
        self.output_dim = self.end_dim - self.start_dim
        self.cheb_k = config.get('cheb_order', 2)
        self.adj_mx = adj_max
        self.projection_layer = nn.Linear(self.hidden_dim * 3, self.output_dim)
        self.attention_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.cb_linear = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.attn_v = nn.Linear(self.hidden_dim, 1, bias=False)
        self.attn_combine = nn.Linear(self.hidden_dim + 4, self.hidden_dim)
        self.use_curriculum_learning = config.get('use_curriculum_learning', False)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'

        self.atgru_cells = nn.ModuleList()
        if self.atten:
            for _ in range(0, self.num_layers):
                self.atgru_cells.append(ATGRUCell(self.num_nodes, self.hidden_dim,
                                                  self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))
        else:
            self.atgru_cells.append(ATGRUCell(self.num_nodes, self.output_dim + 3 + self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))
            for _ in range(1, self.num_layers):
                self.atgru_cells.append(ATGRUCell(self.num_nodes, self.hidden_dim + self.hidden_dim,
                                                  self.hidden_dim, self.cheb_k, self.embed_dim, self.adj_mx))

    def forward(self, y_prev, y_labels, decoder_hidden, encoder_output, node_embeddings, nodevec1, nodevec2):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)

        # if self.atten:
        #     attention_input = torch.cat((hidden_state[-1], input), -1)
        #     attention_weights = F.softmax(self.attention_linear(attention_input), -1).unsqueeze(1)
        #     output = torch.einsum('ijks,iskr->ijkr', attention_weights, encoder_output).squeeze(1)
        #     output = torch.cat((output, input), -1)
        #     output = self.attn_combine(output)
        #     output = F.relu(output)
        # else:
        #     output = input
        decoder_input = y_prev
        outputs = []
        for t in range(self.output_window):
            hidden_states = []
            for layer_num, dcgru_layer in enumerate(self.atgru_cells):
                state = decoder_hidden[layer_num]
                hidden = state.unsqueeze(1).repeat(1, self.input_window, 1, 1)
                energy = torch.tanh(self.attention_linear(torch.cat((hidden, encoder_output), dim=-1)))
                attention = F.softmax(self.attn_v(energy).squeeze(-1), dim=1).unsqueeze(1)
                weighted = torch.einsum('ijks,iksr->ijsr', attention, encoder_output).squeeze(1)
                decoder_input_f = torch.cat((decoder_input, weighted), dim=-1)
                decoder_input = dcgru_layer(decoder_input_f, state, node_embeddings, nodevec1, nodevec2)
                # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
                hidden_states.append(state)
            decoder_hidden = hidden_states
            decoder_output = self.projection_layer(torch.cat((decoder_input, decoder_input_f), dim=-1))
            # decoder_hidden_state = torch.stack(hidden_states)
            decoder_input = torch.cat((decoder_output, y_labels[:, t, :, 1:4]), dim=-1)
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.teacher_forcing_ratio:
                    decoder_input = y_labels[:, t, :, 0:4]
        outputs = torch.stack(outputs, dim=1)
        return outputs
