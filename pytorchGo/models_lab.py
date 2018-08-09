# Created by zhai at 2018/6/10
# Email: zsp1197@163.com
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn

from Parameters import Parameters


class SingleCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        layers = (input_size * 2, input_size * 10, hidden_size // 5, hidden_size // 2)
        layers_dict = OrderedDict()
        layers_list = []
        for i, layer in enumerate(layers):
            if (i == 0):
                in_size = input_size
                out_size = layers[i + 1]
            elif (i == len(layers) - 1):
                break
            else:
                in_size = out_size
                out_size = layers[i + 1]
            layers_dict.update({str(i) + '_' + 'bn': nn.BatchNorm1d(in_size),
                                str(i) + '_' + 'linear': nn.Linear(in_size, out_size),
                                str(i) + '_' + 'non-linear': nn.ReLU()
                                })
            layers_list.extend([
                # nn.BatchNorm1d(in_size),
                nn.Linear(in_size, out_size),
                nn.ReLU()])
        layers_dict.pop('0_bn')
        # layers_list.pop(0)

        self.mmlp = nn.ModuleList(layers_list)
        self.cell = nn.LSTM(input_size=layers[-1], hidden_size=hidden_size, batch_first=True)

    def forward(self, input):
        # 输入必须是1,因为要控制单个cell
        out = input
        for i, l in enumerate(self.mmlp):
            out = l(out)
        # out = self.mmlp(input)
        out = self.cell(out.unsqueeze(1))
        out = out[0].squeeze(1)
        return F.log_softmax(out, dim=1)


class SeqEmbedTarget(nn.Module):
    def __init__(self, target_num, paras=Parameters()):
        self.paras = paras
        self.target_num = target_num
        super(SeqEmbedTarget, self).__init__()
        self.rnn = nn.LSTM(input_size=target_num, hidden_size=self.paras.embed_size, batch_first=True)
        decode_nets = [nn.Linear(self.paras.embed_size, self.paras.embed_size * 4), nn.ReLU(),
                       nn.Linear(self.paras.embed_size * 4, self.paras.embed_size * 20),
                       nn.BatchNorm1d(self.paras.embed_size * 20), nn.ReLU(),
                       nn.Linear(self.paras.embed_size * 20, target_num), nn.BatchNorm1d(target_num), nn.ReLU()]
        self.decode = nn.ModuleList(decode_nets)

    def forward(self, _input):
        # input:(batch,seq_len,1)
        if (self.training == True and self.paras.teacher_forcing_ratio < 1):
            # 如果是训练，则用teacher_force
            _input = self.refine_input_with_teacher_force(_input)
        input = self.refineInput_onehot(_input).to(_input.device)
        out = self.rnn(input)[0][:, -1, :]
        # out = self.decode(out)
        # out = F.relu(out)
        for i, l in enumerate(self.decode):
            out = l(out)
        return F.log_softmax(out, dim=1)

    def refineInput_onehot(self, input):
        return torch.zeros(input.shape[0], input.shape[1], self.target_num).scatter_(2, input.cpu().unsqueeze(-1), 1)

    def refine_input_with_teacher_force(self, input):
        # input:(batch,seq_len,1)
        for batch_input in input:
            seq_len = len(batch_input)
            num_false = round(seq_len * (1 - self.paras.teacher_forcing_ratio))
            random_idxs = random.sample(list(range(seq_len)), num_false)
            for idx in random_idxs:
                batch_input[idx] = self.random_label()
        return input

    def random_label(self):
        return random.choice(list(range(self.target_num)))


class Combine_MLP_SET(nn.Module):
    def __init__(self, mlp, seqet: SeqEmbedTarget, output_size, predict=1, paras=Parameters()):
        super(Combine_MLP_SET, self).__init__()
        self.paras = paras
        self.predict = predict
        self.mlp = mlp
        self.seqet = seqet
        self.combine_coeffi_mlp = nn.Parameter(torch.zeros(output_size), requires_grad=True)
        self.combine_coeffi_seq = nn.Parameter(torch.zeros(output_size), requires_grad=True)
        self.combine_coeffi = nn.Parameter(torch.zeros(output_size), requires_grad=True)

    def forward(self, _inputs, _targets):
        # _inputs: (batch,seq_len,input_size:4)
        # _targets: (batch,seq_len,1)
        seqet_out = self.seqet(_targets)
        mlp_out = self.mlp(_inputs[:, -1, :])
        combine_coeffi_mlp = F.sigmoid(self.combine_coeffi_mlp)
        combine_coeffi_seq = F.sigmoid(self.combine_coeffi_seq)
        out = mlp_out * combine_coeffi_mlp + seqet_out * combine_coeffi_seq
        return out

    def forward_mse(self, _inputs, _targets):
        seqet_out = torch.exp(self.seqet(_targets))
        mlp_out = torch.exp(self.mlp(_inputs[:, -1, :]))
        out = mlp_out * self.combine_coeffi + seqet_out * (1 - self.combine_coeffi)
        return out


class SeqEmbedTime(nn.Module):
    def __init__(self, target_num, paras=Parameters()):
        super(SeqEmbedTime, self).__init__()
        self.paras = paras
        self.target_num = target_num
        self.rnn = nn.LSTM(input_size=target_num, hidden_size=self.paras.embed_size * 10, batch_first=True)
        self.embed_past = nn.ModuleList(
            [nn.Linear(self.paras.embed_size * 10, self.paras.embed_size), nn.BatchNorm1d(self.paras.embed_size),
             nn.ReLU()])
        self.embed_now = nn.ModuleList(
            [nn.Linear(2, self.paras.embed_size // 2), nn.BatchNorm1d(self.paras.embed_size // 2), nn.ReLU(),
             nn.Linear(self.paras.embed_size // 2, self.paras.embed_size), nn.BatchNorm1d(self.paras.embed_size),
             nn.ReLU()])
        decode_nets = [nn.Linear(self.paras.embed_size, self.paras.embed_size * 4), nn.ReLU(),
                       nn.Linear(self.paras.embed_size * 4, self.paras.embed_size * 20),
                       nn.BatchNorm1d(self.paras.embed_size * 20), nn.ReLU(),
                       nn.Linear(self.paras.embed_size * 20, target_num), nn.BatchNorm1d(target_num), nn.ReLU()]
        self.decode = nn.ModuleList(decode_nets)

    def forward(self, _inputs, _targets):
        if (self.training == True and self.paras.teacher_forcing_ratio < 1):
            # 如果是训练，则用teacher_force
            _targets = self.refine_input_with_teacher_force(_targets)
        embed_past = self.refineInput_onehot(_targets).to(_targets.device)
        embed_past = self.rnn(embed_past)[0][:, -1, :]
        for l in self.embed_past:
            embed_past = l(embed_past)
        embed_now = _inputs[:, -1, :2]
        for l in self.embed_now:
            embed_now = l(embed_now)
        out = embed_now + embed_past
        for l in self.decode:
            out = l(out)
        return F.log_softmax(out, dim=1)

    def refineInput_onehot(self, input):
        return torch.zeros(input.shape[0], input.shape[1], self.target_num).scatter_(2, input.cpu().unsqueeze(-1), 1)

    def refine_input_with_teacher_force(self, input):
        # input:(batch,seq_len,1)
        for batch_input in input:
            seq_len = len(batch_input)
            num_false = round(seq_len * (1 - self.paras.teacher_forcing_ratio))
            random_idxs = random.sample(list(range(seq_len)), num_false)
            for idx in random_idxs:
                batch_input[idx] = self.random_label()
        return input

    def random_label(self):
        return random.choice(list(range(self.target_num)))


class Combine_MLP_SETime(nn.Module):
    def __init__(self, mlp, seqeTime: SeqEmbedTime, output_size, predict=1, paras=Parameters()):
        super(Combine_MLP_SETime, self).__init__()
        self.paras = paras
        self.predict = predict
        self.mlp = mlp
        self.seqeTime = seqeTime
        self.combine_coeffi_mlp = nn.Parameter(torch.zeros(output_size), requires_grad=True)
        self.combine_coeffi_seq = nn.Parameter(torch.zeros(output_size), requires_grad=True)
        self.combine_coeffi = nn.Parameter(torch.zeros(output_size), requires_grad=True)

    def forward(self, _inputs, _targets):
        # _inputs: (batch,seq_len,input_size:4)
        # _targets: (batch,seq_len,1)
        seqeTime_out = self.seqeTime(_inputs, _targets)
        mlp_out = self.mlp(_inputs[:, -1, :])
        combine_coeffi_mlp = F.sigmoid(self.combine_coeffi_mlp)
        combine_coeffi_seq = F.sigmoid(self.combine_coeffi_seq)
        combine_coeffi = F.sigmoid(self.combine_coeffi)
        # TODO 融合方法有大问题，没有softmax,每一个也都是log后的
        # out = mlp_out * combine_coeffi_mlp + seqeTime_out * combine_coeffi_seq
        out = torch.log(torch.exp(mlp_out) * combine_coeffi + torch.exp(seqeTime_out) * (1 - combine_coeffi))
        return out

    def forward_mse(self, _inputs, _targets):
        seqet_out = torch.exp(self.seqet(_targets))
        mlp_out = torch.exp(self.mlp(_inputs[:, -1, :]))
        out = mlp_out * self.combine_coeffi + seqet_out * (1 - self.combine_coeffi)
        return out


class SeqAttn(nn.Module):
    def __init__(self, target_num):
        super(SeqAttn, self).__init__()
        self.paras = Parameters()
        self.target_num = target_num
        self.embed_query = nn.Sequential(nn.Linear(in_features=3, out_features=4), nn.ReLU())
        self.embed_key = nn.Sequential(nn.Linear(in_features=4, out_features=4), nn.ReLU())
        self.def_paras_4_predict_state()
        self.def_paras_4_predict_od()
        self.def_paras_4_embed_past()

    def def_paras_4_predict_state(self):
        self.transform_targets = nn.ModuleList([nn.Linear(self.target_num, self.paras.embed_size * 20),
                                                # nn.BatchNorm1d(self.paras.embed_size * 20),
                                                nn.ReLU(),
                                                nn.Linear(self.paras.embed_size * 20, self.paras.embed_size * 10),
                                                # nn.BatchNorm1d(self.paras.embed_size * 10),
                                                nn.ReLU(),
                                                nn.Linear(self.paras.embed_size * 10, self.paras.embed_size),
                                                # nn.BatchNorm1d(self.paras.embed_size),
                                                nn.ReLU()])
        self.transform_query = nn.Sequential(nn.Linear(3, self.paras.embed_size),
                                             nn.BatchNorm1d(self.paras.embed_size),
                                             nn.ReLU())
        self.decode = nn.Sequential(nn.Linear(self.paras.embed_size, self.paras.embed_size * 4), nn.ReLU(),
                                    nn.Linear(self.paras.embed_size * 4, self.paras.embed_size * 20),
                                    nn.BatchNorm1d(self.paras.embed_size * 20), nn.ReLU(),
                                    nn.Linear(self.paras.embed_size * 20, self.target_num),
                                    nn.BatchNorm1d(self.target_num), nn.ReLU())

    def def_paras_4_predict_od(self):
        self.transform_targets_od = nn.ModuleList([nn.Linear(self.target_num, self.paras.embed_size * 20),
                                                   # nn.BatchNorm1d(self.paras.embed_size * 20),
                                                   nn.ReLU(),
                                                   nn.Linear(self.paras.embed_size * 20, self.paras.embed_size * 10),
                                                   # nn.BatchNorm1d(self.paras.embed_size * 10),
                                                   nn.ReLU(),
                                                   nn.Linear(self.paras.embed_size * 10, self.paras.embed_size),
                                                   # nn.BatchNorm1d(self.paras.embed_size),
                                                   nn.ReLU()])
        self.transform_query_od = nn.Sequential(nn.Linear(3, self.paras.embed_size),
                                                nn.BatchNorm1d(self.paras.embed_size),
                                                nn.ReLU())

        self.decode_od = nn.Sequential(nn.Linear(self.paras.embed_size, self.paras.embed_size // 2),
                                       nn.BatchNorm1d(self.paras.embed_size // 2),
                                       nn.ReLU(),
                                       nn.Linear(self.paras.embed_size // 2, 1),
                                       nn.ReLU())

    def def_paras_4_embed_past(self):
        self.rnn = nn.LSTM(input_size=self.target_num, hidden_size=self.paras.embed_size, batch_first=True)

    def forward(self, _inputs, _targets):
        # batch first
        query = torch.stack([_inputs[:, -1, :][:, 0], _inputs[:, -1, :][:, 2], _inputs[:, -1, :][:, 3]],
                            dim=1).unsqueeze(1).to(next(self.parameters()).device)
        attn_weights = self.attn(_inputs[:, :-1, :], query)
        targets_applied = attn_weights * self.refineInput_onehot(_targets)
        targets_applied = torch.sum(targets_applied, 1)
        embed_past = self.embed_past(_targets)
        predicted_states = self.predict_state(query=query, targets_applied=targets_applied, embed_past=embed_past)
        _predicted_state_logits = F.log_softmax(predicted_states, dim=1)
        _predicted_od = self.predict_od(query=query, targets_applied=targets_applied, embed_past=embed_past,
                                        predicted_states=F.softmax(predicted_states, dim=1))
        return _predicted_state_logits, _predicted_od

    def embed_past(self, _targets):
        return self.rnn(self.refineInput_onehot(_targets))[0][:, -1, :]

    def get_loss(self, _predicted_logits, target_idxs, _predicted_ods, target_ods):
        classifi_criterion = nn.NLLLoss()
        od_criterion = nn.MSELoss()
        classifi_loss = classifi_criterion(_predicted_logits, target_idxs)
        od_loss = od_criterion(_predicted_ods, target_ods.reshape(-1, 1))
        # TODO add regulation
        return self.paras.trade_off_classifi * classifi_loss + (
                1 - self.paras.trade_off_classifi) * od_loss, classifi_loss, od_loss

    def predict_od(self, query, targets_applied, embed_past, predicted_states):
        transformed_query = self.transform_query_od(query.squeeze(1))
        transformed_targets = targets_applied
        for net in self.transform_targets_od:
            transformed_targets = net(transformed_targets)
        od = self.decode_od(transformed_query + transformed_targets + embed_past)
        return od

    def predict_state(self, query, targets_applied, embed_past):
        transformed_query = self.transform_query(query.squeeze(1))

        transformed_targets = targets_applied
        for net in self.transform_targets:
            transformed_targets = net(transformed_targets)
        logits = self.decode(transformed_query + transformed_targets + embed_past)
        return logits

    def attn(self, keys, query):
        embed_query = self.embed_query(query).squeeze(1).unsqueeze(2)
        embed_keys = self.embed_key(keys)
        weights = torch.matmul(embed_keys, embed_query)
        weights = F.softmax(weights)
        return weights

    def refineInput_onehot(self, input):
        return torch.zeros(input.shape[0], input.shape[1], self.target_num).scatter_(2, input.cpu().unsqueeze(-1),
                                                                                     1).to(
            next(self.parameters()).device)
