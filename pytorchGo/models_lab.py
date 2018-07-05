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
        self.target_num=target_num
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
        seqeTime_out = self.seqeTime(_inputs,_targets)
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