# Created by zhai at 2018/5/30
# Email: zsp1197@163.com
import torch
import torch.nn.functional as F
import torch.nn as nn


class Combine(nn.Module):
    def __init__(self, seq2seq, mlp):
        super(Combine, self).__init__()
        self.seq2seq = seq2seq
        self.mlp = mlp
        self.output_size = self.seq2seq.decoder.output_size
        self.combine_coeffi = nn.Parameter(torch.zeros(self.output_size), requires_grad=True)
        # self.ln1 = nn.LayerNorm(seq2seq.decoder.output_size)
        # self.ln2 = nn.LayerNorm(seq2seq.decoder.output_size)

    def forward(self, input_bin):
        '''

        :param input_bin: (batch,seq_len,input_size)
        :return:
        '''
        # (seq_len,batch_size,output_size)
        seq2seq_out = torch.stack(self.seq2seq(input_bin)[0])
        # (batch_size*seq_len,input_size)
        input_bin_mlp = input_bin.reshape(-1, input_bin.shape[-1])
        # (batch_size*seq_len,output_size)
        mlp_out = self.mlp(input_bin_mlp)
        # (batch, seq_len, output_size)
        mlp_out = mlp_out.reshape(input_bin.shape[0], input_bin.shape[1], self.seq2seq.decoder.output_size)
        # (seq_len,batch_size,output_size)
        mlp_out = mlp_out.transpose(0, 1)
        # (seq_len*batch_size,output_size)
        seq2seq_out = seq2seq_out.reshape(-1, self.output_size)
        # (seq_len*batch_size,output_size)
        mlp_out = mlp_out.reshape(-1, self.output_size)
        combine_coeffi = F.sigmoid(self.combine_coeffi)
        # combine_coeffi = self.combine_coeffi
        out = mlp_out * combine_coeffi + seq2seq_out * (1 - combine_coeffi)
        # return mlp_out,seq2seq_out
        return out

    def refine_target(self, target_bin):
        target_bin = target_bin.transpose(0, 1)
        return target_bin.reshape(-1)
