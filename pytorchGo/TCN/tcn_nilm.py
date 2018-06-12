# Created by zhai at 2018/6/6
# Email: zsp1197@163.com
import torch
from torch import nn
import torch.nn.functional as F
from pytorchGo.TCN.tcn import TemporalConvNet


class TCN_nilm(nn.Module):
    def __init__(self, input_channel, outputs_chanels: tuple, kernel_size=3, dropout=0):
        super(TCN_nilm, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_channel, num_channels=outputs_chanels, kernel_size=kernel_size,
                                   dropout=dropout)

    def forward(self, input):
        return F.log_softmax(self.tcn(input),dim=1)
