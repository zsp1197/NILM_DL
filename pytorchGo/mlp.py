# Created by zhai at 2018/5/28
# Email: zsp1197@163.com
import torch
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=input_size * 5)
        self.bn1 = nn.BatchNorm1d(num_features=self.linear1.out_features)
        self.linear2 = nn.Linear(in_features=self.linear1.out_features, out_features=input_size * 10)
        self.bn2 = nn.BatchNorm1d(num_features=self.linear2.out_features)
        self.linear3 = nn.Linear(in_features=self.linear2.out_features, out_features=input_size * 15)
        self.bn3 = nn.BatchNorm1d(num_features=self.linear3.out_features)
        self.linear4 = nn.Linear(in_features=self.linear3.out_features, out_features=input_size * 20)
        self.bn4 = nn.BatchNorm1d(num_features=self.linear4.out_features)
        self.linear5 = nn.Linear(in_features=self.linear4.out_features, out_features=output_size // 5)
        self.bn5 = nn.BatchNorm1d(num_features=self.linear5.out_features)
        self.linear6 = nn.Linear(in_features=self.linear5.out_features, out_features=output_size // 3)
        self.bn6 = nn.BatchNorm1d(num_features=self.linear6.out_features)
        self.linear7 = nn.Linear(in_features=self.linear6.out_features, out_features=output_size)
        self.bn7 = nn.BatchNorm1d(num_features=self.linear7.out_features)

    def forward(self, input):
        out = self.linear1(input)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = self.bn5(out)
        out = F.relu(out)
        out = self.linear6(out)
        out = self.bn6(out)
        out = F.relu(out)
        out = self.linear7(out)
        out = self.bn7(out)
        out = F.relu(out)
        out = F.log_softmax(out, dim=1)
        return out
