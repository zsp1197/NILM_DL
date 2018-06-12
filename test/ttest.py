# Created by zhai at 2018/6/1
# Email: zsp1197@163.com
import torch
import torch.nn.functional as F
from test.test_seq2Seq import TestSeq2Seq
import numpy as np
from visdom import Visdom
t=TestSeq2Seq()
t.test_fit_batch_combine()