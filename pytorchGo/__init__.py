# Created by zhai at 2018/6/2
# Email: zsp1197@163.com
# use_cuda = False
import random

import torch

use_cuda = torch.cuda.is_available()
if (use_cuda):
    # device = torch.device("cuda:1")
    device = torch.device("cuda:{0}".format(random.choice(range(torch.cuda.device_count()))))
else:
    device = torch.device('cpu')
print('hi')