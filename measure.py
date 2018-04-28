# Created by zhai at 2018/4/27
# Email: zsp1197@163.com
from copy import deepcopy
from Tools import *
import pandas as pd
import numpy as np
import sys
import torch
use_gpu=torch.cuda.is_available()
def assign_label(todeal:np.array,labels:dict):
    '''
    labels has to be OrderedDict!!!!!
    :param todeal:np.array
    :param labels: {val:idx}
    :return: np.array([idx,idx....])
    '''
    dms=[]
    gpu=len(todeal)>30000 and use_gpu
    todeal=torch.Tensor(todeal).to('cuda' if gpu else 'cpu')
    for state_value,idx in labels.items():
        state_value=torch.Tensor([state_value]).to('cuda' if gpu else 'cpu')
        dms.append(torch.abs(todeal-state_value))
    dms=torch.stack(dms).to('cuda' if gpu else 'cpu')
    dms=dms.t()
    result=[]
    for i,dm_single in enumerate(dms):
        result.append(dm_single.argmin())
    result=torch.IntTensor(result).cpu().numpy()
    return result
