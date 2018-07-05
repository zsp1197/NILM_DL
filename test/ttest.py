# Created by zhai at 2018/6/1
# Email: zsp1197@163.com
import torch
import torch.nn.functional as F
from test.test_seq2Seq import TestSeq2Seq
import numpy as np
from visdom import Visdom
import pandas as pd
path='/mnt/hdd/zhai/data/appliances.h5'
store=pd.HDFStore(path)
newstore=pd.HDFStore('/mnt/hdd/zhai/data/appliances_2.h5')
print(store.keys())
main_key='/mainMeter'
main_meter_index=store[main_key].index
for key in store.keys():
    if(key==main_key):
        continue
    else:
        newstore[key]=store[key].add(pd.Series(data=0,index=main_meter_index),fill_value=0)
newstore[main_key]=store[main_key]
print(newstore.keys())