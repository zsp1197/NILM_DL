# Created by zhai at 2018/9/20
# Email: zsp1197@163.com
from copy import deepcopy

import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm

import Tools
from measure import assign_label

store=pd.HDFStore('ampds.hdf5')
df=store['/submeters']
apps=list(df.columns)
print(apps)
# vid apps

states_dict={'CDE':[4700,0], 'CWE':[200,500,900,0], 'DWE':[760,0], 'EQE':[30,40,0], 'FGE':[135,480,0], 'GRE':[120,500,800,0], 'HPE':[1800,2400,0]}
for idx in df.index:
    pass
centers_val_dict={}
centers_idx_dict={}
for app in apps:
    centers_list=states_dict[app]
    app_val={}
    app_idx={}
    num=0
    for center in centers_list:
        app_val.update({center:num})
        app_idx.update({num:center})
        num+=1
    centers_val_dict.update({app:app_val})
    centers_idx_dict.update({app:app_idx})
print(centers_val_dict)
print(centers_idx_dict)

centers_idx_df=deepcopy(df)
for app in tqdm(apps):
    centers_idx_df[app]=assign_label(todeal=df[app],labels=centers_val_dict[app])

print(centers_idx_df)

store2=pd.HDFStore('centers_idx_df.store')
store2['/centers_idx_df']=centers_idx_df
store2.close()