# Created by zhai at 2018/9/22
# Email: zsp1197@163.com
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import numpy as np
import Tools
from measure import assign_label, encode_list

states_dict={'CDE':[4700,0], 'CWE':[200,500,900,0], 'DWE':[760,0], 'EQE':[30,40,0], 'FGE':[135,480,0], 'GRE':[120,500,800,0], 'HPE':[1800,2400,0]}
store=pd.HDFStore('centers_idx_df.store')
df=store['/centers_idx_df']
apps=list(df.columns)
print(apps)

num_of_states=[len(states_dict[i]) for i in apps]
print(num_of_states)
super_SS=[]
super_SS_idxes=[]
ss_naive_idxes_dict={}
ss_idxes_dict={}
ss_idxes_value_dict={}

num=0
for idx in tqdm(df.index):
    state_idxes=list(df.ix[idx])
    encoded=encode_list(states=state_idxes, num_of_states=num_of_states)
    super_SS.append(encoded)
    if(encoded in list(ss_naive_idxes_dict.keys())):
        continue
    else:
        ss_naive_idxes_dict.update({encoded:num})
        ss_idxes_dict.update({num:encoded})
        ss_idxes_value_dict.update({num:sum([states_dict[apps[i]][value] for i,value in enumerate(state_idxes)])})
        num+=1
    super_SS_idxes.append(ss_naive_idxes_dict[encoded])
Tools.serialize_object((super_SS,super_SS_idxes,ss_naive_idxes_dict,ss_idxes_dict,ss_idxes_value_dict),'ampds_ss.tuple')


