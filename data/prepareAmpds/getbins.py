# Created by zhai at 2018/9/22
# Email: zsp1197@163.com
from copy import deepcopy

import Tools
import pandas as pd

from Parameters import Parameters
from datamarket.House import House
store=pd.HDFStore('ampds.hdf5')
df=store['/submeters']
apps=list(df.columns)
print(apps)
# vid apps
paras=Parameters()

states_dict={'CDE':[4700,0], 'CWE':[200,500,900,0], 'DWE':[760,0], 'EQE':[30,40,0], 'FGE':[135,480,0], 'GRE':[120,500,800,0], 'HPE':[1800,2400,0]}
house=House(None,None,parameters=paras,debug=True)
super_SS,super_SS_idxes,ss_naive_idxes_dict,ss_idxes_dict,ss_idxes_value_dict=Tools.deserialize_object('ampds_ss.tuple')
mainMeter=deepcopy(store['/mainMeter'])
store.close()
labels=[ss_naive_idxes_dict[ss] for ss in super_SS]
main_meter_labels_ps=pd.Series(index=mainMeter.index,data=labels)
inputs,targets=house.dl_bins_inputs_targets(mainMeter,main_meter_labels_ps)

Tools.serialize_object((inputs,targets),f'ampds_inputs_targets_{paras.delta_detection}')