# Created by zhai at 2018/9/20
# Email: zsp1197@163.com
import pandas as pd
from datetime import datetime
import numpy as np
import Tools
csvfile='/mnt/hdd/zhai/python/NILM_DL/data/ampds2_Electricity_P.csv'
df=pd.read_csv(csvfile)
print(datetime.utcfromtimestamp(df['UNIX_TS'][0]))
print(datetime.utcfromtimestamp(df['UNIX_TS'][len(df)-2]))
print(df.columns)

apps=['CDE','CWE','DWE','EQE','FGE','GRE','HPE']
timeindex=[pd.Timestamp.utcfromtimestamp(i) for i in df['UNIX_TS']]
df.index=timeindex
df.pop('UNIX_TS')
df2=pd.DataFrame(index=timeindex)
mainMeter=pd.Series()
for app in apps:
    ps=df[app]
    mainMeter=mainMeter.add(ps,fill_value=0)
    df2[app]=ps

#     Tools.server_ps_plot(ps[::10],label=app)
store=pd.HDFStore('ampds.hdf5')
store['/submeters']=df2
store['/mainMeter']=mainMeter
store.close()
store=pd.HDFStore('ampds.hdf5')
pass