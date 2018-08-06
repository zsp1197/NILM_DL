# Created by zhai at 2018/7/1
# Email: zsp1197@163.com
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

import Tools

matplotlib.style.use('seaborn')
store=pd.HDFStore('reproduce_ps_2.h5')
# store=pd.HDFStore('reproduce_ps.h5')
print(store.keys())


# pre=store['predicted']
# truth=store['truth']
pre=store['predicted'][2500:6500]
truth=store['mainMeter'][2500:6500]

apps=[]
def getLabel(key):
    app=key.split('__')[0]
    if(app in apps):
        result=app+'_'+str(apps.count(app)+1)
    else:
        result=app
    apps.append(app)
    return result

plots=[]
for key in list(store.keys()):
    _key=key.lstrip('/')
    if(_key in ('predicted','mainMeter')):
        continue
    else:
        # plots.append(store[_key])
        store[_key].plot(label=getLabel(_key))

# Tools.server_pss_plot([pre,truth])
# Tools.server_pss_plot(plots)
truth.plot(label='Ground Truth Total',color='orange')
pre.plot(label='Estimated Total')
plt.xlim((pre.index[0],pre.index[-1]))
plt.legend()
plt.ylabel('Power (W)')
plt.xlabel('Time')
plt.show()