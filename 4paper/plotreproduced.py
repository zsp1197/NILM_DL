# Created by zhai at 2018/7/1
# Email: zsp1197@163.com
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn')
store=pd.HDFStore('reproduce_ps.h5')
print(store.keys())
pre=store['predicted'][2500:6500]
truth=store['truth'][2500:6500]
truth.plot(label='Ground Truth',color='orange')
pre.plot(label='Estimated')
plt.xlim((pre.index[0],pre.index[-1]))
plt.legend()
plt.ylabel('Power (W)')
plt.xlabel('Time')
plt.show()