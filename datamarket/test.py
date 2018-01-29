import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a=pd.Series(data=[1,2,3,4,5,6],index=[1,2,3,4,5,6])
b=pd.Series(data=[-3,-4,-5,-6],index=[3,4,5,8])
c=a+b
# print(c)
# print(a.add(b,fill_value=0))

rng1 = pd.date_range('1/1/2011', periods=4, freq='H')
rng2 = pd.date_range('1/1/2011', periods=4, freq='H')
ts1 = pd.Series(3+np.random.randn(len(rng1)), index=rng1)
ts2 = pd.Series(100+np.random.randn(len(rng2)), index=rng2)
ts=ts1.add(ts2,fill_value=0)

# print(len(ts))
# print(len(ts.index.unique()))
#
# print(ts.index.unique())
#
# ax=plt.gca()
# ax.annotate("Annotation",
#             xy=(1, 10), xycoords='data',
#             xytext=(2, 15), arrowprops=dict(facecolor='black', shrink=0.05)
#             )
# plt.plot([1,100],[1,100])
# plt.xlim([0,10])
# plt.show()

# print(ts)
# ts=ts.diff()
# ts[ts.index[0]]=999
ts[ts.index[1]]=-59
event=ts[abs(ts)<102]
for i,(index,value) in enumerate(event.iteritems()):
    print(i)
    print(index)
    print(value)
print(ts)
print(ts.mean())