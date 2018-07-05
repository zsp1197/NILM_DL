# Created by zhai at 2018/6/19
# Email: zsp1197@163.com
import matplotlib
from visdom import Visdom

import Tools
from test.test_seq2Seq import TestSeq2Seq
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn')
t=TestSeq2Seq()
t.prepareCredients()
ps_togo=t.main_meter_ps[645109:645646]
Tools.server_ps_plot(ps_togo)
# ps_togo.plot()
# plt.ylim(0,2385)
# plt.xlim(ps_togo.index[0],ps_togo.index[-1])
# plt.xlabel('Time')
# plt.ylabel('Power (W)')
# plt.show()

breaks=[(0,93),(93,102),(101,134),(134,153),(152,315),(314,341),(341,403),(402,431)]
ps_togo=ps_togo[0:breaks[-1][1]]
for i,_break in enumerate(breaks):
    _ps=ps_togo[_break[0]:_break[1]]
    _ps.plot(label='bin {0}'.format(i + 1),linewidth=3)
plt.legend(fontsize='x-large')
plt.ylim(0,2385)
plt.xlim(ps_togo.index[0],ps_togo.index[-1])
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.show()