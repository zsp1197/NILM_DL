# Created by zhai at 2018/9/18
# Email: zsp1197@163.com
import sys, os

sys.path.append('...')
from visdom import Visdom
import Tools
import numpy as np

sourcesPath = '/mnt/hdd/zhai/python/NILM_DL/4paper/dlflex/losses_0.02'
# sourcesPath = '/mnt/hdd/zhai/python/NILM_DL/4paper/dlflex/losses'
losses_dict = {}
for fileName in os.listdir(sourcesPath):
    if (fileName.split('.')[-1] == 'list' and fileName.split('_')[0] == 'losses'):
        seq_len = int(fileName.split('_')[1].split('.')[0])
        # print(seq_len)
        losses_tuple = Tools.deserialize_object(os.path.join(sourcesPath, fileName))
        losses_dict.update({seq_len: losses_tuple})

totallosses_lst, classifi_losses_lst, od_losses_lst = [], [], []
legends = []
accepted_keys = [5, 15, 18, 20, 25]
cut_idx = 800
for key, losses_tuple in losses_dict.items():
    if (True):
    # if (key in accepted_keys):
        totallosses, classifi_losses, od_losses = np.array(losses_tuple)[:,0:cut_idx]
        totallosses_lst.append(totallosses)
        classifi_losses_lst.append(classifi_losses)
        od_losses_lst.append(od_losses)
        legends.append(f'win_size={str(key)}')
vis = Visdom()
# totallosses_lst, classifi_losses_lst, od_losses_lst = totallosses_lst[0:cut_idx], classifi_losses_lst[
#                                                                                   0:cut_idx], od_losses_lst[0:cut_idx]
vis.line(
         Y=np.column_stack(totallosses_lst),
         opts=dict(legend=legends, xlabel='iteration', ylabel='loss', xtickmin=0, ytype='log')
         )

vis.line(
    Y=np.column_stack(od_losses_lst),
    opts=dict(legend=legends, xlabel='iteration', ylabel='OD loss', xtickmin=0)
)

vis.line(
    Y=np.column_stack(classifi_losses_lst),
    opts=dict(legend=legends, xlabel='iteration', ylabel='SS loss', xtickmin=0)
)
