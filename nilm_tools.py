# Created by zhai at 2018/8/25
# Email: zsp1197@163.com
import numpy as np
def decode_delta_time(delta:np.array):
    return (2**delta-1)*1000

def decode_time_of_day(st:np.array):
    sec_of_day=st*(3600*24)
    return sec_of_day