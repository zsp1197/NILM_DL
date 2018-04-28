from copy import deepcopy

import Tools as zht
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Parameters import Parameters
# import Data_store

class Event_detection(object):
    @zht.check_func_input_output_type_static
    def __init__(self, ps:pd.Series, parameters:Parameters):
        self.ps=ps
        self.parameters=parameters
        self.ps_diff=deepcopy(ps).diff()
        self.ps_diff[self.ps.index[0]]=self.parameters.penalty

    def delta_based(self):
        '''
        return: list of tuples (starttime(pd.timestamp),endtime(pd.timestamp), delta_value)
        '''
        result=[]
        event_series=self.ps_diff[abs(self.ps_diff)>=self.parameters.delta_detection]
        for i, (index, value) in enumerate(event_series.iteritems()):
            if(len(event_series)==1):
                result.append((index, self.ps.index[-1], self.ps.values[0]))
                break
            try:
                if(i==0):
                    result.append((index,event_series.index[i+1]-self.parameters.sample_T,self.ps.values[0]))
                elif(i==len(event_series)-1):
                    result.append((index,self.ps.index[-1],value))
                else:
                    result.append((index,event_series.index[i+1]-self.parameters.sample_T,value))
            except:
                print(event_series)
                print('error in delta_based')
                return [(self.ps.index[0],self.ps.index[-1],self.ps.values[0])]
        return result


    def knn_based(self):
        '''
        return: list of tuples (starttime(pd.timestamp),endtime(pd.timestamp), cluster center value)
        '''
        return zht.ps2description(ps=self.ps, centers=self.parameters.sax_steps)