# Created by zhai at 2018/1/16
# Email: zsp1197@163.com
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Tools
from .Appliance_class import Appliance_state
import heapq


class State_r2(object):
    def __init__(self, states_tuple):
        self.states_tuple = states_tuple
        self.value = self.getValue()

    def getValue(self):
        value = 0
        if (self.states_tuple is None):
            return 0
        for state in self.states_tuple:
            value += state.center_value
        return value

    def state_r2_score(self, time_bin):
        states_tuple = self.states_tuple
        state_scores = []
        if(states_tuple is None):
            # 如果这个状态是全关闭状态，那它理应被考虑，给一个很大的分数
            return 999
        for state in states_tuple:
            state_scores.append(state.state_score(time_bin))
        return np.array(state_scores).mean()


class State_r3(object):
    def __init__(self, value):
        self.value = value

    def set_state_r2_list(self, state_r2_list,ifRefine=True):
        # self.__state_r2_list = state_r2_list
        if ifRefine:
            self.__state_r2_list = self.refine_states_r2(state_r2_list)
        else:
            self.__state_r2_list=state_r2_list

    def get_state_r2_list(self):
        return self.__state_r2_list

    def refine_states_r2(self, state_r2_list):
        import warnings
        if(len(state_r2_list)==0):
            warnings.warn("r3中的state_r2_list竟然是空的？？！！")
            return state_r2_list
        result = []
        for state_r2 in state_r2_list:
            min_distance = max(50, 0.1 * state_r2.value)
            if (abs(state_r2.value - self.value) <= min_distance):
                result.append(state_r2)
        if(len(result)==0): result=state_r2_list
        return result

    def state_r2_scores_4_bin(self, state_r2_list, time_bin):
        '''

        :param time_bin: (starttime(pd.timestamp),endtime(pd.timestamp), pd.Timedelta)
        :return:
        '''
        scores = []
        for state_r2 in state_r2_list:
            scores.append(state_r2.state_r2_score(time_bin))
        return scores

    def refine_states_r2_by_time(self, time_bin, num_of_r2=50):
        scores = self.state_r2_scores_4_bin(self.__state_r2_list, time_bin=time_bin)
        nlargest_move_duplicates = Tools.list_move_duplicates(heapq.nlargest(num_of_r2, scores))
        indexes = []
        for mem in nlargest_move_duplicates:
            theindexList = [i for i, var in enumerate(scores) if var == mem]
            indexes += theindexList
            state_r2_list = Tools.list_select_with_indexes(thelist=self.get_state_r2_list(), indexes=indexes)
        try:
            self.set_state_r2_list(state_r2_list=state_r2_list)
        except:
            self.set_state_r2_list(state_r2_list=[])
