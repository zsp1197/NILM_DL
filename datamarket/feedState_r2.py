# Created by zhai at 2018/1/16
# Email: zsp1197@163.com
# from readData.readStates import *
import itertools
import matplotlib.pyplot as plt
# from superStates import State_r2
from datamarket.superStates import *


def getState_r2_list(states_list):
    states_list = states_list
    num_of_states = len(states_list)
    state_r2_list = [State_r2(None)]

    def checkDupicates(states_tuple):
        names = [state.appliance_type + '_' + state.instance for state in states_tuple]
        if (len(names) == len(list(set(names)))):
            return False
        else:
            return True

    for num_of_combination in range(1, num_of_states + 1):
        for states_tuple in itertools.combinations(states_list, num_of_combination):
            if (checkDupicates(states_tuple)):
                continue
            state_r2_list.append(State_r2(states_tuple=states_tuple))
    return state_r2_list


# def getStates_list(thepath):
#     states_dict = getUserStates(thepath=thepath)
#     states_list = feed_states(states_dict)
#     return states_list
#
# # thepath='D:\SJTU\湖北项目\数据/ori/xusuqian'
# # getState_r2_list(thepath)
# # state_r2_values=[i.value for i in state_r2_list]
