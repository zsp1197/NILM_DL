import traceback
import warnings

import pandas as pd
import numpy as np
import math
import Tools
from .Distribution import Distribution
from .Return_class import *
from Tools import *


class Appliance_state(object):
    '''
    appliance state
    '''

    # @check_func_input_output_type_static
    def __init__(self, appliance_type: str, instance: str, state_value, PROPERTIES=None, thedict=None,
                 dataset='redd'):
        self.appliance_type = appliance_type
        self.instance = instance
        self.dataset = dataset
        self.PROPERTIES = PROPERTIES
        self.thedict = thedict
        self.center_value(state_value)
        self.id = appliance_type + '.' + instance + '.' + str(state_value)

    def center_value(self, center_value):
        self.center_value = center_value

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def feed(self, used_properties='all'):
        if (used_properties == 'all'):
            used_properties = self.PROPERTIES
        else:
            for i in used_properties:
                if (i not in self.PROPERTIES):
                    raise ValueError('no such property ' + i)
        for property in used_properties:
            try:
                self.thedict[property]
            except:
                raise ValueError("no such property in the given dict")
            if (property == 'start_time'):
                self._start_time = self.thedict[property]
            elif (property == 'end_time'):
                self._end_time = self.thedict[property]
            elif (property == 'power_value'):
                self._power_value = self.thedict[property]
            elif (property == 'delta_time'):
                self._delta_time = self.thedict[property]
            else:
                raise ValueError('undefined property!')

    @check_func_input_output_type_static
    def get_property(self, property: str) -> Return_class:
        # return Return_class(0)
        return Return_class(self.thedict[property])

    def feed2distribution(self):
        try:
            instance_dict = deepcopy(self.thedict[self.appliance_type][self.instance][self.center_value])
            for key in instance_dict:
                instance_dict[key] = Distribution(args=instance_dict[key])
            # TODO 为了应付空distribution的权宜之计
            if (len(instance_dict) >= 2):
                self.distributions = instance_dict
            else:
                self.distributions = None
        except:
            self.distributions = None
            print("分布提取错误在 " + self.appliance_type + ' ' + self.instance + ' ' + str(self.center_value))

    def state_score(self, time_bin):
        starttime, endtime, deltatime = time_bin
        starttime = Tools.timestamp_2_location_of_day(starttime)
        endtime = Tools.timestamp_2_location_of_day(endtime)
        deltatime = Tools.timedelta_2_naive(deltatime)
        try:
            score = self.distributions['start_time'].cdf(starttime) * (1 - self.distributions['end_time'].cdf(endtime))
        except:
            print('没有这个' + self.appliance_type + ' ' + str(self.center_value))
            return 0
        if (math.isnan(score)):
            print()
        return score


class Appliance_class():
    '''
    contains all instances of a single kind of appliance
    '''
    APPLIANCE_CANDIDATE_LIST = ('light', 'washer dryer', 'electric furnace', 'microwave',
                                'CE appliance', 'waste disposal unit', 'smoke alarm', 'fridge', 'electric stove',
                                'dish washer', 'electric space heater')

    # APPLIANCE_CANDIDATE_LIST =('fridge', 'light', 'microwave', 'dish washer', 'electric stove')

    @check_func_input_output_type_static
    def __init__(self, appliance_type: str, bigdict=None, dataset: str = 'redd',
                 PROPERTIES: tuple = ('power_value', 'start_time', 'end_time', 'delta_time')):
        assert appliance_type in self.APPLIANCE_CANDIDATE_LIST
        self.appliance_type = appliance_type
        self.dataset = dataset
        self.PROPERTIES = PROPERTIES
        self.bigdict = bigdict
        self.feed_appliance(bigdict)

    @check_func_input_output_type_static
    def feed_appliance(self, appliance_dict: dict, contain_off: bool = False):
        '''
        given a dict contains all instance of a appliance, refine the appliance to appliance_state_list
        :param appliance_dict:
        :param contain_off: if False, ignore off-state
        :return:
        '''
        self.appliance_state_list = []
        for instance, instance_dict in appliance_dict.items():
            if (instance_dict == None):
                continue
            for state, state_dict in instance_dict.items():
                if (not contain_off):
                    # 如果不包括off状态，则剔除最小的那个
                    if ((state == min(instance_dict.keys())) & (state < 10)):
                        continue
                appliance_state = self.feed_appliance_state(instance, state)
                # appliance_state.center_value(state)
                self.appliance_state_list.append(appliance_state)

    @check_func_input_output_type_static
    def feed_appliance_state(self, instance: str, state) -> Appliance_state:
        appliance_state = Appliance_state(appliance_type=self.appliance_type, thedict=self.bigdict[instance][state],
                                          instance=instance, state_value=state, PROPERTIES=self.PROPERTIES,
                                          dataset=self.dataset)
        return appliance_state

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Appliance(object):
    '''
    this class reltaes to a single applaince with multiple appliance_states
    '''

    def __init__(self, appliance_type: str, instance: str):
        self.appliance_type = appliance_type
        self.instance = instance

    @check_func_input_output_type_static
    def feed_appliance_state(self, state_value, thedict) -> Appliance_state:
        appliance_state = Appliance_state(appliance_type=self.appliance_type, thedict=thedict,
                                          instance=self.instance, state_value=state_value)
        return appliance_state

    def feed_appliance_state_r1_encoding_one_by_one(self, state_value, thedict):
        try:
            self.states_encoding
        except:
            self.states_encoding = {}
            self.val_idx_dict = {}
        idx=len(self.states_encoding)
        self.states_encoding.update(
            {idx: self.feed_appliance_state(state_value, thedict=thedict)})
        self.val_idx_dict.update({state_value:idx})