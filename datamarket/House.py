from copy import deepcopy

from conda._vendor.tqdm import tqdm

import Tools as zht
import pandas as pd
import numpy as np

from Parameters import Parameters
# from datamarket.Data_store import Data_store
from datamarket.Event_detection import Event_detection
from datamarket.Appliance_class import Appliance_state
from datamarket.Data_store import Data_store
from datamarket.StateManager import StateManager
from datamarket.feedState_r2 import getState_r2_list
from datamarket.superStates import State_r2
from measure import *
import math

'''
e.g.
path = 'allappliances.hdf5'
data_store = Data_store(path)
self.house = House(data_store=data_store, house_idx='1', parameters=Parameters())
self.house.aggregate()
gs = Good_sections(ps=self.house.mainMeter, parameters=Parameters())
for ps in gs.pss:
    chunks_list = self.house.sax(ps=ps)
    for i, chunk in enumerate(chunks_list):
        print(str(i))
        thelabel = self.house.labeling(chunk=chunk)
'''


class Chunk(object):
    def __init__(self, delta_value, total_value, start_time, delta_time):
        self.delta_value = delta_value
        self.total_value = total_value
        self.start_time = start_time
        self.delta_time = delta_time
        self.delta_value_2 = self.refine_delta_value(delta_value)
        self.total_value_2 = self.refine_total_value(total_value)
        self.start_time_2 = timestamp_2_per_of_day(start_time)
        self.delta_time_2 = self.refine_delta_time(delta_time)

    def refine_delta_time(self, delta_time: pd.Timedelta):
        seconds = math.log2(0.001 * delta_time.seconds + 1)
        return seconds

    def refine_total_value(self, total_value):
        return math.log2(0.01 * total_value + 1)

    def refine_delta_value(self, delata_value):
        return delata_value * 0.01


class House(object):
    # @zht.check_func_input_output_type_static
    def __init__(self, data_store: Data_store, house_idx: str, parameters: Parameters, check_good_sections=False,
                 debug=False):
        if (debug):
            if (parameters != None):
                self.parameters = parameters
            pass
        else:
            # house_idx is str! maybe can be named as zhai
            self.data_store = data_store
            self.house_idx = house_idx
            self.parameters = parameters
            self.get_instances_belong2House()
            if check_good_sections:
                self.instance_good_sections_dict = self.get_instance_sections()

    def get_states_list_sm_by_TIE(self, states_dict):
        self.states_dict_tie = states_dict
        self.states_list = []
        for appliance_name, instance in self.instance_names:
            try:
                centers = list(self.states_dict_tie[appliance_name][instance].keys())
            except:
                pass
            for center in centers:
                if (center < 3):
                    continue
                self.states_list.append(
                    Appliance_state(appliance_type=appliance_name, instance=instance, state_value=center,
                                    PROPERTIES=None,
                                    thedict=None, dataset='redd_tie'))
        # self.state_r2_list=getState_r2_list(self.states_list)
        # return self.states_list
        self.sm = StateManager(instance_names=self.instance_names, states_dict=self.states_dict_tie)

    def get_total_good_section(self):
        self.aggregate()
        self.gs = Good_sections(ps=self.mainMeter, parameters=self.parameters)
        pss = self.gs.pss

    @zht.check_func_input_output_type_static
    def labeling(self, chunk: Chunk):
        '''

        :param chunk:good_section chunk of total house power reading
        :return: tuple(appliance_name,instance_name)
        '''
        distances_list = []
        for mem in self.instance_names:
            appliance_name, instance_name = mem
            distances_list.append(self.instance_good_sections_dict[appliance_name][instance_name].labeling(chunk=chunk))
        theidx = distances_list.index(min(distances_list))
        return self.instance_names[theidx]

    def estimate_label_4_chunk_of_labels(self, start: pd.Timestamp, end: pd.Timestamp, label_ps: pd.Series):
        to_estimate_ps = label_ps[start:end]
        counts = np.bincount(to_estimate_ps.values)
        most_member = np.argmax(counts)
        # percentage = list(to_estimate_ps.values).count(most_member) / len(to_estimate_ps)
        try:
            self.percentage.append(list(to_estimate_ps.values).count(most_member) / len(to_estimate_ps))
        except:
            self.percentage = [list(to_estimate_ps.values).count(most_member) / len(to_estimate_ps)]
        return most_member

    def get_instances_belong2House(self):
        '''
        self.instance_names: [tuples(appliance_name,instance_name)]
        '''
        keys = self.data_store.keys
        self.house_keys_dict = {}
        self.instance_names = []
        for key in keys:
            appliance_name = key.split('/')[2]
            instance_name = key.split('/')[3]
            if (str(self.house_idx) == instance_name.split('-')[0]):
                self.instance_names.append((appliance_name, instance_name))
        self.instance_names = zht.list_move_duplicates(self.instance_names)

    def aggregate(self):
        '''
        add with resample to 1Hz
        :return:
        '''
        considered_appliances = self.parameters.considered_appliances
        virgin = True
        for appliance_instance_tuple in self.instance_names:
            appliance_name, instance_name = appliance_instance_tuple
            if ((considered_appliances == []) or (appliance_name in considered_appliances)):
                # toappend = self.fill_chunks(
                #     ps=self.data_store.get_instance_ps(appliance_name=appliance_name, instance=instance_name))
                try:
                    toappend = self.appliance_pss_equal_length_dict[appliance_name][instance_name]
                except:
                    toappend = self.fill_chunks(
                        ps=self.data_store.get_instance_ps(appliance_name=appliance_name, instance=instance_name))
                    print('建议先执行self.pss_equal_length，输出self.appliance_pss_equal_length_dict')
                if (virgin):
                    final_ps = deepcopy(toappend)
                    virgin = False
                else:
                    final_ps = final_ps.add(deepcopy(toappend), fill_value=0)
        self.mainMeter = final_ps.add(
            pd.Series(data=0, index=pd.date_range(start=min(final_ps.index), end=max(final_ps.index), freq='s')),
            fill_value=0)
        return self.mainMeter

    def walk_state_r2(self):
        '''
        first, get {1:pd.Series(index=times,data=[state_idx])}
        :return:
        '''
        appliance_state_idxes = {}
        try:
            self.mainMeter
        except:
            self.aggregate()
        startTime, endTime = min(self.mainMeter.index), max(self.mainMeter.index)
        self.state_r2_dict = {}
        for appliance_idx, appliance in tqdm(self.sm.appliance_encode_dict.items()):
            ps = self.appliance_pss_equal_length_dict[appliance.appliance_type][appliance.instance].add(
                pd.Series(index=pd.date_range(start=startTime, end=endTime, freq='s'), data=0), fill_value=0)
            appliance_state_idxes.update({appliance_idx: pd.Series(index=ps.index, data=assign_label(todeal=ps.data,
                                                                                                     labels=appliance.val_idx_dict))})

        mainMeter_labels = []
        num_of_states = [len(appliance.states_encoding) for appliance_idx, appliance in
                         self.sm.appliance_encode_dict.items()]
        for stamp in tqdm(pd.date_range(start=startTime, end=endTime, freq='s')):
            states = [idx_series[stamp] for appliance_idx, idx_series in appliance_state_idxes.items()]
            idx = encode_list(states=states, num_of_states=num_of_states)
            try:
                self.state_r2_dict[idx]
            except:
                # 因为电器编号是从1开始，所以此处要用i+1,注意，所有dict应该是OrderedDict,在python>=3.6可自动满足
                self.state_r2_dict.update({idx: State_r2(states_tuple=tuple(
                    [self.sm.appliance_encode_dict[i + 1].states_encoding[state_idx] for i, state_idx in
                     enumerate(states)]))})
            mainMeter_labels.append(idx)
        self.mainMeter_labels_ps = pd.Series(index=self.mainMeter.index, data=mainMeter_labels)

    def series2bins(self, series):
        '''

        :param series:
        :return:
        e.g.
        data = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5]
        series=pd.Series(data=[1,1,1,2,2,2,3,3,3,4,5,5],index=range(0,len(data)))
        print(self.series2bins(series))
        get: [(0, 2, 1), (3, 5, 2), (6, 8, 3), (9, 9, 4), (10, 11, 5)]
        '''
        result_list = []
        ps_index = series.index
        for i, var in enumerate(series.data):
            if (i == 0):
                temp = var
                last_i = 0
            elif (i == len(series) - 1):
                # 最后一个特殊处理
                theTuple = (ps_index[last_i], ps_index[i], temp)
                result_list.append(theTuple)
            else:
                if (temp != var):
                    theTuple = (ps_index[last_i], ps_index[i - 1], temp)
                    result_list.append(theTuple)
                    temp = var
                    last_i = i
        return result_list

    @zht.check_func_input_output_type_static
    def sax(self, ps: pd.Series):
        '''
        refine power series to sax description
        :param ps: pd.Series index:timestamp value: power readings
        :param max_T: max-time between samples
        :return: list[chunk]
        '''
        # 以前KNN聚类的方法
        ed = Event_detection(ps=ps, parameters=self.parameters)
        naive_description = ed.delta_based()
        # naive_description = ed.knn_based()
        result = []
        for i, thenaive in enumerate(naive_description):
            # if (i == 0):
            #     result.append(Chunk(thenaive[2], thenaive[2], thenaive[0], thenaive[1] - thenaive[0]))
            # else:
            #     result.append(Chunk(delta_value=
            #                         ps[thenaive[0]] - ps[naive_description[i - 1][1]], total_value=thenaive[2],
            #                         start_time=thenaive[0], delta_time=thenaive[1] - thenaive[0]))
            if (i == 0):
                result.append(
                    Chunk(delta_time=thenaive[1] - thenaive[0], delta_value=ps.values[0], total_value=ps.values[0],
                          start_time=thenaive[0]))
            else:
                try:
                    delta_value = ps[thenaive[0]] - ps[naive_description[i - 1][1]]
                except:
                    delta_value = ps[thenaive[0]] - ps[naive_description[i][0]]
                result.append(Chunk(delta_value=
                                    delta_value,
                                    total_value=ps[thenaive[0]:thenaive[1]].mean(),
                                    start_time=thenaive[0], delta_time=thenaive[1] - thenaive[0]))
        return result

    def dl_bins_inputs_targets(self, main_meter_ps: pd.Series, main_meter_labels_ps: pd.Series):
        '''
        # TODO important notes
        :param main_meter_ps:
        :return: [(start_time,delta_time,total_value,delta_value)]
        '''
        chunks_list = self.sax(ps=main_meter_ps)
        self.chunks_list = chunks_list
        inputs = [(chunk.start_time_2, chunk.delta_time_2, chunk.total_value_2, chunk.delta_value_2) for chunk in
                  chunks_list]
        targets = [
            self.estimate_label_4_chunk_of_labels(start=chunk.start_time, end=chunk.start_time + chunk.delta_time,
                                                  label_ps=main_meter_labels_ps) for chunk in chunks_list]
        return inputs, targets

    def dl_bins_noRefine(self):
        try:
            self.chunks_list
        except:
            raise ValueError('没找到chunks_list，先执行dl_bins_inputs_targets')
        inputs = [(chunk.start_time, chunk.delta_time, chunk.total_value, chunk.delta_value) for chunk in
                  self.chunks_list]
        return inputs

    def pss_equal_length(self):
        '''

        :return: added self.instance_good_sections_dict={appliance_name:{instance:Good_section}}
                 added self.appliance_pss_equal_length_dict={appliance_name:{instance:pd.Series(1Hz)}}
        '''
        try:
            self.instance_good_sections_dict
        except:
            self.instance_good_sections_dict = self.get_instance_sections()
        self.appliance_pss_equal_length_dict = {}
        for appliance_name, instance_dict in self.instance_good_sections_dict.items():
            appliance_dict = {}
            for instance, gs in instance_dict.items():
                instance_ps = pd.Series()
                for ps in gs.pss:
                    instance_ps = instance_ps.add(ps, fill_value=0)
                instance_ps = pd.Series(data=0,
                                        index=pd.date_range(start=min(instance_ps.index),
                                                            end=max(instance_ps.index))).add(instance_ps, fill_value=0)
                appliance_dict.update({instance: instance_ps})
            self.appliance_pss_equal_length_dict.update({appliance_name: appliance_dict})

    def get_good_pss(self, ps: pd.Series):
        '''
        ru ti
        :param ps: pd.Series index:timestamp value: power readings
        :param max_T: max-time between samples
        :return: good_sections
        '''
        gs = Good_sections(ps=ps, parameters=self.parameters)
        # gs_time_chunks = gs.get_chunks_index()
        # pss = [zht.ps_between_timestamp(ps, chunk[0], chunk[1]) for chunk in gs_time_chunks]
        return gs.pss

    @zht.check_func_input_output_type_static
    def fill_chunks(self, ps=pd.Series) -> pd.Series:
        pss = self.get_good_pss(ps=ps)
        pss = [zht.up_sample_ps(ps=theps) for theps in pss]
        return zht.ps_concatenate(pss=pss)

    def get_instance_sections(self):
        '''

        :return:{appliance_name:{instance_name:Good_sections}}
        '''
        # [tuples(appliance_name, instance_name)]
        instance_names = self.instance_names
        appliance_names = zht.list_move_duplicates([mem[0] for mem in instance_names])
        result = {}
        for appliance_name in appliance_names:
            result[appliance_name] = {}
        for appliance_name, instance_name in instance_names:
            result[appliance_name][instance_name] = Good_sections(
                ps=self.data_store.get_instance_ps(appliance_name=appliance_name, instance=instance_name),
                parameters=self.parameters, appliance_name=appliance_name, instance_name=instance_name)
        return result


class Good_sections(object):
    @zht.check_func_input_output_type_static
    def __init__(self, ps: pd.Series, parameters: Parameters, appliance_name=None, instance_name=None):
        # 每个时间只有一个数
        assert len(ps.index.unique()) == len(ps)
        self.ps = ps
        self.parameters = parameters
        self.chunks_index = self.get_chunks_index()
        self.pss = self.get_pss()
        self.appliance_name = appliance_name
        self.instance_name = instance_name

    def get_chunks_index(self):
        '''
        :return:[tuples(start_time,end_time)]
        '''
        result = []
        indexes = list(self.ps.index)
        deltas = np.diff(indexes)
        start = indexes[0]
        for i, delta in enumerate(deltas):
            if (i == len(deltas) - 3):
                result.append((start, indexes[-1]))
            if (delta > self.parameters.max_T):
                result.append((start, indexes[i]))
                start = indexes[i + 1]
        return result

    def get_pss(self):
        pss = [zht.up_sample_ps(self.ps[chunk[0]: chunk[1]]) for chunk in self.chunks_index]
        # pss=[]
        # for chunk in self.chunks_index:
        #
        #     pss.append(zht.ps_between_timestamp(self.ps, chunk[0], chunk[1]))
        self.ps = zht.ps_concatenate(pss=pss)
        return pss

    @zht.check_func_input_output_type_static
    def labeling(self, chunk: Chunk):
        '''
        find out is the chunk from the instance
        :param chunk:
        :return: bool
        '''

        def similarity(delta_value):
            ps = zht.ps_between_timestamp(ps=self.ps, start=start_time - self.parameters.labeling_window_size / 2,
                                          end=start_time + self.parameters.labeling_window_size / 2)
            if ((ps.values[-1] - ps.values[0]) * delta_value <= 0):
                # 增长趋势相反
                return self.parameters.penalty
            else:
                return abs(ps.values[-1] - ps.values[0] - delta_value)

        start_time = chunk.start_time
        delta_value = chunk.delta_value
        for chunk_index in self.chunks_index:
            chunk_start_time, chunk_end_time = chunk_index
            if (chunk_start_time < start_time < chunk_end_time):
                return similarity(delta_value)
        return self.parameters.penalty


def feed_Houses(data_store, houses_idx=None):
    houses = []
    if (houses_idx == None):
        houses_idx = []
        for key in data_store.keys:
            houses_idx.append(key.split('/')[3].split('-')[0])
        houses_idx = tuple(zht.list_move_duplicates(houses_idx))
    for house_idx in houses_idx:
        houses.append(
            House(data_store=data_store, house_idx=house_idx, parameters=Parameters(), check_good_sections=True))
    return houses
