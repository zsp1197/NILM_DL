from copy import deepcopy

import zhai_tools.Tools as zht
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Parameters import Parameters
from datamarket.Data_store import Data_store
from datamarket.Event_detection import Event_detection

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


class House(object):
    @zht.check_func_input_output_type_static
    def __init__(self, data_store: Data_store, house_idx: str, parameters: Parameters,check_good_sections=True):
        # house_idx is str! maybe can be named as zhai
        self.data_store = data_store
        self.house_idx = house_idx
        self.parameters = parameters
        self.get_instances_belong2House()
        if check_good_sections:
            self.instance_good_sections_dict = self.get_instance_sections()

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
                toappend = self.fill_chunks(
                    ps=self.data_store.get_instance_ps(appliance_name=appliance_name, instance=instance_name))

                toappend.plot(label=appliance_name)

                if (virgin):
                    final_ps = deepcopy(toappend)
                    virgin = False
                else:
                    final_ps = final_ps.add(deepcopy(toappend), fill_value=0)
        self.mainMeter = final_ps
        return self.mainMeter

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
                result.append(Chunk(delta_value=
                                    ps[thenaive[0]] - ps[naive_description[i - 1][1]],
                                    total_value=ps[thenaive[0]:thenaive[1]].mean(),
                                    start_time=thenaive[0], delta_time=thenaive[1] - thenaive[0]))
        return result

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
        pss = [zht.up_sample_ps(zht.ps_between_timestamp(self.ps, chunk[0], chunk[1])) for chunk in self.chunks_index]
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


def feed_Houses(data_store):
    houses_idx = []
    houses = []
    for key in data_store.keys:
        houses_idx.append(key.split('/')[3].split('-')[0])
    houses_idx = tuple(zht.list_move_duplicates(houses_idx))
    for house_idx in houses_idx:
        houses.append(House(data_store=data_store, house_idx=house_idx, parameters=Parameters()))
    return houses
