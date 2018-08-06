# Created by zhai at 2018/6/25
import os
from time import time
from unittest import TestCase
from Parameters import Parameters
from datamarket.Data_store import Data_store
from datamarket.House import House
from datamarket.StateManager import DLStateManager
from datamarket.vidinference import Inference_House_Vid
from pytorchGo.Infer import Predictor, Infer
from pytorchGo.mlp import MLP
from pytorchGo.seq2seq import *
import numpy as np

from pytorchGo.models_lab import SingleCell, SeqEmbedTarget, Combine_MLP_SET, SeqEmbedTime, Combine_MLP_SETime

# Email: zsp1197@163.com
from test.test_seq2Seq import TestSeq2Seq


class TestInference_House_Vid(TestCase):
    def test_get_credients(self):
        self.credients()

    def credients(self):
        self.t = TestSeq2Seq()
        self.t.get_infer_4_vid()
        self.t.infer.get_predicted_targets()
        best_start = 4532
        best_end = 4981
        self.inference_House_Vid = Inference_House_Vid(house=self.t.house, label_series=self.t.label_series,
                                                       main_meter_ps=self.t.main_meter_ps,
                                                       label_series_bin=self.t.infer.target_bin,
                                                       predicted_label_series_bin=self.t.infer.predicted_targets[best_start:best_end],
                                                       inputs_bin=self.t.inputs_noRefine[best_start:best_end], dlsm=self.t.dlsm)

    def test_walk_appliances_consumption(self):
        self.credients()
        start=time()
        self.inference_House_Vid.walk_appliances_consumption()
        end=time()
        print('使用时间{}'.format(end-start))

    def test_reproduce_ps(self):
        self.credients()
        self.inference_House_Vid.walk_appliances_consumption()
        self.inference_House_Vid.reproduce_ps()

    def test_f_scores(self):
        self.credients()
        self.inference_House_Vid.walk_appliances_consumption()
        print(self.inference_House_Vid.f_scores())

    def test_save_dic_dic_ps(self):
        self.credients()
        self.inference_House_Vid.walk_appliances_consumption()
        self.inference_House_Vid.save_dic_dic_ps(self.inference_House_Vid.appliance_consumption_predicted, 'haolei')

    def test_proportion_total_energy_assigned(self):
        self.credients()
        self.inference_House_Vid.walk_appliances_consumption()
        print(self.inference_House_Vid.proportion_total_energy_assigned())

    def test_all_metrics(self):
        self.credients()
        start = time()
        self.inference_House_Vid.walk_appliances_consumption()
        end = time()
        print('使用时间{}'.format(end - start))
        print(self.inference_House_Vid.f_scores())
        print(self.inference_House_Vid.proportion_total_energy_assigned())

    def test_percentages_apps(self):
        self.credients()
        self.inference_House_Vid.walk_appliances_consumption()
        self.inference_House_Vid.percentages_apps()

    def test_yxf(self):
        path = '/home/uftp'
        file = 'forzspsparsehmm'
        # file='forzspsiqb30s'
        filepath = os.path.join(path, file)
        store = pd.HDFStore(filepath)
        print(store.keys())
        print(store['ptruth'].keys())

    def test_yxf_results(self):
        self.credients()
        house = self.inference_House_Vid.house
        # path = '/mnt/hdd/zhai/python/yxfGo/siqphmm'
        path = '/mnt/hdd/zhai/python/yxfGo/sparsehmm'
        # path = '/home/uftp'
        file,estkey = 'forzspsparsehmm','/pesti'
        # file, estkey = 'forzspsiqb30s', '/estim'
        # file, estkey = 'forzspsiqb30s', '/estim'
        print(file)
        store = pd.HDFStore(os.path.join(path, file))
        df = store[estkey]
        keys = list(df.keys())

        print(keys)
        instances = [(i.split('__')[0].lstrip('/'), (i.split('__'))[1]) for i in keys]
        print(instances)

        appliance_consumption_predicted = {}
        for appliane_state in house.state_r2_dict[list(house.state_r2_dict.keys())[0]].states_tuple:
            try:
                appliance_consumption_predicted[appliane_state.appliance_type]
            except:
                appliance_consumption_predicted.update({appliane_state.appliance_type: {}})
            appliance_consumption_predicted[appliane_state.appliance_type].update(
                {appliane_state.instance: pd.Series()})
        for i, key in enumerate(keys):
            app, instance = instances[i]
            appliance_consumption_predicted[app][instance] = df[key]
        print()
        self.inference_House_Vid.set_appliance_consumption_predicted(
            appliance_consumption_predicted=appliance_consumption_predicted)
        print(self.inference_House_Vid.f_scores())
        print(self.inference_House_Vid.proportion_total_energy_assigned())

    def test_get_data_folds(self):
        self.credients()
        startTime = pd.Timestamp('2011-05-03 17:33:18')
        endTime = pd.Timestamp('2011-05-08 13:01:29')
        mainMeter = self.inference_House_Vid.house.mainMeter
        appliance_pss_equal_length_dict = self.inference_House_Vid.house.appliance_pss_equal_length_dict
        instance_names = self.inference_House_Vid.house.instance_names
        mainMeter_cutted, mainMeter_remaining = Tools.cut_series(mainMeter, startTime, endTime, True)
        appliance_pss_cutted = {}
        appliance_pss_remaining = {}
        for (app, instance) in tqdm(instance_names, desc='cut ps'):
            # appliance_pss_cutted[app][instance], appliance_pss_remaining[app][instance] = Tools.cut_series(
            #     series=appliance_pss_equal_length_dict[app][instance], startTime=startTime, endTime=endTime,
            #     concat=True)
            appliance_pss_cutted[app + '__' + instance], appliance_pss_remaining[
                app + '__' + instance] = Tools.cut_series(
                series=appliance_pss_equal_length_dict[app][instance], startTime=startTime, endTime=endTime,
                concat=True)
        print()

    def test_refine_ori_store(self):
        self.credients()
        startTime = pd.Timestamp('2011-05-03 17:33:18')
        endTime = pd.Timestamp('2011-05-08 13:01:29')
        instance_names = self.inference_House_Vid.house.instance_names
        # (mainMeter_cutted, mainMeter_remaining)=Tools.deserialize_object('mainMeter_cut_remain')
        # appliance_pss_cutted=Tools.deserialize_object('appliance_pss_cutted')
        # appliance_pss_remaining=Tools.deserialize_object('appliance_pss_remaining')
        store_train = pd.HDFStore('/mnt/hdd/zhai/data/appliances_folds_train_2.h5')
        store_test = pd.HDFStore('/mnt/hdd/zhai/data/appliances_folds_test_2.h5')
        ss = pd.HDFStore('/mnt/hdd/zhai/data/appliances_2.h5')
        for key in ss.keys():
            store_test[key], store_train[key] = Tools.cut_series(series=ss[key], startTime=startTime, endTime=endTime,
                                                                 concat=True)
        store_test.close()
        store_train.close()