# Created by zhai at 2018/6/25
import os
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
        self.inference_House_Vid = Inference_House_Vid(house=self.t.house, label_series=self.t.label_series,
                                                       main_meter_ps=self.t.main_meter_ps,
                                                       label_series_bin=self.t.infer.target_bin,
                                                       predicted_label_series_bin=self.t.infer.predicted_targets,
                                                       inputs_bin=self.t.inputs_noRefine, dlsm=self.t.dlsm)

    def test_walk_appliances_consumption(self):
        self.credients()
        self.inference_House_Vid.walk_appliances_consumption()

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
        path = '/home/uftp'
        # file,estkey = 'forzspsparsehmm','/pesti'
        file,estkey = 'forzspsiqb30s','/estim'
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
        print(self.inference_House_Vid.proportion_total_energy_assigned())
        print(self.inference_House_Vid.f_scores())