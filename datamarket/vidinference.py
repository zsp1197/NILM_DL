# Created by zhai at 2018/6/25
# Email: zsp1197@163.com
from multiprocessing.pool import Pool

from sklearn.metrics import f1_score
from tqdm import tqdm

tqdm.monitor_interval = 0
import multiprocessing
import Tools
from datamarket.House import feed_Houses
from datamarket.NILMDataset import NILMDataset
import pandas as pd
import numpy as np


class Inference_House_Vid():
    def __init__(self, house, label_series, inputs_bin, main_meter_ps, label_series_bin, predicted_label_series_bin,
                 dlsm):
        # self.house=house
        self.house = Tools.deserialize_object('house.store')
        # self.house=self.prepareHouse('1')
        self.inputs_bin = inputs_bin
        self.label_series = label_series
        self.main_meter_ps = main_meter_ps
        self.label_series_bin = label_series_bin
        self.dlsm = dlsm
        self.predicted_label_series_bin = predicted_label_series_bin

    def prepareHouse(self, house_idx='1'):
        try:
            nilm_dataset = NILMDataset('../data/allappliances4.hdf5', '../data/allappliancesdict_refine4')
        except:
            nilm_dataset = NILMDataset('../data/temp/allappliances4.hdf5', '../data/temp/allappliancesdict_refine4')
        house = feed_Houses(nilm_dataset.datastore, (house_idx,))[0]
        house.get_states_list_sm_by_TIE(nilm_dataset.states_dict)
        house.pss_equal_length()
        house.aggregate()
        house.walk_state_r2()
        return house

    def walk_appliances_consumption(self):
        # 获取所有电器的估计功率消耗series
        result = {}
        for appliane_state in self.house.state_r2_dict[list(self.house.state_r2_dict.keys())[0]].states_tuple:
            try:
                result[appliane_state.appliance_type]
            except:
                result.update({appliane_state.appliance_type: {}})
            result[appliane_state.appliance_type].update({appliane_state.instance: pd.Series()})
        self.appliance_consumption_predicted = result
        # self.load_appliance_consumption_predicted('dic_dic_ps',self.house.instance_names)
        try:
            self.load_appliance_consumption_predicted('dic_dic_ps', self.house.instance_names)
            print('用已经预处理好的predicted')
        except:
            for input_bin, target_ss in tqdm(zip(self.inputs_bin, self.predicted_label_series_bin),
                                             desc='appliance consumption'):
                self.deal_time_instance(input_bin, target_ss)
            try:
                self.save_dic_dic_ps(self.appliance_consumption_predicted)
            except:
                print('保存失败')
        print()

    def set_appliance_consumption_predicted(self,appliance_consumption_predicted):
        self.appliance_consumption_predicted=appliance_consumption_predicted

    def reproduce_ps(self):
        r2_dict = self.house.state_r2_dict
        predicted_total = pd.Series()
        for app, instance in self.house.instance_names:
            instance_ps = self.appliance_consumption_predicted[app][instance]
            predicted_total = predicted_total.add(instance_ps, fill_value=0)
        Tools.server_pss_plot([predicted_total, self.main_meter_ps])

    def f_scores(self):
        scores_dict = {}
        for appliane_state in self.house.state_r2_dict[list(self.house.state_r2_dict.keys())[0]].states_tuple:
            try:
                scores_dict[appliane_state.appliance_type]
            except:
                scores_dict.update({appliane_state.appliance_type: {}})
            scores_dict[appliane_state.appliance_type].update({appliane_state.instance: pd.Series()})
        for app, instance in self.house.instance_names:
            scores_dict[app][instance] = self.tp_fn_fp_tn_4ps(self.appliance_consumption_predicted[app][instance],
                                                         self.house.appliance_pss_equal_length_dict[app][instance],
                                                         10)
        tp_total,fn_total,fp_total,tn_total=0,0,0,0
        f1_macro=0
        for app, instance in self.house.instance_names:
            tp, fn, fp, tn=scores_dict[app][instance]
            tp_total+=tp
            fn_total+=fn
            fp_total+=fp
            tn_total+=tn
        def f1(_tp,_fp,_fn):
            return 2*_tp/(2*_tp+_fp+_fn)
        f1_micro=f1(tp_total,fp_total,fn_total)
        for app, instance in self.house.instance_names:
            tp, fn, fp, tn=scores_dict[app][instance]
            try:
                f1_macro+=f1(tp,fp,fn)
            except:
                print('ERROR in f1')
        f1_macro=f1_macro/len(self.house.instance_names)
        return f1_micro,f1_macro

    def deal_time_instance(self, input_bin, target_ss):
        start_time = input_bin[0]
        delta_time = input_bin[1]
        real_num = self.dlsm.idx2stateidx[target_ss]
        for appliane_state in self.house.state_r2_dict[real_num].states_tuple:
            # self.appliance_consumption_predicted[appliane_state.appliance_type][appliane_state.instance] = \
            # self.appliance_consumption_predicted[appliane_state.appliance_type][
            #     appliane_state.instance].add(pd.Series(
            #     index=pd.date_range(start=start_time, end=start_time + delta_time, freq='S'),
            #     data=appliane_state.center_value), fill_value=0)
            self.appliance_consumption_predicted[appliane_state.appliance_type][appliane_state.instance] = pd.concat([
                self.appliance_consumption_predicted[appliane_state.appliance_type][
                    appliane_state.instance], pd.Series(
                    index=pd.date_range(start=start_time, end=start_time + delta_time, freq='S'),
                    data=appliane_state.center_value)])

    def save_dic_dic_ps(self, dic_dic_ps, name='dic_dic_ps'):
        store = pd.HDFStore(name)
        store['\dict'] = pd.DataFrame.from_dict(dic_dic_ps)
        store.close()

    def load_appliance_consumption_predicted(self, name, keys):
        store = pd.HDFStore(name)
        dic_dic_ps_df = store['\dict']
        for appliance_type, instance in keys:
            #     data=appliane_state.center_value), fill_value=0)
            self.appliance_consumption_predicted[appliance_type][instance] = dic_dic_ps_df[appliance_type][instance]

    def proportion_total_energy_assigned(self):
        truth_dict = self.house.appliance_pss_equal_length_dict
        predicted_dict = self.appliance_consumption_predicted
        app_instances = self.house.instance_names
        timestamps = self.main_meter_ps.index
        # 1-a/(2b)
        a, b = 0, 0
        for t in tqdm(timestamps, desc='proportion_total_energy_assigned'):
            for app, instance in app_instances:
                try:
                    a = a + np.abs(truth_dict[app][instance][t] - predicted_dict[app][instance][t])
                    b = b + predicted_dict[app][instance][t]
                except:
                    # 当前时间电器没数据，则认为判断对了
                    pass
                # b = b + predicted_dict[app][instance][t]
        return 1 - a / (2 * b)

    def percentages_apps(self):
        app_instances = self.house.instance_names
        truth_dict = self.house.appliance_pss_equal_length_dict
        predicted_dict = self.appliance_consumption_predicted
        percentages_pre = {}
        percentages_truth = {}
        for app, instance in app_instances:
            percentages_pre.update({app + '__' + instance: predicted_dict[app][instance].sum()})
            percentages_truth.update({app + '__' + instance: truth_dict[app][instance].sum()})

        def getTotalDict(thedict):
            total = 0
            for key, cal in thedict.items():
                total = total + cal
            for key, cal in thedict.items():
                thedict[key] = cal / total
            return thedict

        percentages_pre = getTotalDict(percentages_pre)
        percentages_truth = getTotalDict(percentages_truth)
        print()

    def tp_fn_fp_tn_4ps(self, predicted_ps, truth_ps, _on=10):
        y_true, y_pred= self.get_pres_truth_list(predicted_ps, truth_ps, _on)
        assert len(y_true)==len(y_pred)
        tp,fn,fp,tn=0,0,0,0
        for _truth,_pred in zip(y_true,y_pred):
            if(_truth==1 and _pred==1):
                tp+=1
            elif(_truth==1 and _pred==0):
                fn+=1
            elif (_truth == 0 and _pred == 1):
                fp += 1
            elif (_truth == 0 and _pred == 0):
                tn += 1
        return (tp,fn,fp,tn)


    def get_pres_truth_list(self, predicted_ps, truth_ps, _on):
        if (len(predicted_ps) != len(truth_ps)):
            print('两个序列不一样长')
        start = min(min(predicted_ps.index), min(truth_ps.index))
        end = max(max(predicted_ps.index), max(truth_ps.index))
        y_true, y_pred = [], []
        for time_stamp in pd.date_range(start, end):
            try:
                _pre = predicted_ps[time_stamp]
                _truth = truth_ps[time_stamp]
            except:
                continue
            if (_pre > _on):
                y_pred.append(1)
            else:
                y_pred.append(0)
            if (_truth > _on):
                y_true.append(1)
            else:
                y_true.append(0)
        return y_true, y_pred