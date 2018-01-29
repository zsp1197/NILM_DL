import traceback

from sklearn import mixture
from sklearn.covariance import EllipticEnvelope

import Tools,os
from copy import deepcopy

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
# from remote_visdom.VisdomGo import *
from collections import Iterable
from scipy.spatial import distance_matrix
# import baddict
from datamarket.Data_store import Data_store


class Clustering(object):
    _on_threshold = 8
    center_file = 'allappliances_refinedes.csv'

    def __init__(self, hdf5_path='我这么做是为什么，我快乐么，这是我的初心么，好累啊'):
        if(not hdf5_path=='我这么做是为什么，我快乐么，这是我的初心么，好累啊'):
            self.data_store = Data_store(hdf5_path)

        Y = np.array([150, 100, 80, 50, 40, 30, 20, 10, 6, 5, 2])
        X = np.array([1500, 1000, 700, 500, 400, 200, 100, 50, 10, 4, 0])
        self.p = np.poly1d(np.polyfit(X, Y, 3))

    def deal_with_ps_b(self, ps=None,not_deal_off=True):
        a = self.find_states_kmeans_step1(ps)
        naiveresult = self.cluster2df(a, ps)
        b = self.find_states_kmeans_step2(naiveresult, leastmembers=40)
        return [i['value'][0] for i in b]

    def deal_with_ps(self, ps=None, key=None, not_deal_off=True):
        def do_clusting(ps):
            if (len(ps) == 0):
                return []
            a = self.find_states_kmeans_step1(ps)
            naiveresult = self.cluster2df(a, ps)
            b = self.find_states_kmeans_step2(naiveresult, leastmembers=40)
            if (len(b) == 0):
                pass
            c = self.find_states_kmeans_step3(resultlist=b, ps=ps)
            cluster_centers = [i['value'][0] for i in c]
            return self.find_states_kmeans_step4(cluster_centers)

        def get_off_centers(ps_off):
            if (len(ps_off) == 0):
                return []
            a = self.find_states_kmeans_step1(ps=Tools.ps_refine_on(ps=ps, on=False, threshold=self._on_threshold),
                                              maxclusters=2)
            naiveresult = self.cluster2df(a, ps)
            b = self.find_states_kmeans_step2(naiveresult, leastmembers=40)
            return [i["value"][0] for i in b]

        if ((~isinstance(ps, pd.Series)) & (key == None)):
            raise ValueError
        if (key != None):
            ps = self.data_store.get_ps(key=key)
        if (not not_deal_off):
            off_centers = get_off_centers(ps_off=Tools.ps_refine_on(ps=ps, on=False, threshold=self._on_threshold))
            on_centers = do_clusting(Tools.ps_refine_on(ps=ps, on=True, threshold=self._on_threshold))
            for i in on_centers:
                off_centers.append(i)
            return self.find_states_kmeans_step4(off_centers)
        else:
            return do_clusting(ps)

    '''
    a typical example to get the cluster_centers according to the power_series

    a=find_states_kmeans_step1(ps4test)
    naiveresult=cluster2df(a,ps4test)
    b=find_states_kmeans_step2(naiveresult)
    c=find_states_kmeans_step3(resultlist=b,ps=ps4test)
    cluster_centers=[i['value'] for i in c]
    print cluster_centers
    '''

    def naivearray2smart(self, thearray):
        '''
        kmeans can only recognize stuff like np.array([[member1],[member2],[member3]])
        this method can modify [member1,member2,member3] to latter stuff
        '''
        for i, var in enumerate(thearray):
            if (i == 0):
                result = [[var]]
            else:
                result.append([var])
        return np.array(result)

    def find_states_kmeans_step1(self, ps, maxclusters=10, hasoffstate=True):
        '''
        ps must be a np.array object, and each member is a list contains the high dimention data
        step1 aims to get a bigger cluster_centers, which must be >=real_clusters
        ps: pandas.series object
        '''
        ps = self.naivearray2smart(ps.values)
        threshold_1 = 0.2
        threshold_2 = 0.5
        # threshold_3 = 0.2
        random_state = 10

        if (maxclusters == 2):
            # if not, the range(minstates, maxclusters) could raise exception
            maxclusters = 3
        if (hasoffstate):
            minstates = 2
        else:
            # has not been tested
            minstates = 1
        for n_clusters in range(minstates, maxclusters + 1):
            if (len(ps) <= n_clusters):
                raise ValueError("too short ps")
            # init=np.array([for i in ])
            init = [[i] for i in range(0, n_clusters)] * ((max(ps) - min(ps)) / (n_clusters - 1))
            if (ps.shape[0] > 40000):
                '''if num of samples is biger than 40000, run the minibatchkmeans'''
                # I guess.......
                batch_size = ps.shape[0] // 1000
                clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                                            batch_size=batch_size)
            else:
                clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
            clusterer.fit(ps)
            if n_clusters == minstates:
                clusterer_list = [clusterer]
            else:
                clusterer_list.append((clusterer))
                # not deepcopy(), take care!
                if (clusterer_list[-1] is clusterer_list[-2]):
                    raise LookupError
            if (n_clusters == minstates):
                loss = [clusterer.inertia_]
            else:
                loss.append(clusterer.inertia_)
                try:
                    if (loss[-1] - loss[-2] > 0):
                        warnings.warn(
                            'zhai: loss has been increased, the following steps maybe confused, considering change the initial random value',
                            UserWarning)
                    loss1 = abs(loss[-2] - loss[-1]) / loss[-1]
                    loss2 = abs(loss[-3] - loss[-2]) / loss[-2]
                    # if (step2):
                    #     threshold_1 = threshold_3
                    if ((loss1 < threshold_1) and (loss2 < threshold_1)):
                        if ((loss1 - loss2) > threshold_2):
                            warnings.warn('zhai: Hmm... maybe a bigger maxclusters could get a better result',
                                          UserWarning)
                        # if (step2):
                        #     if (n_clusters == ps.shape[0]):
                        #         losses=[i.inertia_ for i in clusterer_list]
                        #         return clusterer
                        #     return clusterer_list[-2]
                        return clusterer_list[-3]
                except Exception:
                    pass

        warnings.warn('zhai: maxclusters has been reached, consider a bigger maxclusters?',
                      UserWarning)
        return clusterer_list[-1]

    def find_states_kmeans_step2(self, resultlist, leastmembers=1):
        '''
        this step could delete the clusters which have nor much members(>10) or a little  average_inertia, which is calculated by polyfitting
        :param resultlist: required by running cluster2df()
        :return: similar to the input
        '''
        result = []
        for i, var in enumerate(resultlist):
            num = len(var["members"])
            center = var["value"]
            average_inertia = var["average_inertia"]
            if (num > leastmembers and average_inertia < self.p(center)):
                result.append(var)
        return result

    def refine_centers(self, resultlist):
        '''
        in this step, each cluster should be taken seriously, the algorithm could find centers which are close to each other,
        and combine to a single center
        the recommented input is the output of find_states_kmeans_step2()
        :param resultlist: a list contains dicts
                rach member in the dict is a cluster, described by a dict{"index","value","members","average_inertia"}
        :return:the kmeans object contains the refined cluster centers
        '''
        threshold = 0.065
        tocluster = [i["value"] for i in resultlist]
        maxclusters = len(resultlist)
        minstates = 2
        if (maxclusters == 1):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            minstates = 1
        for n_clusters in range(minstates, maxclusters + 1):
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            if (maxclusters == 1):
                clusterer = KMeans(n_clusters=1, random_state=10)
                clusterer.fit(tocluster)
                return clusterer
            clusterer.fit(tocluster)
            loss = clusterer.transform(tocluster)
            for i in loss:
                i.sort()
                if (i[0] > threshold * i[1]):
                    # the n_clusters should be updated
                    break
                return clusterer
        warnings.warn('zhai: all cluster centers are close to each other')
        return clusterer

    def find_states_kmeans_step3(self, resultlist, ps):
        clusterer = self.refine_centers(resultlist)
        return self.cluster2df(clusterer=clusterer, ps=ps)

    def find_states_kmeans_step4(self, centers_list):
        for i, var in enumerate(centers_list):
            try:
                centers_list[i] = var[0]
            except:
                pass

        centers_list.sort()
        if (len(centers_list) == 0):
            raise ValueError('no centers!')

        def if_close_matrix(thelist):
            if (len(thelist) in (1, 2)):
                warnings.warn("list length equals to 1")
                return None
            dm = distance_matrix(self.naivearray2smart(thelist), self.naivearray2smart(thelist))
            unzip_lst = zip(*dm)
            for i, var1 in enumerate(unzip_lst):
                for j, var2 in enumerate(var1):
                    if (i == j):
                        dm[i, j] = 0
                    else:
                        distance_threshold = max(self.p(thelist[i]), self.p(thelist[j]))
                        if (dm[i][j] < distance_threshold):
                            dm[i][j] = 1
                        else:
                            dm[i][j] = 0
            return dm

        dm = if_close_matrix(centers_list)
        if (dm is None):
            return centers_list
        while (not np.array_equal(dm, np.zeros((len(dm), len(dm))))):
            print(centers_list)
            for i, var1 in enumerate(dm):
                if_break = False
                # todeal = var1[i:]
                for j, var2 in enumerate(var1):
                    if (var2 == 1):
                        centers_list.append((centers_list[i] + centers_list[j]) / 2)
                        centers_list = Tools.remove_items_from_list(centers_list, (i, j))
                        centers_list.sort()
                        if_break = True
                        break
                if (if_break):
                    dm = if_close_matrix(centers_list)
                    if (dm is None):
                        return centers_list
                    break
        return centers_list

    def get_averageinertia(self, label, thelist):
        result = 0
        for i in thelist:
            result = result + abs(i.values[0] - label[0])
        if (len(thelist) == 0):
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            return 0
        return result / len(thelist)

    def cluster2df(self, clusterer, ps):
        '''
        the function could convert a power series to a detailed description according to clusterer
        :param clusterer: kmeans object
        :param ps:pandas.Series
        :return: the result is a list contains dicts
                each member in the dict is a cluster, described by a dict{"index","value","members","average_inertia"}
        '''
        pslabel = clusterer.predict(self.naivearray2smart(ps.values))
        label = [i for i in clusterer.cluster_centers_]
        thelist = []
        resultlist = []
        for i in label:
            thelist.append([])
        for i, var in enumerate(pslabel):
            thelist[var].append(pd.Series(data=ps[i], index=pd.DatetimeIndex([ps.index[i]])))
        for i, var in enumerate(thelist):
            resultlist.append({"index": i, "value": label[i], "members": thelist[i],
                               "average_inertia": self.get_averageinertia(label[i], thelist[i])})
        return resultlist

    def list2series(self, thelist):
        '''
        as a way to remedy cluster2df["members"], the function convert a list contains series to a single series
        :param thelist: [pd.series,pd.series,pd.series,...,pd.series], each member in the list share a common form of index
        :return: combied pd.series
        '''
        result = pd.Series()
        for i in self.thelist:
            result = result.append(i, verify_integrity=True)
        return result

    def get_label_idx_list(self, ps, centers):
        '''
        
        :param ps: 
        :param centers: list of cluster centers
        :return: list having same length as ps, and each member is assigned a center index of centers
        '''
        assert isinstance(ps, pd.Series) & isinstance(centers, Iterable)
        dm = distance_matrix(self.naivearray2smart(ps.values), self.naivearray2smart(centers))
        label_idx_list = []
        for mem in dm:
            idx = np.argmin(mem)
            label_idx_list.append(idx)
        return label_idx_list

    def ps2description(self, ps, centers):
        '''
        convert pd.Series to a detailed description accrding to centers
        conters are list of cluster centers, and 
        :param ps: 
        :param centers: 
        :return: list of tuples (starttime(pd.timestamp),endtime(pd.timestamp), cluster center value)
        '''
        label_idx_list = self.get_label_idx_list(ps, centers)
        result_list = []
        ps_index = ps.index
        for i, var in enumerate(label_idx_list):
            if (i == 0):
                temp = var
                last_i = 0
            # elif (i == len(ps)):
            #     pass
            else:
                if ((temp != var) or (i == len(ps)-1)):
                    theTuple = (ps_index[last_i], ps_index[i], centers[temp])
                    result_list.append(theTuple)
                    temp = var
                    last_i = i
        return result_list

    def description_refine(self, description_list):
        '''
        
        :param description_list: get from ps2description
        :return: dict{center value:[Tuples(startTime,endTime,timedelta)]}
        '''
        result = {}
        for mem in description_list:
            value = mem[2]
            start = mem[0]
            end = mem[1]
            try:
                state_list = result[value]
            except:
                result.update({value: []})
                state_list = result[value]
            #TODO if noise
            if((end-start).seconds>10):
                state_list.append((start, end, end - start))
        return result

    def pandasTime_2_number(self, refined_description_dict):
        '''
        
        :param refined_description_dict: from description_refine
        :return: dict{center value:[Tuples(startTime minutes of the day,endTime minutes of the day.exist time in minutes)]}
        '''
        result = {}
        for key in refined_description_dict:
            refined_description_list = refined_description_dict[key]
            tempList = []
            for mem in refined_description_list:
                tempList.append((Tools.timestamp_2_location_of_day(mem[0]), Tools.timestamp_2_location_of_day(mem[1]),
                                 Tools.timedelta_2_naive(mem[2])))
            result.update({key: tempList})
        return result

    def outlier_detection(self, to_fit, outliers_fraction=0.2):
        '''
        assuming the to_fit fits a gaussian_pdf_truncated distribution, giving the proportion of outliers in the data set, get the inliers
        :param to_fit:list of data
        :param outliers_fraction:
        :return: list:inliers
        '''
        classifier = EllipticEnvelope(contamination=outliers_fraction)
        try:
            classifier.fit(np.array(to_fit))
        except:
            warnings.warn("zhai: out_lier detection falied! Maybe there are no out_liers?")
            return to_fit
        return [to_fit[i] for (i, var) in enumerate(classifier.predict(np.array(to_fit))) if var == 1]

    def find_gaussian(self, to_fit, outliers_fraction=None):
        '''
        given a data set, find Gaussian distributed parameters
        :param to_fit:list of data, notice that each member in the list is also a list, describing different dimentions,
         however, a check is implemented. 
        :return: tuple(mean of gaussian_pdf_truncated, covariance of gaussian_pdf_truncated)
        '''
        from scipy.stats import norm
        if (~(isinstance(to_fit[0], list))):
            to_fit = [[i] for i in to_fit]
        if (outliers_fraction != None):
            to_fit_inliers = self.outlier_detection(to_fit, outliers_fraction=outliers_fraction)
        else:
            to_fit_inliers = to_fit
        mu, std = norm.fit(to_fit_inliers)
        return (mu, std)

    def time_numbers_2_gaussian(self, members_time_dict, outliers_fraction=0.1):
        '''
        
        :param members_time_dict: get from pandasTime_2_number
        :param outliers_fraction: 
        :return: dict{cluster center value:dict{"start_time":tuple(mean,std),"end_time":tuple(mean,std),"delta_time":tuple(mean,std)}}
        '''
        result = {}
        for key in members_time_dict:
            thelist = members_time_dict[key]
            start_tofit = []
            end_tofit = []
            delta_tofit = []
            key_dict = {}
            for mem in thelist:
                start_tofit.append(mem[0])
                end_tofit.append(mem[1])
                delta_tofit.append(mem[2])
            key_dict.update({"start_time": self.find_gaussian(to_fit=start_tofit, outliers_fraction=outliers_fraction)})
            key_dict.update({"end_time": self.find_gaussian(to_fit=end_tofit, outliers_fraction=outliers_fraction)})
            key_dict.update({"delta_time": self.find_gaussian(to_fit=delta_tofit, outliers_fraction=outliers_fraction)})
            result.update({key: key_dict})
        return result

    def ps_and_center_2_timedict(self, ps, centers):
        '''
        
        :param ps: power series, pandas.Series
        :param centers: list of cluster centers
        :return: dict{cluster center value:dict{"start_time":tuple(mean,std),"end_time":tuple(mean,std),"delta_time":tuple(mean,std)}}
        '''
        a = self.ps2description(ps=ps, centers=centers)
        b = self.description_refine(a)
        c = self.pandasTime_2_number(b)
        d = self.time_numbers_2_gaussian(c)
        return d

    def get_values_list_of_ps_idx(self, index_list, ps):
        values = ps.values
        return [values[idx] for idx in index_list]

    def ps_and_center_2_powerdict(self, ps, centers):
        '''
        
        :param ps: power series, pandas.Series
        :param centers: list of cluster centers
        :return: dict{cluster center value:dict{"power_value":tuple(mean,std)}}
        '''
        result = {}
        label_idx_list = self.get_label_idx_list(ps=ps, centers=centers)
        for i, center_value in enumerate(centers):
            values_list = self.get_values_list_of_ps_idx(index_list=Tools.idx_of_mem_list(label_idx_list, i), ps=ps)
            mean, std = self.find_gaussian(to_fit=values_list, outliers_fraction=0.002)
            result.update({center_value: {"power_value": (mean, std)}})
        return result

    def combine_dicts(self, dicts_list):
        '''
        combine dicts
        :param dicts_list:[dict1,dict2,...] 
        :return: combined dict
        '''
        cluster_centers = dicts_list[0].keys()
        result = {}
        for dict in dicts_list:
            assert cluster_centers == dict.keys()
        for cluster_center in cluster_centers:
            thedict = {}
            for dict in dicts_list:
                try:
                    thedict[cluster_center].update(dict[cluster_center])
                except:
                    thedict[cluster_center] = deepcopy(dict[cluster_center])
            result.update(thedict)
        return result

    def clustering_then_csv(self, days=4):
        result = []
        keys_dict = self.data_store.keys_dict
        for appliance_dict_key in keys_dict:
            for instance_dict_key in keys_dict[appliance_dict_key]:
                try:
                    if (instance_dict_key in baddict[appliance_dict_key]): continue
                except:
                    pass
                try:
                    instance_keys_list = keys_dict[appliance_dict_key][instance_dict_key]
                    if (len(instance_dict_key) >= days):
                        ps = Tools.ps_concatenate([self.data_store.get_ps(key) for key in instance_keys_list[0:days]])
                    else:
                        ps = Tools.ps_concatenate([self.data_store.get_ps(key) for key in instance_keys_list])
                    cluster_centers = self.deal_with_ps(ps, not_deal_off=False)
                    print("Done with " + appliance_dict_key + ' ' + instance_dict_key)
                    print(cluster_centers)
                    # ps.plot()
                    # plt.show()
                    a = [appliance_dict_key, instance_dict_key]
                    [a.append(str(cluster_centers[i])) for i, var in enumerate(cluster_centers)]
                    result.append(deepcopy(a))
                    # visdom_line(y=[ps.values,[i for i in range(0,len(ps))]],append=True)
                except Exception as err:
                    print("ERROR in " + appliance_dict_key + ' ' + instance_dict_key)
                    traceback.print_tb(err.__traceback__)

        Tools.list2csv(result, self.center_file)

    def read_center_file(self, filePath=center_file):
        '''
        
        :param filePath: 
        :return:{appliance name:{instance name:[center value]}} 
        '''
        df = pd.read_csv(filePath, index_col=0, header=-1)
        appliance_names_list = self.data_store.appliance_names
        result = {}
        for appliance_name in appliance_names_list:
            df_appliance = df.loc[appliance_name]
            temp_dict = {}
            for i in range(0, len(df_appliance)):
                try:
                    single_instance_series = df_appliance.iloc[i].dropna()
                except:
                    pass
                single_instance_list = single_instance_series.tolist()
                instance_name = single_instance_list.pop(0)
                temp_dict.update({instance_name: single_instance_list})
            result.update({appliance_name: temp_dict})
        return result

    def appliance_instance_2_powerTime_dict(self, appliance_name, instance):
        center_dict = self.centerDict
        try:
            centers = center_dict[appliance_name][instance]
        except:
            warnings.warn('no such instance ' + appliance_name + ' ' + instance)
            return
        ps = self.data_store.get_instance_ps(appliance_name=appliance_name, instance=instance)

        return self.behavior_dicts(centers, ps)

    def behavior_dicts(self, centers, ps):
        timedict = self.ps_and_center_2_timedict(ps=ps, centers=centers)
        powerdict = self.ps_and_center_2_powerdict(ps=ps, centers=centers)
        powerConsumeDict = self.ps_and_center_2_powerConsumeDict(ps=ps, centers=centers)
        if (len(timedict) == 0):
            print("get it!")
            pass
        print(len(timedict))
        return self.combine_dicts([timedict, powerdict, powerConsumeDict])

    def deal_all_instance(self):
        appliance_names_list = self.data_store.appliance_names
        keys_dict = self.data_store.keys_dict
        result = {}
        for appliance_name in appliance_names_list:
            instance_name_list = [i for i in keys_dict[appliance_name].keys()]
            temp_dict = {}
            for instance in instance_name_list:
                if(instance=='5-18'):
                    print()
                try:
                    thedict = self.appliance_instance_2_powerTime_dict(appliance_name=appliance_name, instance=instance)
                    temp_dict.update({instance: thedict})
                except Exception as err:
                    traceback.print_tb(err.__traceback__)
            result.update({appliance_name: temp_dict})
        return result

    def ps_and_center_2_powerConsumeDict(self, ps, centers):
        a = self.ps2description(ps=ps, centers=centers)
        b = self.description_refine(a)
        result = {}
        for i, center_value in enumerate(centers):
            description_list_tuples = b[center_value]
            values_list = [Tools.power_consumption_between_time(ps=ps, startTime=mem[0], endTime=mem[1]) for mem in
                           description_list_tuples]
            mean, std = self.find_gaussian(to_fit=values_list, outliers_fraction=0.002)
            result.update({center_value: {"consumption_value": (mean, std)}})
        return result

    def getCenterDict(self,thepath):
        '''

        :get self.centerDict: {appliance name:{instance name:[center value]}}
        '''
        df = pd.read_csv(os.path.join(thepath, 'states.txt'), header=None, index_col=0)
        appliance_instances = list(df.index)
        states = []
        appliance_names=[i.split('_')[0] for i in appliance_instances]
        result=dict(zip(appliance_names,[{} for i in range(len(appliance_names))]))

        for appliance_name in appliance_instances:
            appliance_type, instance = appliance_name.split('_')
            states_series = df.loc[appliance_name][1:]
            states_np = states_series.values
            states_np = states_np.astype(np.float64)
            states_np = states_np[~np.isnan(states_np)]
            temp=list(states_np)
            temp.append(0)
            result[appliance_type][instance]=temp
        self.centerDict=result
        print()
