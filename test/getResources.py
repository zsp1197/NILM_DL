# Created by zhai at 2018/1/28
# Email: zsp1197@163.com
import multiprocessing
from copy import deepcopy
from unittest import TestCase
import datamarket
from Parameters import Parameters
from datamarket.House import House
import pandas as pd
from datamarket.Data_store import Data_store
import matplotlib.pyplot as plt
import numpy as np
import Tools
import zhai_tools.Clustering as zhcluster
import time


def feed_Houses(data_store):
    houses_idx = []
    houses = []
    for key in data_store.keys:
        houses_idx.append(key.split('/')[3].split('-')[0])
    houses_idx = tuple(Tools.list_move_duplicates(houses_idx))
    for house_idx in houses_idx:
        houses.append(
            House(data_store=data_store, house_idx=house_idx, parameters=Parameters(), check_good_sections=False))
    return houses


path = 'allappliances.hdf5'
data_store = Data_store(path)
houses = feed_Houses(data_store)
print()
house = houses[1]
ps = pd.Series()
thepss = []
for appliance, instance in house.instance_names:
    dayps = house.data_store.get_instance_ps(appliance_name=appliance, instance=instance)
    thepss.append(dayps)
    ps = ps.add(dayps, fill_value=0)
    # Tools.server_ps_plot(ps=dayps, label=appliance)

startDayStr, endDayStr = min(ps.index)._date_repr, max(ps.index)._date_repr
daysTimeStamp = [i._date_repr for i in list(pd.date_range(start=startDayStr, end=endDayStr, freq='D'))]



def getLabels(starttime, endtime, house):
    labels = []
    for appliance, instance in house.instance_names:
        theps = house.data_store.get_instance_ps(appliance_name=appliance, instance=instance)[starttime: endtime]
        if(len(theps)>0 and np.mean(theps)>5):
            labels.append(1)
        else:
            labels.append(0)
    return tuple(labels)





def dealDay(i, dayps,day,house):
    print('进入进程，处理第{0}个，'.format(i)+day)
    thelongdescription=[]
    instance_labels=[]
    # dayps = ps[day]
    if (len(dayps) == 0):
        # daysTimeStamp.pop(daysTimeStamp.index(day))
        print(day + '是诗意的一天，这天没有用电呢')
        return None,None
    clustering = zhcluster.Clustering()
    centers = clustering.deal_with_ps_b(ps=dayps)
    # list of tuples (starttime(pd.timestamp),endtime(pd.timestamp), cluster center value)
    descriptions = clustering.ps2description(ps=dayps, centers=centers)
    for description in descriptions:
        starttime, endtime, center = description
        theps = ps[starttime:endtime]
        description = (
            Tools.timestamp_2_location_of_day(starttime), Tools.timedelta_2_naive(endtime - starttime),
            np.mean(theps.values.reshape(-1, 1)), np.std(theps.values.reshape(-1, 1)))
        thelongdescription.append(description)
        instance_labels.append(getLabels(starttime, endtime, house))
    return (thelongdescription,instance_labels,i)
pool = multiprocessing.Pool()
poolResults=[]
instances=[]
labels=[]

daysTimeStamp=daysTimeStamp[0:3]

for i,day in enumerate(daysTimeStamp):
    # thelongdescription, instance_labels=dealDay(i, day)
    poolResults.append(pool.apply_async(dealDay,(i,ps[day],day,deepcopy(house),)))
    # instances+=thelongdescription
    # labels+=instance_labels
pool.close()
pool.join()
poolReturns=[]
for res in poolResults:
    poolReturns.append(res.get())


def selectI(i, poolReturns):
    theOne=None
    for mem in poolReturns:
        if(mem[2]==i):
            theOne=mem
    return theOne[0],theOne[1]

for i in range(len(daysTimeStamp)):
    instance,label=selectI(i,poolReturns)
    if((instance == None) and (label == None)):
        print(daysTimeStamp[i])
        continue
    instances+=instance
    labels+=label
print(instances)
print(labels)
Tools.serialize_object(instances, 'instances_pool')
Tools.serialize_object(labels, 'labels_pool')
print()
