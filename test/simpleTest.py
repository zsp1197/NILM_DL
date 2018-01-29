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

thelongdescription = []
labels = []


def getLabels(starttime, endtime, house):
    labels = []
    for appliance, instance in house.instance_names:
        theps = house.data_store.get_instance_ps(appliance_name=appliance, instance=instance)[starttime: endtime]
        if(len(ps>0) and np.mean(theps)>5):
            labels.append(1)
        else:
            labels.append(0)
    return [tuple(labels)]



pool = multiprocessing.Pool()
for i,day in enumerate(daysTimeStamp):
    dayps = ps[day]
    if (len(dayps) == 0):
        daysTimeStamp.pop(daysTimeStamp.index(day))
        print(day + '是诗意的一天，这天没有用电呢')
        continue
    clustering = zhcluster.Clustering()
    centers = clustering.deal_with_ps_b(ps=dayps)
    # list of tuples (starttime(pd.timestamp),endtime(pd.timestamp), cluster center value)
    descriptions = clustering.ps2description(ps=dayps, centers=centers)
    thenewdescriptions = []
    for description in descriptions:
        starttime, endtime, center = description
        theps = ps[starttime:endtime]
        description = (
            Tools.timestamp_2_location_of_day(starttime), Tools.timedelta_2_naive(endtime - starttime),
            np.mean(theps.values.reshape(-1, 1)), np.std(theps.values.reshape(-1, 1)))
        thenewdescriptions.append(description)
        labels += getLabels(starttime, endtime, house)
    thelongdescription += thenewdescriptions
Tools.serialize_object(thelongdescription, 'instances')
Tools.serialize_object(labels, 'labels')
print()
