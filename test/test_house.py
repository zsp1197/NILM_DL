# Created by zhai at 2017/10/30
import datetime
from copy import deepcopy
from unittest import TestCase
import datamarket
from Parameters import Parameters
from datamarket import House
import pandas as pd
from datamarket.Data_store import Data_store
import matplotlib.pyplot as plt
import numpy as np
import zhai_tools.Tools as zht
import zhai_tools.Clustering as zhcluster
import time


class TestHouse(TestCase):
    path = 'allappliances.hdf5'
    # path = 'D:\SJTU\pythoncode\summerTime\data\\allappliances.hdf5'
    data_store = Data_store(path)
    houses = House.feed_Houses(data_store)

    def test_aggregate(self):
        for house in self.houses:
            ps = house.aggregate()
            ps.plot(label=house.house_idx)
            plt.legend()
            plt.show()
            # ps=ps.between_time(pd.Timestamp('2011-05-08'), pd.Timestamp('2011-05-15'))

    def test_get_good_sections(self):
        house = self.houses[0]
        ps = house.aggregate()
        # ps=ps.between_time('2011-04-24 20:00', '2011-04-25 22:00')
        # ps.plot()
        pss = house.get_good_pss(ps=ps)
        for theps in pss:
            theps.plot()

        # df=ps.to_frame()
        # print(df)
        plt.show()

    def test_sax(self):
        rng1 = pd.date_range('1/1/2011', periods=24, freq='H')
        rng2 = pd.date_range('1/2/2011', periods=24, freq='H')
        rng3 = pd.date_range('1/3/2011', periods=22, freq='H')
        ts1 = pd.Series(10 + np.random.randn(len(rng1)), index=rng1)
        ts1 = zht.up_sample_ps(ps=ts1)
        ts2 = pd.Series(1000 + np.random.randn(len(rng2)), index=rng2)
        ts3 = pd.Series(500 + np.random.randn(len(rng3)), index=rng3)

        ts = ts1.add(ts2, fill_value=0)
        ts = ts.add(ts3, fill_value=0)
        ts.plot()
        a = self.sax(ts)
        print(a)
        plt.show()

    def sax(self, ps: pd.Series):
        '''
        refine power series to sax description
        :param ps: pd.Series index:timestamp value: power readings
        :param max_T: max-time between samples
        :return: list[tuples(delata_value,total_value,start_time,delta_time)]
        '''
        # list of tuples (starttime(pd.timestamp),endtime(pd.timestamp), cluster center value)
        naive_description = zht.ps2description(ps=ps, centers=Parameters().sax_steps)
        result = []
        for i, thenaive in enumerate(naive_description):
            if (i == 0):
                result.append((thenaive[2], thenaive[2], thenaive[0], thenaive[1] - thenaive[0]))
            else:
                result.append(
                    (thenaive[2] - naive_description[i - 1][2], thenaive[2], thenaive[0], thenaive[1] - thenaive[0]))
        return result

    def test_up_sample(self):
        rng1 = pd.date_range('1/1/2011', periods=24, freq='H')
        # rng2 = pd.date_range('1/2/2011', periods=24, freq='H')
        # rng3 = pd.date_range('1/3/2011', periods=22, freq='H')
        ts1 = pd.Series(10 + np.random.randn(len(rng1)), index=rng1)
        start = time.clock()
        ts2 = zht.up_sample_ps(ps=deepcopy(ts1))
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)

        start = time.clock()
        ts3 = self.up_sample_ps(ps=ts1)
        elapsed = (time.clock() - start)
        print("Time used:", elapsed)
        assert ts2.values==ts3.values
        assert ts2.index==ts3.index
        assert ts2==ts3

    def up_sample_ps(self, ps: pd.Series, freq: str = 'S'):
        '''
        the data maybe compressed, pro-long the data with a fixed sample period
        :param ps:pd.Series(index=datatimeindex,data=power_read)
        :return: pd.Seires
        '''
        index = pd.to_datetime(ps.index)
        longindex = pd.date_range(start=min(index), end=max(index), freq=freq)
        pdf = pd.DataFrame(index=longindex, columns=['0'])
        pdf.ix[index, 0] = ps.values.tolist()
        pdf = pdf.fillna(method='pad')
        return pdf
