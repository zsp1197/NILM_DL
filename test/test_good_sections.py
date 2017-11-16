# Created by zhai at 2017/11/2
from copy import deepcopy
from unittest import TestCase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Parameters import Parameters
from datamarket.Data_store import Data_store
from datamarket.House import Good_sections, House, feed_Houses
import zhai_tools.Tools as zht

# Email: zsp1197@163.com
class TestGood_sections(TestCase):
    def init(self):
        rng = list(pd.date_range('1/1/2011', periods=36, freq='s'))
        rng_ori = deepcopy(rng)
        for i, var in enumerate(rng):
            if (i > 5):
                rng[i] = rng[i] + pd.Timedelta('20s')
            if (i > 10):
                rng[i] = rng[i] + pd.Timedelta('20s')
            if (i > 20):
                rng[i] = rng[i] + pd.Timedelta('20s')
        self.ps = pd.Series(6 + np.random.randn(len(rng)), index=rng)
        ps_ori = pd.Series(self.ps.values, rng_ori)
        # ps_ori.plot()
        self.ps.plot()
        # plt.show()

    def test_get_chunks_index(self):
        self.init()
        para = Parameters()
        para.max_T = pd.Timedelta('10s')
        gs = Good_sections(ps=self.ps, parameters=para)
        print(self.ps)
        print(gs.get_chunks_index())
        plt.show()

    def test_labeling(self):
        self.init4houses()
        self.house.aggregate()
        gs = Good_sections(ps=self.house.mainMeter, parameters=Parameters())
        count = 0
        for ps in gs.pss:
            chunks_list = self.house.sax(ps=ps)
            for i, chunk in enumerate(chunks_list):
                print(str(i))
                thelabel = self.house.labeling(chunk=chunk)
                print(thelabel)
                # ps.plot(color='k',label='total')
                ax = plt.gca()
                ax.annotate(str(thelabel[0]),
                            xy=(chunk.start_time, chunk.total_value), xycoords='data',
                            xytext=(chunk.start_time + pd.Timedelta('3s'), chunk.total_value+10),
                            arrowprops=dict(facecolor='black', shrink=0.03)
                            )
            count+=1
            ps.plot(label='total',color='b')
            if(count>3):
                break
        # self.house.mainMeter.plot(label='total')
        print(chunk)
        plt.xlim([chunks_list[0].start_time,chunk.start_time])
        plt.legend()
        print()
        plt.show()

    def init4houses(self):
        path = 'allappliances.hdf5'
        # path = 'D:\SJTU\pythoncode\summerTime\data\\allappliances.hdf5'
        data_store = Data_store(path)
        house = House(data_store=data_store, house_idx='1', parameters=Parameters())
        self.house=house
