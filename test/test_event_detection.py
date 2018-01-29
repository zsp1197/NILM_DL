# Created by zhai at 2017/11/16
from unittest import TestCase


import Tools as zht
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Parameters import Parameters
from datamarket.Data_store import Data_store
from datamarket.Event_detection import Event_detection


class TestEvent_detection(TestCase):
    path = 'allappliances.hdf5'
    # path = 'D:\SJTU\pythoncode\summerTime\data\\allappliances.hdf5'
    data_store = Data_store(path)
    ps=data_store.get_instance_ps(appliance_name='fridge',instance='1-5')['2011-4-20']
    ps=zht.up_sample_ps(ps=ps)
    def test_delta_based(self):
        ed=Event_detection(ps=self.ps,parameters=Parameters())
        events=ed.delta_based()
        for event in events:
            ax = plt.gca()
            ax.annotate('event',
                        xy=(event[0], self.ps[event[0]]), xycoords='data',
                        xytext=(event[0] + pd.Timedelta('3s'), self.ps[event[0]] + 10),
                        arrowprops=dict(facecolor='black', shrink=0.03)
                        )
        self.ps.plot()
        plt.show()