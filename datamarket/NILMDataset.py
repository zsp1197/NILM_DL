# Created by zhai at 2018/4/25
# Email: zsp1197@163.com
from datamarket.House import *
from datamarket.Data_store import Data_store
from Tools import *


class NILMDataset(object):
    def __init__(self,dataset_path,dictpath):
        self.datastore=Data_store(dataset_path)
        self.states_dict=deserialize_object(dictpath)

    def getStates(self):
        # self.states_list=feed_states(self.states_dict)
        self.houses=feed_Houses(self.datastore)

    # def labeling_ps_with