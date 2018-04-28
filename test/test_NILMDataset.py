# Created by zhai at 2018/4/25
# Email: zsp1197@163.com
from unittest import TestCase
from datamarket.NILMDataset import NILMDataset

class TestNILMDataset(TestCase):
    def setUp(self):
        self.nilm_dataset=NILMDataset('../data/allappliances4.hdf5','../data/allappliancesdict_refine4')

    def test_house(self):
        self.nilm_dataset.getStates()
        # self.nilm_dataset.houses[0].get_states_list_sm_by_TIE(self.nilm_dataset.states_dict)
        for house in self.nilm_dataset.houses:
            house.get_states_list_sm_by_TIE(self.nilm_dataset.states_dict)
        pass

    # def test_refine_ps(self):

# tn=TestNILMDataset()
# pass