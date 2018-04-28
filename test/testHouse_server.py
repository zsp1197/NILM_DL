# Created by zhai at 2018/4/28
# Email: zsp1197@163.com
from datamarket.NILMDataset import NILMDataset

nilm_dataset=NILMDataset('../data/allappliances4.hdf5','../data/allappliancesdict_refine4')
nilm_dataset.getStates()
# self.nilm_dataset.houses[0].get_states_list_sm_by_TIE(self.nilm_dataset.states_dict)
for house in nilm_dataset.houses:
    house.get_states_list_sm_by_TIE(nilm_dataset.states_dict)
house=nilm_dataset.houses[2]
house.get_instance_sections()
pass