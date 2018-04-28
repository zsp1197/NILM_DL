# Created by zhai at 2018/4/27
# Email: zsp1197@163.com
from datamarket.Appliance_class import *


class StateManager(object):
    def __init__(self, instance_names,states_dict):
        self.instance_names = instance_names
        self.states_dict = states_dict
        self.appliance_encode_dict = self.get_appliance_encode(instance_names)
        self.get_appliance_state_r1_encode()


    def get_appliance_encode(self, instance_names):
        '''

        :param house:
        :return: {idx:Appliance_class}
        '''
        result = {}
        for i, appliance_instance_tuple in enumerate(instance_names):
            # start from 1
            result.update({i + 1: Appliance(appliance_type=appliance_instance_tuple[0],
                                                  instance=appliance_instance_tuple[1])})
        return result

    def get_appliance_state_r1_encode(self):
        for idx,appliance in self.appliance_encode_dict.items():
            the_states_dict=self.states_dict[appliance.appliance_type][appliance.instance]
            for state_value,thedict in the_states_dict.items():
                appliance.feed_appliance_state_r1_encoding_one_by_one(state_value=state_value,thedict=thedict)