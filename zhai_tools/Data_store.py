import pandas as pd
from copy import deepcopy

import zhai_tools.Tools as  Tools
class Data_store(object):
    '''
    keys: all keys belonging to the store
    appliance_names: name list of all appliances
    keys_dict: {appliance name:{instance name:[key]}}
    '''
    def __init__(self, redd_hdf5_path):
        self.store = pd.HDFStore(redd_hdf5_path)
        self.get_applance_keys()

    def get_applance_keys(self):
        self.keys = self.store.keys()
        appliance_names=[i.split('/')[2] for i in self.keys]
        self.appliance_names=Tools.list_move_duplicates(appliance_names)
        self.keys_dict=self.get_keys_dict()

    def get_keys_dict(self):
        result={}
        for key in self.keys:
            appliance_name=key.split('/')[2]
            instance_name=key.split('/')[3]
            try:
                # 看有没有这个电器对应的dict，若没有则创建
                instance_dict=result[appliance_name]
            except:
                instance_dict={appliance_name:{}}
                result.update(instance_dict)
            try:
                instance_list=instance_dict[instance_name]
            except:
                instance_dict.update({instance_name:[]})
                instance_list = instance_dict[instance_name]
            instance_list.append(key)
        return result

    def get_ps(self,key):
        theS= self.store[key]
        return pd.Series(index=theS.index, data=theS.values)

    def get_instance_ps(self,appliance_name,instance):
        keys=self.keys_dict[appliance_name][instance]
        pss=[self.get_ps(key) for key in keys]
        return Tools.ps_concatenate(pss)

    def get_the_appliance_the_day(self,appliance_name,day,iflonged=True):
        if(appliance_name not in self.appliance_names):
            raise LookupError
        for i in self.keys:
            if (day in i and appliance_name in i):
                if(iflonged):
                    if("longed" in i):
                        theS = self.store[i]
                        break
                    else:
                        continue
                else:
                    theS = self.store[i]
                    break
        try:
            theS
        except:
            raise ValueError('can not find the day or the appliance' )
        ps = pd.Series(index=theS.index, data=theS.values)
        return ps

    # def get_keys_dict_for_houses(self):
    #     result={}
    #     keys_dict=deepcopy(self.keys_dict)
    #     houses=[]
    #     for key in self.keys:
    #         houses.append(key.split('/')[3].split('-')[0])
    #     houses=tuple(Tools.list_move_duplicates(houses))
    #     for house in houses:
    #         house_dict={}
    #         for appliance_name, instances_dict in keys_dict.items():
    #             new_list=[]
    #             for instance_name,keys_list in instances_dict.items():
    #                 if(house in instance_name):
    #                     new_list.append(keys_list)

