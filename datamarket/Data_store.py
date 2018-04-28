import pandas as pd
from copy import deepcopy

import Tools


class Data_store(object):
    '''
    keys: all keys belonging to the store
    appliance_names: name list of all appliances
    keys_dict: {appliance name:{instance name:[key]}}
    '''
    def __init__(self, redd_hdf5_path):
        self.redd_hdf5_path=redd_hdf5_path
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
                #TODO 第一天数据被舍弃
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
            raise ValueError('can not find the day or the appliance')
        ps = pd.Series(index=theS.index, data=theS.values)
        return ps

    # def unknown_app_power(self, pss, meter_power):
    #     '''
    #
    #     :param pss:
    #     :param meter_power:
    #     :return:
    #     '''
    #     for i, ps in enumerate(pss):
    #         if i == 0:
    #             total = pss[i]
    #         else:
    #             total += pss[i]
    #     result = meter_power - total
    #     idx = result < 0
    #     count = np.count_nonzero(idx)
    #     result[idx] = np.zeros(count)
    #     clustering = Clustering()
    #     centers_list = clustering.deal_with_ps(ps=result, not_deal_off=False)
    #     return result, centers_list