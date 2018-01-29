import datetime

import numpy as np
import pandas as pd
import pickle
import collections
import functools
import inspect
from scipy.spatial import distance_matrix
from copy import deepcopy
import matplotlib.pyplot as plt


def list2csv(thelist, file_path, seq=','):
    my_df = pd.DataFrame(thelist)
    my_df.to_csv(file_path, index=False, header=False, sep=seq)


def list_move_duplicates(list):
    s = []
    for i in list:
        if i not in s:
            s.append(i)
    return s


def timestamp_2_location_of_day(timestamp, acc='min'):
    if (acc == 'min'):
        return timestamp.hour * 60 + timestamp.minute
    elif (acc == 'hour'):
        return timestamp.hour
    else:
        raise ValueError


def timedelta_2_naive(timedelta, acc='min'):
    if (acc == 'min'):
        return timedelta.seconds / 60
    elif (acc == 'hour'):
        return timedelta.seconds / 3600
    elif (acc == 'second'):
        return timedelta.seconds
    else:
        raise ValueError


def idx_of_mem_list(thelist, mem):
    return [i for i, x in enumerate(thelist) if x == mem]




def ps_refine_on(ps, threshold=8, on=True):
    ps_value = ps.values
    ps_index = ps.index
    value = []
    index = []
    for i, var in enumerate(ps_value):
        if (on):
            if (var > threshold):
                index.append(ps_index[i])
                value.append(var)
        elif (~on):
            if (var < threshold):
                index.append(ps_index[i])
                value.append(var)
    if (len(value) == 0):
        # Warning.warn('no on/off state')
        return []
    elif (len(value) < 20):
        return ps
    return pd.Series(data=value, index=index)


def list_equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched


def serialize_object(object, filePath):
    f = open(filePath, 'wb')
    pickle.dump(object, f)


def deserialize_object(filePath):
    # Restore from a file
    f = open(filePath, 'rb')
    data = pickle.load(f)
    return data


def remove_items_from_list(thelist, remove_indices):
    return [i for j, i in enumerate(thelist) if j not in remove_indices]


def check_func_input_output_type_static(func):
    # adding restriction to the type of input and output variable by annotation
    msg = ('Expected type {expected!r} for argument {argument}, '
           'but got type {got!r} with value {value!r}')
    # 获取函数定义的参数
    sig = inspect.signature(func)
    parameters = sig.parameters  # 参数有序字典
    arg_keys = tuple(parameters.keys())  # 参数名称
    return_type = sig.return_annotation

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        CheckItem = collections.namedtuple('CheckItem', ('anno', 'arg_name', 'value'))
        check_list = []

        # collect args   *args 传入的参数以及对应的函数参数注解
        for i, value in enumerate(args):
            arg_name = arg_keys[i]
            anno = parameters[arg_name].annotation
            check_list.append(CheckItem(anno, arg_name, value))

        # collect kwargs  **kwargs 传入的参数以及对应的函数参数注解
        for arg_name, value in kwargs.items():
            anno = parameters[arg_name].annotation
            check_list.append(CheckItem(anno, arg_name, value))

        # check type
        for item in check_list:
            if (item.anno == item.anno == inspect._empty):
                continue
            if not isinstance(item.value, item.anno):
                error = msg.format(expected=item.anno, argument=item.arg_name,
                                   got=type(item.value), value=item.value)
                raise TypeError(error)
        if (return_type == inspect._empty):
            return func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
            if not isinstance(result, return_type):
                raise TypeError("wrong return type")
            return result

    return wrapper

def n_smallest_of_list(n: int, list: list):
    import heapq
    return heapq.nsmallest(n, list)[-1]

#TODO test
@check_func_input_output_type_static
def ps_concatenate(pss:list)->pd.Series:
    # return pd.Series(pd.concat(pss))
    for i,var in enumerate(pss):
        if(i==0):
            result=var
        else:
            result=result.add(var,fill_value=0)
    return result

def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)


@check_func_input_output_type_static
def most_member_of_list(theist: list):
    candidate_list = list(set(theist))
    num_of_member = [theist.count(i) for i in candidate_list]
    biggest = max(num_of_member)
    if (num_of_member.count(biggest) > 1):
        result = []
        for i in range(0, num_of_member.count(biggest)):
            idx = num_of_member.index(biggest)
            result.append(deepcopy(candidate_list[idx]))
            candidate_list.pop(idx)
            num_of_member.pop(idx)
        return tuple(result)
    return candidate_list[num_of_member.index(max(num_of_member))]


@check_func_input_output_type_static
def most_member_of_list_with_weights(thelist: list, weights: list):
    candidate_list = tuple(set(thelist))
    votes = [0 for i in range(0, len(candidate_list))]
    for i, appliance_name in enumerate(candidate_list):
        for j, mem in enumerate(thelist):
            if (mem == appliance_name):
                votes[i] = votes[i] + weights[j]
    return candidate_list[votes.index(max(votes))]


@check_func_input_output_type_static
def power_consumption_between_time(ps: pd.Series, startTime: pd.Timestamp, endTime: pd.Timestamp):
    theps = ps[startTime:endTime]
    return ps_consumption(theps)


def ps_consumption(theps):
    result = 0
    for i, item in enumerate(theps):
        if (i == len(theps) - 1):
            break
        result = result + item * (theps.index[i + 1] - theps.index[i]).seconds
    return result


@check_func_input_output_type_static
def ps_between_timestamp(ps: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    #TODO 此处总是出错，hunk的时间戳找的有错!
    # df = ps.to_frame()
    # a = df[start:end]
    # # return a[0]
    # try:
    #     return pd.Series(index=a.index, data=a[0])
    # except Exception as e:
    #     return pd.Series(index=a.index, data=a)
    return ps[start:end]


def naivearray2smart(thearray):
    '''
    kmeans can only recognize stuff like np.array([[member1],[member2],[member3]])
    this method can modify [member1,member2,member3] to latter stuff
    '''
    for i, var in enumerate(thearray):
        if (i == 0):
            result = [[var]]
        else:
            result.append([var])
    return np.array(result)


@check_func_input_output_type_static
def get_label_idx_list(ps: pd.Series, centers: list):
    '''

    :param ps:
    :param centers: list of cluster centers
    :return: list having same length as ps, and each member is assigned a center index of centers
    '''
    # dm = distance_matrix(naivearray2smart(ps.values), naivearray2smart(centers))
    try:
        dm = distance_matrix(naivearray2smart([float(i[0]) for i in ps.values.tolist()]), naivearray2smart(centers))
    except:
        dm = distance_matrix(naivearray2smart(ps.values), naivearray2smart(centers))
    label_idx_list = []
    for mem in dm:
        idx = np.argmin(mem)
        label_idx_list.append(idx)
    return label_idx_list


def ps2description(ps, centers):
    '''
    这个和summerTime里面那个有很大区别！主要是最后一个chunk被保留！
    convert pd.Series to a detailed description accrding to centers
    conters are list of cluster centers, and 
    :param ps: 
    :param centers: 
    :return: list of tuples (starttime(pd.timestamp),endtime(pd.timestamp), cluster center value)
    '''
    label_idx_list = get_label_idx_list(ps, centers)
    result_list = []
    ps_index = ps.index
    for i, var in enumerate(label_idx_list):
        if (i == 0):
            temp = var
            last_i = 0
        elif (i == len(ps) - 1):
            theTuple = (ps_index[last_i], ps_index[i - 1], centers[temp])
            result_list.append(theTuple)
        else:
            if (temp != var):
                theTuple = (ps_index[last_i], ps_index[i - 1], centers[temp])
                result_list.append(theTuple)
                temp = var
                last_i = i
    return result_list

@check_func_input_output_type_static
def up_sample_ps(ps: pd.Series, freq: str = 'S')->pd.Series:
    '''
    the data maybe compressed, pro-long the data with a fixed sample period
    :param ps:pd.Series(index=datatimeindex,data=power_read)
    :return: pd.Series
    '''
    index = pd.to_datetime(ps.index)
    longindex = pd.date_range(start=min(index), end=max(index), freq=freq)
    pdf = pd.DataFrame(index=longindex, columns=['0'])
    pdf.ix[index, 0] = ps.values.tolist()
    pdf = pdf.fillna(method='pad')['0']
    return pdf


def ps2description_event(ps, para):
    #TODO 顾名思义
    return None