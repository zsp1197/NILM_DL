import datetime
import shutil
import numpy as np
import pandas as pd
import pickle
import collections
import functools
import inspect

from copy import deepcopy


def list2csv(thelist, file_path):
    my_df = pd.DataFrame(thelist)
    my_df.to_csv(file_path, index=False, header=False)


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
    elif (acc == 'second'):
        return timestamp.hour * 3600 + timestamp.minute*60+timestamp.second
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


def ps_concatenate(pss):
    return pd.concat(pss)


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

def copyfile(src:str,tgt:str):
    shutil.copyfile(src, tgt)

def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        path + ' 创建成功'
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        path + ' 目录已存在'
        return False

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

@check_func_input_output_type_static
def n_smallest_of_list(n:int,list:list):
    import heapq
    return heapq.nsmallest(n, list)[-1]

def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

@check_func_input_output_type_static
def most_member_of_list(theist:list):
    candidate_list = list(set(theist))
    num_of_member = [theist.count(i) for i in candidate_list]
    biggest=max(num_of_member)
    if(num_of_member.count(biggest)>1):
        result=[]
        for i in range(0,num_of_member.count(biggest)):
            idx=num_of_member.index(biggest)
            result.append(deepcopy(candidate_list[idx]))
            candidate_list.pop(idx)
            num_of_member.pop(idx)
        return tuple(result)
    return candidate_list[num_of_member.index(max(num_of_member))]


@check_func_input_output_type_static
def most_member_of_list_with_weights(thelist: list, weights: list):
    candidate_list = tuple(set(thelist))
    votes=[0 for i in range(0,len(candidate_list))]
    for i,appliance_name in enumerate(candidate_list):
        for j,mem in enumerate(thelist):
           if(mem==appliance_name):
               votes[i]=votes[i]+weights[j]
    return candidate_list[votes.index(max(votes))]

@check_func_input_output_type_static
def power_consumption_between_time(ps:pd.Series,startTime:pd.Timestamp,endTime:pd.Timestamp):
    theps=ps[startTime:endTime]
    return ps_consumption(theps)


def ps_consumption(theps):
    result = 0
    for i, item in enumerate(theps):
        if (i == len(theps) - 1):
            break
        result = result + item * (theps.index[i + 1] - theps.index[i]).seconds
    return result




def up_sample_ps(ps: pd.Series, freq: str = 'S'):
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
    index=pdf.index
    data=[i[0] for i in pdf.values]
    return pd.Series(index=index,data=data)

@check_func_input_output_type_static
def list_select_with_indexes(thelist:list, indexes:list):
    if(len(indexes)==1):
        return [thelist[indexes[0]]]
    from operator import itemgetter
    return list(itemgetter(*indexes)(thelist))


def get_batch(inputs, targets, num_per_batch):
    '''
    最后一个batch可能数目不足num_per_batch
    :param inputs:
    :param targets:
    :param num_per_batch:
    :return:
    '''
    assert len(inputs) == len(targets)
    input_iter = inputs.__iter__()
    target_iter = targets.__iter__()
    input_batch = []
    target_batch = []
    num = 0
    for _input in input_iter:
        input_batch.append(_input)
        target_batch.append(target_iter.__next__())
        num += 1
        if (num == num_per_batch):
            yield input_batch, target_batch
            input_batch = []
            target_batch = []
            num = 0
    if (len(input_batch) > 0):
        yield input_batch, target_batch


@check_func_input_output_type_static
def split_list_to_chunks(thelist:list,n:int):
    """n-sized chunks from thelist."""
    return [thelist[i:i + n] for i in range(0, len(thelist), n)]

@check_func_input_output_type_static
def server_ps_plot(ps:pd.Series,label=' '):
    '''
    excecute "python -m visdom.server" in the console of server beforehand
    login http://202.120.60.50:8097/ and see what you got
    :param ps:
    :return:
    '''
    from visdom import Visdom
    viz = Visdom()
    viz.line(X=np.array(range(len(ps))),Y=ps.values,opts=dict(
        xlabel='Time',
        ylabel='Power',
        title=label
    ))


def server_pss_plot(pss:list):
    '''
    excecute "python -m visdom.server" in the console of server beforehand
    login http://202.120.60.50:8097/ and see what you got
    :param pss:list of ps
    :return:
    '''
    from visdom import Visdom
    viz = Visdom()
    lengths=[len(ps) for ps in pss]
    viz.line(X=np.array(range(max(lengths))),Y=np.column_stack([ps.values for ps in pss]),opts=dict(
        xlabel='Time',
        ylabel='Power'
    ))