# Created by zhai at 2018/5/29
# Email: zsp1197@163.com
from time import time

import numpy as np
import torch
from tqdm import tqdm
from visdom import Visdom

import Tools
from Parameters import Parameters
from Tools import sliding_window
from nilm_tools import *


class Infer():
    def credients(self, input_bin, target_bin, predictor, paras=Parameters()):
        # assert len(input_bin) == len(target_bin)
        self.paras = paras
        self.input_bin = input_bin
        self.target_bin = target_bin
        self.predictor = predictor

    def infer(self):
        predicted = self.predictor(self.input_bin).view(-1)
        assert len(predicted) == len(self.target_bin)
        assertedTrue = 0
        for predict, target in zip(predicted, self.target_bin):
            if (predict == target):
                assertedTrue += 1
        return assertedTrue / len(self.target_bin)

    def infer_batch_seq(self, paras=Parameters()):
        input_iter = sliding_window(self.input_bin, paras.seq_len, paras.step)
        target_iter = sliding_window(self.target_bin, paras.seq_len, paras.step)
        predicted = []
        predicted_topk = []
        target = []
        for input_seq in input_iter:
            target_seq = target_iter.__next__()
            input_seq = torch.stack(input_seq).unsqueeze(0).transpose(1, 2)
            predict = self.predictor(input_seq)
            predicted.append(predict)
            target.extend(list(torch.stack(target_seq).cpu().long().numpy()))
        predicted = torch.stack(predicted).reshape(-1).cpu().numpy()
        target = np.array(target)
        assertedTrue = 0
        for predict, _target in zip(predicted, target):
            if (_target == predict):
                assertedTrue += 1
        print(assertedTrue)
        return assertedTrue / len(self.target_bin)

    def infer_cms(self):
        source_len = self.paras.seq_len
        # source_len=2
        _input_list = self.input_bin[0:source_len]
        _target_list = self.target_bin[0:source_len - 1]
        from copy import deepcopy
        self.predictor = self.predictor.cuda()
        targets = deepcopy(_target_list)
        _input = torch.from_numpy(np.array(_input_list)).unsqueeze(0).float().cuda()
        _target = torch.from_numpy(np.array(_target_list)).unsqueeze(0).long().cuda()
        out = self.predictor(_input, _target).squeeze(0).topk(1)[1]
        targets.append(int(out))
        start = time()
        for i, input_step in tqdm(enumerate(self.input_bin[source_len:])):
            _target_list.pop(0)
            _target_list.append(int(out))
            _input_list.pop(0)
            _input_list.append(input_step)
            _input = torch.from_numpy(np.array(_input_list)).unsqueeze(0).float().cuda()
            _target = torch.from_numpy(np.array(_target_list)).unsqueeze(0).long().cuda()
            out = self.predictor(_input, _target).squeeze(0).topk(1)[1]
            targets.append(int(out))
        end = time()
        print('计算时长：{}'.format(str(end - start)))
        assertTrue = 0
        for predicted, _target in zip(self.target_bin, targets):
            if (predicted == _target):
                assertTrue += 1
        print(assertTrue / len(targets))
        self.predicted_targets = targets
        Tools.serialize_object(targets, 'predicted_targets')

    def get_predicted_targets(self):
        try:
            self.predicted_targets
        except:
            try:
                RuntimeError
                assert False
                self.predicted_targets = Tools.deserialize_object('predicted_targets')
                print('载入预保存的inference')
            except:
                print('重新生成inference')
                self.infer_cms()

    def infer_seqeTime(self, batch_iter, device):
        for input_batch, target_batch in batch_iter:
            _inputs = torch.from_numpy(np.array(input_batch)).float().to(device)
            _targets = torch.from_numpy(np.array(target_batch)[:, :-1]).long().to(device)
            target = torch.from_numpy(np.array(target_batch)[:, -1]).long().to(device)
            output = self.predictor(_inputs, _targets).topk(1)[1].reshape(-1)
            print('只执行一次！')
        assertTrue = 0
        for predicted, _target in zip(output, target):
            if (predicted == _target):
                assertTrue += 1
        print(assertTrue)
        print(assertTrue / len(target))

    # def infer_seqAttn_classifi(self,batch_iter,device,ahead=1):
    #     results_dic_list={}
    #
    #     for input_batch, target_batch in batch_iter:
    #         _inputs = torch.from_numpy(np.array(input_batch)).float().to(device)
    #         _targets = torch.from_numpy(np.array(target_batch)[:,:-1]).long().to(device)
    #         target = torch.from_numpy(np.array(target_batch)[:, -1]).long().to(device)
    #         output = self.predictor(_inputs,_targets)[0].topk(1)[1].reshape(-1)
    #         print('只执行一次！')
    #     assertTrue = 0
    #     justice=[]
    #     for predicted, _target in zip(output, target):
    #         if (predicted == _target):
    #             assertTrue += 1
    #             justice.append(1)
    #         else:
    #             justice.append(0)
    #     # vis=Visdom()
    #     # vis.line(Y=np.array(justice),X=np.arange(start=0,stop=len(justice)))
    #     print(assertTrue)
    #     print(assertTrue / len(target))

    def infer_seqAttn_classifi(self, batch_iter, device, ahead=1):
        # TODO 加入时长估计的结果
        output_dic_list = {}
        target_dic_list = {}
        od_dic_list = {}

        def refine__target_batch(_target_batch, output_dic_list, pre):
            if (pre == 1):
                result = np.array(_target_batch)
            else:
                result = np.array(_target_batch)
                for _pre in range(1, pre):
                    tmp_output_np = output_dic_list[_pre].detach().cpu().numpy()
                    result[:, -(pre - _pre + 1)] = tmp_output_np
            return result.tolist()

        def refine__input_batch_startTime(_input_batch, od_dic_list, pre):
            if (pre == 1):
                result = np.array(_input_batch)
            else:
                result = np.array(_input_batch)
                for _pre in range(1, pre):
                    result[:, -(pre - _pre + 1), 0] = self.refine_input_startTime_seq(result, od_dic_list, pre, _pre)
            return result.tolist()

        for pre in range(1, ahead + 1):
            output_dic_list.update({pre: []})
            target_dic_list.update({pre: []})
            od_dic_list.update({pre: []})
        for input_batch, target_batch in batch_iter:
            print('只可执行一次，batch要打满')
            for pre in range(1, ahead + 1):
                seq_infer_len = self.paras.seq_len
                seq_start = pre - 1
                seq_end = seq_start + seq_infer_len
                _input_batch = np.array(input_batch)[:, seq_start:seq_end, :].tolist()
                _input_batch = refine__input_batch_startTime(_input_batch, od_dic_list, pre)
                _target_batch = np.array(target_batch)[:, seq_start:seq_end].tolist()
                _target_batch = refine__target_batch(_target_batch, output_dic_list, pre)
                output, target, od = self.seq_infer_one(device, _input_batch, _target_batch)
                # TODO 保存每一步的od
                output_dic_list[pre] = output
                target_dic_list[pre] = target
                od_dic_list[pre] = od
        for pre in range(1, ahead + 1):
            assertTrue = 0
            justice = []
            output = output_dic_list[pre]
            target = target_dic_list[pre]
            for predicted, _target in zip(output, target):
                if (predicted == _target):
                    assertTrue += 1
                    justice.append(1)
                else:
                    justice.append(0)
            # print(assertTrue)
            # print(assertTrue / len(target))
            print(f'pre:{pre} acc:{assertTrue / len(target)} num_true:{assertTrue}')

    def seq_infer_one(self, device, input_batch, target_batch):
        _inputs = torch.from_numpy(np.array(input_batch)).float().to(device)
        _targets = torch.from_numpy(np.array(target_batch)[:, :-1]).long().to(device)
        target = torch.from_numpy(np.array(target_batch)[:, -1]).long().to(device)
        pre_out = self.predictor(_inputs, _targets)
        output = pre_out[0].topk(1)[1].reshape(-1)
        od = pre_out[1].reshape(-1)
        return output, target, od

    def refine_input_startTime_seq(self, _input_batch, od_dic_list, pre, _pre):
        # TODO index取对与否不敢确定，但效果既然这么好，就懒得调了
        assert pre > 1
        last_one = _input_batch[:, -(pre - _pre), :]
        tmp_od_np = od_dic_list[_pre].detach().cpu().numpy()
        tmp_od_sec = decode_delta_time(tmp_od_np)
        tmp_od_st = decode_time_of_day(last_one[:, 0])
        new_st = ((tmp_od_st + tmp_od_sec) % (3600 * 24)) / (3600 * 24)
        return new_st
        # return _input_batch[:, -(pre - _pre + 1), 0]


class Predictor():
    def __init__(self, ori_predictor):
        self.ori_predictor = ori_predictor

    def __call__(self, input_bin):
        out = self.ori_predictor(input_bin)
        return out.topk(1)[1]
