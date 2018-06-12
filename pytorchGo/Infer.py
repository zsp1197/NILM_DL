# Created by zhai at 2018/5/29
# Email: zsp1197@163.com
import numpy as np
import torch
from tqdm import tqdm

from Parameters import Parameters
from Tools import sliding_window


class Infer():
    def credients(self, input_bin, target_bin, predictor):
        # assert len(input_bin) == len(target_bin)
        self.input_bin = input_bin
        self.target_bin = target_bin
        self.predictor = predictor

    def infer(self):
        predicted = self.predictor(self.input_bin)
        assert len(predicted) == len(self.target_bin)
        assertedTrue = 0
        for predict, target in zip(predicted, self.target_bin):
            if (predict == target):
                assertedTrue += 1
        return assertedTrue / len(self.target_bin)

    def infer_batch_seq(self,paras=Parameters()):
        input_iter = sliding_window(self.input_bin, paras.seq_len, paras.step)
        target_iter = sliding_window(self.target_bin, paras.seq_len, paras.step)
        predicted = []
        predicted_topk = []
        target = []
        for input_seq in input_iter:
            target_seq = target_iter.__next__()
            input_seq = torch.stack(input_seq).unsqueeze(0).transpose(1,2)
            predict = self.predictor(input_seq)
            predicted.append(predict)
            target.extend(list(torch.stack(target_seq).cpu().long().numpy()))
        predicted=torch.stack(predicted).reshape(-1).cpu().numpy()
        target=np.array(target)
        assertedTrue = 0
        for predict, _target in zip(predicted, target):
            if (_target == predict):
                assertedTrue += 1
        print(assertedTrue)
        return assertedTrue / len(self.target_bin)

    def infer_cms(self,paras=Parameters()):
        source_len=paras.seq_len
        # source_len=3
        _input_list=self.input_bin[0:source_len]
        _target_list=self.target_bin[0:source_len-1]
        from copy import deepcopy
        self.predictor=self.predictor.cuda()
        targets=deepcopy(_target_list)
        _input=torch.from_numpy(np.array(_input_list)).unsqueeze(0).float().cuda()
        _target=torch.from_numpy(np.array(_target_list)).unsqueeze(0).long().cuda()
        out=self.predictor(_input,_target).squeeze(0).topk(1)[1]
        targets.append(int(out))
        for i,input_step in tqdm(enumerate(self.input_bin[source_len:])):
            _target_list.pop(0)
            _target_list.append(int(out))
            _input_list.pop(0)
            _input_list.append(input_step)
            _input = torch.from_numpy(np.array(_input_list)).unsqueeze(0).float().cuda()
            _target = torch.from_numpy(np.array(_target_list)).unsqueeze(0).long().cuda()
            out = self.predictor(_input, _target).squeeze(0).topk(1)[1]
            targets.append(int(out))
        assertTrue = 0
        for predicted, _target in zip(self.target_bin, targets):
            if (predicted == _target):
                assertTrue += 1
        print(assertTrue / len(targets))



class Predictor():
    def __init__(self, ori_predictor):
        self.ori_predictor=ori_predictor

    def __call__(self, input_bin):
        out=self.ori_predictor(input_bin)
        return out.topk(1)[1]