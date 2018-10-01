# Created by zhai at 2018/9/27
# Email: zsp1197@163.com
import os

import datetime
import torch
from time import time
from unittest import TestCase
import random
from tqdm import tqdm

import Tools
from Parameters import Parameters
from pytorchGo.models_lab import SeqAttn
import numpy as np

use_cuda = torch.cuda.is_available()
if (use_cuda):
    # device = torch.device("cuda:3")
    try:
        import GPUtil
        if(len(GPUtil.getAvailable(maxLoad=0.1,limit=torch.cuda.device_count()))==torch.cuda.device_count()):
            # 全都比较闲就随便选一个
            device = torch.device("cuda:{0}".format(random.choice(range(torch.cuda.device_count()))))
        else:
            gpu_nums=GPUtil.getAvailable(order='memory', limit=torch.cuda.device_count())
            device = torch.device("cuda:{0}".format(gpu_nums[0]))
    except:
        device = torch.device("cuda:{0}".format(random.choice(range(torch.cuda.device_count()))))
else:
    device = torch.device('cpu')

class TestAmpds(TestCase):
    def prepareCredients(self):
        from pytorchGo.seq2seq import Seq2Seq_mine
        self.super_SS, self.super_SS_idxes, self.ss_naive_idxes_dict, self.ss_idxes_dict, self.ss_idxes_value_dict = Tools.deserialize_object(
            '../data/prepareAmpds/ampds_ss.tuple')
        self.inputs,self.targets=Tools.deserialize_object('../data/prepareAmpds/ampds_inputs_targets_20')
        self.paras=Parameters()
        self.seq2seq=Seq2Seq_mine(None, None, self.paras,debug=True)
        self.output_size=len(self.ss_idxes_dict)+1

    def test_SeqAttn(self):
        self.prepareCredients()
        min_loss = 999
        seqAttn = SeqAttn(target_num=self.output_size).to(device)
        opt = torch.optim.Adam(seqAttn.parameters(), lr=self.paras.learning_rate)
        sp = Tools.ServerUpdateTracePlot()
        sp_class = Tools.ServerUpdateTracePlot()
        sp_od = Tools.ServerUpdateTracePlot()
        totallosses, classifi_loss_list, od_loss_list = [], [], []
        ifFolds = False

        # folds
        if (ifFolds):
            self.inputs, self.targets = self.folds(self.inputs, self.targets, True)

        for epoch in tqdm(range(self.paras.epoch)):
            # TODO batch大小写死的！
            batch_iter = self.seq2seq.get_batch(inputs=self.inputs, targets=self.targets,
                                                num_per_batch=len(self.inputs)//4)
            for input_batch, target_batch in batch_iter:
                _inputs = torch.from_numpy(np.array(input_batch)).float().to(device)
                _targets = torch.from_numpy(np.array(target_batch)[:, :-1]).long().to(device)
                target = torch.from_numpy(np.array(target_batch)[:, -1]).long().to(device)
                _predicted_logits, _predicted_ods = seqAttn(_inputs, _targets)
                loss, classifi_loss, od_loss = seqAttn.get_loss(_predicted_logits=_predicted_logits, target_idxs=target,
                                                                _predicted_ods=_predicted_ods,
                                                                target_ods=_inputs[:, -1, 1])
                opt.zero_grad()
                loss.backward()
                opt.step()
                totallosses.append(loss.cpu().detach().numpy())
                classifi_loss_list.append(classifi_loss.cpu().detach().numpy())
                od_loss_list.append(od_loss.cpu().detach().numpy())
            if (epoch % 2 == 0):
                sp.update(y=loss.data, xlabel=f'total loss_{self.paras.seq_len}_{self.paras.trade_off_classifi}')
                sp_class.update(y=classifi_loss.data, xlabel=f'classifi_loss_{self.paras.seq_len}_{self.paras.trade_off_classifi}')
                sp_od.update(y=od_loss.data, xlabel=f'od_loss_{self.paras.seq_len}_{self.paras.trade_off_classifi}')
                if (loss.data.cpu().numpy() < min_loss):
                    min_loss = loss.data.cpu().numpy()
                    print(
                        f'save_{epoch} loss={str(min_loss)} classLoss={str(classifi_loss.data.cpu().numpy())} odLoss={str(od_loss.data.cpu().numpy())}')
                    if (ifFolds):
                        torch.save(seqAttn.state_dict(),
                                   Tools.create_parent_directory(f'../pths/ampds/folds/seqAttnEmbed4_{self.paras.seq_len}_{self.paras.trade_off_classifi}.pth'))
                    else:
                        # TODO 删掉 _20
                        torch.save(seqAttn.state_dict(),
                                   Tools.create_parent_directory(f'../pths/ampds_20/seqAttnEmbed4_{self.paras.seq_len}_{self.paras.trade_off_classifi}.pth'))
                if (epoch % 50 == 0):
                    losses = (totallosses, classifi_loss_list, od_loss_list)
                    if (ifFolds):
                        Tools.serialize_object(losses,
                                               Tools.create_parent_directory(f'../4paper/dlflex/folds/ampds/losses_{self.paras.trade_off_classifi}/losses_{self.paras.seq_len}.list'))
                    else:
                        Tools.serialize_object(losses,
                                               Tools.create_parent_directory(f'../4paper/dlflex/ampds/losses_{self.paras.trade_off_classifi}/losses_{self.paras.seq_len}.list'))

    def test_predict_SeqAttn_multiple(self):
        self.prepareCredients()
        trade_off_classifi = 0.05
        ahead = 7
        ifFolds = False
        for seq_len in range(32):
            self.paras.seq_len=seq_len
            if (ifFolds):
                file = f'../pths/ampds/folds/seqAttnEmbed4_{seq_len}_{trade_off_classifi}.pth'
            else:
                # TODO
                file = f'../pths/ampds_20/seqAttnEmbed4_{seq_len}_{trade_off_classifi}.pth'
            if (not os.path.exists(file)):
                continue
            print(file)
            past = datetime.datetime.now()
            seqAttn = SeqAttn(target_num=self.output_size).to(device)
            seqAttn.load_state_dict(torch.load(file))
            seqAttn.eval()

            if (ifFolds):
                inputs, targets = self.folds(self.inputs, self.targets, False)
            else:
                inputs, targets = self.inputs, self.targets
            # 处理判断接下来预判断多个的情况
            batch_iter = self.seq2seq.get_batch(inputs=inputs, targets=targets,
                                                num_per_batch=self.paras.batch_size,
                                                seq_len=self.paras.seq_len + ahead - 1)
            from pytorchGo.Infer import Infer
            infer = Infer()
            infer.credients(input_bin=self.inputs, target_bin=self.targets, predictor=seqAttn,paras=self.paras)
            infer.infer_seqAttn_classifi(batch_iter=batch_iter, device=device, ahead=ahead,
                                         r2_dict=self.ss_idxes_value_dict,bigIter=True)
            now = datetime.datetime.now()
            elapsedTIme = now - past
            print(f'使用时间：{elapsedTIme}')