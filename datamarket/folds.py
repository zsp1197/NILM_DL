# Created by zhai at 2018/6/29
# Email: zsp1197@163.com

import numpy as np
from scipy.stats import entropy
from scipy.spatial import distance
import pandas as pd

import Tools


class Folds():
    def __init__(self, inputs: np.array, targets: np.array):
        self.inputs = inputs
        self.targets = targets

    def cut_sequence(self, start_idx, end_idx, seq: np.array, concat=False):
        cutted = seq[start_idx:end_idx]
        remaining = [seq[0:start_idx], seq[end_idx:-1]]
        if (concat):
            remaining = np.concatenate(remaining)
        return cutted, remaining

    def seq_of_probabilities(self, seq_of_ints: np.array, ints: np.array):
        bins = list(range(min(ints), max(ints) + 2))
        hist, edges = np.histogram(seq_of_ints, bins, density=True)
        return hist

    def walk_seq_kl(self):
        # 看看最合适序列
        length = len(self.targets)
        win_len = length // 10
        theints = np.array(list(range(min(self.targets), max(self.targets) + 1)))
        distances = []
        start_idxes = []
        min_distance = 999999
        for start in range(length):
            if (start < 2 * win_len or start > length - 1.5 * win_len):
                continue
            cutted, remains = self.cut_sequence(start_idx=start, end_idx=start + win_len, seq=self.targets)
            theremains = np.concatenate(remains)
            hist_cutted = self.seq_of_probabilities(seq_of_ints=cutted, ints=theints)
            hist_remains = self.seq_of_probabilities(seq_of_ints=theremains, ints=theints)
            # distance=(entropy(hist_cutted,hist_remains)+entropy(hist_remains,hist_cutted))/2
            thedistance = distance.euclidean(hist_cutted, hist_remains)
            distances.append(thedistance)
            start_idxes.append(start)
            if (thedistance < min_distance):
                min_distance = thedistance
                best_train = theremains
                best_infer = cutted
                best_start = start
                best_end = start + win_len
        return best_start, best_end

    def vid_cutted(self, chunks_list, best_start, best_end):
        # start_bin = chunks_list[best_start]
        # end_bin = chunks_list[best_end]
        series = []
        series_idx = []
        for i, bin in enumerate(chunks_list[best_start:best_end + 1]):
            series.append(pd.Series(data=bin.total_value,
                                    index=pd.date_range(start=bin.start_time, end=bin.start_time + bin.delta_time,
                                                        freq='S')))
            series_idx.append(pd.Series(data=best_start + i,
                                        index=pd.date_range(start=bin.start_time, end=bin.start_time + bin.delta_time,
                                                            freq='S')))
        total_series = pd.concat(series)
        Tools.server_ps_plot(total_series)
        Tools.server_ps_plot(pd.concat(series_idx))

    def load_folds_on_the_fly_credientes(self, predictor, folds_inputs, folds_targets, criterion, device,paras):
        self.paras=paras
        self.predictor = predictor
        self.folds_inputs = folds_inputs
        self.folds_targets = folds_targets
        self.criterion = criterion
        self.device = device

    def folds_on_the_fly(self):
        predicted = self.predictor(self.folds_inputs)
        loss = self.criterion(predicted, self.folds_targets)
        return loss

    def folds_on_the_fly_2(self):
        batch_iter = self.get_batch(inputs=self.folds_inputs, targets=self.folds_targets,seq_len=self.paras.seq_len,
                                    num_per_batch=len(self.folds_inputs) + 999)
        import torch
        for input_batch, target_batch in batch_iter:
            _inputs = torch.from_numpy(np.array(input_batch)).float().to(self.device)
            _targets = torch.from_numpy(np.array(target_batch)[:, :-1]).long().to(self.device)
            target = torch.from_numpy(np.array(target_batch)[:, -1]).long().to(self.device)
            _predicted_logits, _predicted_ods = self.predictor(_inputs, _targets)
            loss, classifi_loss, od_loss = self.predictor.get_loss(_predicted_logits=_predicted_logits,
                                                                   target_idxs=target,
                                                                   _predicted_ods=_predicted_ods,
                                                                   target_ods=_inputs[:, -1, 1])
        return loss, classifi_loss, od_loss

    def get_batch(self, inputs, targets, num_per_batch, seq_len=0, step=1):
        '''
        最后一个batch可能数目不足num_per_batch
        :param inputs:
        :param targets:
        :param num_per_batch:
        :return:
        '''
        assert len(inputs) == len(targets)
        if (seq_len <= 0):
            seq_len = seq_len
        input_wins_iter = Tools.sliding_window(inputs, size=seq_len, step=step)
        target_wins_iter = Tools.sliding_window(targets, size=seq_len, step=step)
        input_batch = []
        target_batch = []
        num = 0
        for input_win in input_wins_iter:
            input_batch.append(input_win)
            target_batch.append(target_wins_iter.__next__())
            num += 1
            if (num == num_per_batch):
                yield input_batch, target_batch
                input_batch = []
                target_batch = []
                num = 0
        yield input_batch, target_batch
