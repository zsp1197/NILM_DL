# Created by zhai at 2018/6/29
# Email: zsp1197@163.com

import numpy as np
from scipy.stats import entropy
from scipy.spatial import distance
class Folds():
    def __init__(self, inputs: np.array, targets: np.array):
        self.inputs = inputs
        self.targets = targets

    def cut_sequence(self, start_idx, end_idx, seq:np.array,concat=False):
        cutted = seq[start_idx:end_idx]
        remaining = [seq[0:start_idx], seq[end_idx:-1]]
        if(concat):
            remaining=np.concatenate(remaining)
        return cutted, remaining

    def seq_of_probabilities(self, seq_of_ints: np.array, ints: np.array):
        bins = list(range(min(ints), max(ints) + 2))
        hist, edges = np.histogram(seq_of_ints, bins, density=True)
        return hist

    def walk_seq_kl(self):
        # 看看最合适序列
        length = len(self.targets)
        win_len = length // 10
        theints=np.array(list(range(min(self.targets),max(self.targets)+1)))
        distances=[]
        start_idxes=[]
        min_distance=999999
        for start in range(length):
            if(start<2*win_len or start>length-1.5*win_len):
                continue
            cutted, remains = self.cut_sequence(start_idx=start, end_idx=start + win_len,seq=self.targets)
            theremains = np.concatenate(remains)
            hist_cutted=self.seq_of_probabilities(seq_of_ints=cutted,ints=theints)
            hist_remains=self.seq_of_probabilities(seq_of_ints=theremains,ints=theints)
            # distance=(entropy(hist_cutted,hist_remains)+entropy(hist_remains,hist_cutted))/2
            thedistance=distance.euclidean(hist_cutted,hist_remains)
            distances.append(thedistance)
            start_idxes.append(start)
            if(thedistance<min_distance):
                min_distance=thedistance
                best_train=theremains
                best_infer=cutted
                best_start=start
                best_end=start+win_len
        return best_start,best_end