# Created by zhai at 2018/6/29
from unittest import TestCase
import numpy as np
import Tools
import pandas as pd
# Email: zsp1197@163.com
from datamarket.folds import Folds
from test.test_seq2Seq import TestSeq2Seq


class TestFolds(TestCase):
    def setUp(self):
        # self.targets=np.array(Tools.deserialize_object('targets_list.store'))
        # self.inputs=np.array(Tools.deserialize_object('inputs_list.store'))
        # self.ints=np.array(list(range(min(self.targets),max(self.targets)+1)))
        # self.folds=Folds(self.inputs,self.targets)
        pass

    def test_cut_sequence(self):
        inputs=[(i,i*10,i*100) for i in list(range(100))]
        targets=list(range(100))
        best_start,best_end=15,25
        cutted_inputs, remaining_inputs = self.folds.cut_sequence(start_idx=best_start, end_idx=best_end,
                                                             seq=np.array(inputs),
                                                             concat=True)
        cutted_targets, remaining_targets = self.folds.cut_sequence(start_idx=best_start, end_idx=best_end,
                                                               seq=np.array(targets),
                                                               concat=True)
        print()

    def test_seq_of_probabilities(self):
        print(self.folds.seq_of_probabilities(self.targets,self.ints))

    def test_walk_seq_kl(self):
        self.folds.walk_seq_kl()

    def get_credients(self):
        self.t = TestSeq2Seq()
        self.t.prepareCredients()


