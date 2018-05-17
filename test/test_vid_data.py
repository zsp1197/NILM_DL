# Created by zhai at 2018/5/15
# Email: zsp1197@163.com
from unittest import TestCase

from sklearn.manifold import TSNE

from Parameters import Parameters
from datamarket.Data_store import Data_store
from datamarket.House import House
from datamarket.StateManager import DLStateManager
from pytorchGo.s2v import S2V
from pytorchGo.seq2seq import *
import numpy as np
from visdom import Visdom


class TestVid(TestCase):
    def prepareCredients(self):
        self.house = House(data_store=Data_store('../data/allappliances4.hdf5'), house_idx='DEBUG',
                           parameters=Parameters(), check_good_sections=False)
        self.house.data_store.store.close()
        self.get_ps()
        self.dl_bins_inputs_targets()

    def get_ps(self):
        store = pd.HDFStore('../data/mainMeter_ps_labels', mode='r')
        keys = store.keys()
        self.label_series = deepcopy(store[keys[0]])
        store.close()
        store = pd.HDFStore('../data/mainMeter_ps', mode='r')
        keys = store.keys()
        self.main_meter_ps = deepcopy(store[keys[0]])
        store.close()
        print('you can proceed with HDFstore now')

    def dl_bins_inputs_targets(self):
        # self.get_ps()
        main_meter_ps_sax = self.house.sax(ps=self.main_meter_ps)
        self.dlsm = DLStateManager(self.label_series)
        self.inputs, self.targets = self.house.dl_bins_inputs_targets(main_meter_ps=self.main_meter_ps,
                                                                      main_meter_labels_ps=self.dlsm.main_meter_labels_idx2_ps)

    def refine_main_meter_labels_idx2_ps(self, main_meter_labels_idx2_ps):
        main_meter_labels_idx2_ps_refine = []
        temp = -999
        for i, var in enumerate(main_meter_labels_idx2_ps):
            if (var != temp):
                main_meter_labels_idx2_ps_refine.append(var)
                temp = var
        return main_meter_labels_idx2_ps_refine

    def get_transmission(self, totrans):
        size = len(self.dlsm.idx2stateidx)
        trans = np.zeros((size, size))
        for i, var in enumerate(totrans):
            if (i == 0):
                last = var
                continue
            trans[last][var] += 1
            trans = F.softmax(torch.Tensor(np.array(trans)), dim=1)
        return trans.numpy()

    def test_get_transmission(self):
        self.prepareCredients()
        trans = self.get_transmission(self.refine_main_meter_labels_idx2_ps(self.dlsm.main_meter_labels_idx2_ps))
        best_trans = []
        torch.nn.Embedding(len(self.dlsm.idx2stateidx) + 1, 10).load_state_dict('s2v_embedding_states.pth')
        for transi in trans:
            best_trans = transi.argmax()

    def test_vis_test_get_transmission(self):
        self.prepareCredients()
        trans = self.get_transmission(self.refine_main_meter_labels_idx2_ps(self.dlsm.main_meter_labels_idx2_ps))
        embedding = torch.nn.Embedding(len(self.dlsm.idx2stateidx) + 1, 10)
        embedding.load_state_dict(torch.load('s2v_embedding_states.pth'))
        points_high = np.array([embedding(torch.LongTensor([i])).reshape(-1).detach().numpy() for i in
                                list(self.dlsm.idx2stateidx.keys())])
        embedded_2 = TSNE(n_components=3).fit_transform(points_high)
        vis = Visdom()
        vis.scatter(
            X=embedded_2
        )