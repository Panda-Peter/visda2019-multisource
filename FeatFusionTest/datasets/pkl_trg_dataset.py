import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
import copy
from lib.config import cfg
import lib.utils as utils

class PklTrgDataset(data.Dataset):
    def __init__(self, domain, phase, cnt_pcls):
        self.cnt_pcls = cnt_pcls
        self.full_feats = []
        for net in cfg.MODEL.NET_TYPE:
            path = os.path.join(cfg.DATA_LOADER.DATA_ROOT, net, domain + '_' + phase + '.pkl')
            data = pickle.load(open(path, 'rb'), encoding='bytes')
            self.full_feats.append(data['feats'])

    def __len__(self):
        return len(self.full_feats[0])

    def filter_trg_plabels(self, probs):
        self.feats, self.labels = utils.filter_trg_plabels(self.full_feats, probs)
        self.clsmap = {}
        for c in range(cfg.MODEL.CLASS_NUM):
            self.clsmap[c] = []
        for i in range(len(self.labels)):
            self.clsmap[self.labels[i]].append(i)

    def __getitem__(self, index):
        indices = []
        for c in range(cfg.MODEL.CLASS_NUM):
            sidx = random.sample(self.clsmap[c], self.cnt_pcls)
            indices += sidx

        sfeats = []
        for i in range(len(cfg.MODEL.NET_TYPE)):
            sfeats.append(self.feats[i][indices, :])
        slabels = self.labels[indices]

        return sfeats, slabels