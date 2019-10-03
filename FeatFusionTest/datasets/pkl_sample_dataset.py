import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
import copy
from lib.config import cfg

class PklSampleDataset(data.Dataset):
    def __init__(self, domain, phase, cnt_pcls):
        self.cnt_pcls = cnt_pcls
        self.feats = []
        self.labels = None
        for net in cfg.MODEL.NET_TYPE:
            path = os.path.join(cfg.DATA_LOADER.DATA_ROOT, net, domain + '_' + phase + '.pkl')
            data = pickle.load(open(path, 'rb'), encoding='bytes')
            self.feats.append(data['feats'])
            self.labels = data['labels']

        self.clsmap = {}
        for c in range(cfg.MODEL.CLASS_NUM):
            self.clsmap[c] = []
        for i in range(len(self.labels)):
            self.clsmap[self.labels[i]].append(i)

    def __len__(self):
        return len(self.labels)

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