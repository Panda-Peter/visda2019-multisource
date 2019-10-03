import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
import copy
from lib.config import cfg

class PklDataset(data.Dataset):
    def __init__(self, domain, phase):
        self.feats = []
        self.labels = None
        for net in cfg.MODEL.NET_TYPE:
            path = os.path.join(cfg.DATA_LOADER.DATA_ROOT, net, domain + '_' + phase + '.pkl')
            data = pickle.load(open(path, 'rb'), encoding='bytes')
            self.feats.append(data['feats'])
            self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feats = []
        for i in range(len(cfg.MODEL.NET_TYPE)):
            feats.append(self.feats[i][index, :])
        label = self.labels[index]
        return feats, label
