import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from lib.config import cfg
from models.basic import Basic

class MLP(Basic):
    def __init__(self, in_dim=2048, out_dim=1000, **kwargs):
        super(MLP, self).__init__(in_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, cfg.MODEL.CLASS_NUM)

    def forward(self, src_feats, src_labels, trg_feats, trg_labels):
        feats = self.Forward(src_feats)
        src_logits = self.fc3(feats)

        feats = self.Forward(trg_feats)
        trg_logits = self.fc3(feats)
        return self.cal_loss(src_logits, src_labels, trg_logits, trg_labels)

    def test(self, feats):
        feats = self.Forward(feats)
        logits = self.fc3(feats)
        probs = F.softmax(logits, dim=-1)
        return probs.data.cpu().numpy()
