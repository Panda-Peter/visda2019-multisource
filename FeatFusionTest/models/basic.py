import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from lib.config import cfg
import losses

class Basic(nn.Module):
    def __init__(self, in_dim=2048, out_dim=1000, **kwargs):
        super(Basic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.xent_loss = losses.create('CrossEntropy')
        if cfg.LOSSES.TRG_GXENT_WEIGHT > 0:
            self.gxent_loss = losses.create('GeneralEntropy')

    def Forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def cal_loss(self, src_logits, src_labels, trg_logits, trg_labels):
        xent_loss =  self.xent_loss(src_logits, src_labels)
        loss_info = { '01. Cross Entropy Loss: ': xent_loss.data.cpu().numpy() }
        loss = xent_loss

        if cfg.LOSSES.TRG_GXENT_WEIGHT > 0:
            gxent_loss = self.gxent_loss(trg_logits, trg_labels)
            loss_info['02. General Cross Entropy Loss: '] = gxent_loss.data.cpu().numpy()
            loss += cfg.LOSSES.TRG_GXENT_WEIGHT * gxent_loss

        return loss, loss_info

