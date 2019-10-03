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
import losses

class Bilinear(nn.Module):
    def __init__(self, in_dim=[2048], out_dim=1000, **kwargs):
        super(Bilinear, self).__init__()
        fc1s = []
        fc2s = []
        for indim in in_dim:
            fc1 = nn.Sequential(
                nn.Linear(indim, out_dim),
                nn.BatchNorm1d(out_dim, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout()
            )
            fc2 = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim, affine=True),
                nn.ReLU(inplace=True),
                #nn.Dropout()
            )
            fc1s.append(fc1)
            fc2s.append(fc2)
        self.fc1s = nn.ModuleList(fc1s)
        self.fc2s = nn.ModuleList(fc2s)  
        self.fc3 = nn.Linear(out_dim, cfg.MODEL.CLASS_NUM)

        self.xent_loss = losses.create('CrossEntropy')
        if cfg.LOSSES.TRG_GXENT_WEIGHT > 0:
            self.gxent_loss = losses.create('GeneralEntropy')

    def forward(self, src_feats, src_labels, trg_feats, trg_labels):
        feats_arr = []
        for i, feat in enumerate(src_feats):
            feat = self.fc1s[i](feat)
            feat = self.fc2s[i](feat)
            feats_arr.append(feat)
        feats = feats_arr[0]
        for i in range(1, len(src_feats)):
            feats = feats * feats_arr[i]
        src_logits = self.fc3(feats)
        
        feats_arr = []
        for i, feat in enumerate(trg_feats):
            feat = self.fc1s[i](feat)
            feat = self.fc2s[i](feat)
            feats_arr.append(feat)
        feats = feats_arr[0]
        for i in range(1, len(trg_feats)):
            feats = feats * feats_arr[i]
        trg_logits = self.fc3(feats)
        return self.cal_loss(src_logits, src_labels, trg_logits, trg_labels)

    def test(self, trg_feats):
        feats_arr = []
        for i, feat in enumerate(trg_feats):
            feat = self.fc1s[i](feat)
            feat = self.fc2s[i](feat)
            feats_arr.append(feat)
        feats = feats_arr[0]
        for i in range(1, len(trg_feats)):
            feats = feats * feats_arr[i]
        logits = self.fc3(feats)
        probs = F.softmax(logits, dim=-1)
        return probs.data.cpu().numpy()

    def cal_loss(self, src_logits, src_labels, trg_logits, trg_labels):
        xent_loss =  self.xent_loss(src_logits, src_labels)
        loss_info = { '01. Cross Entropy Loss: ': xent_loss.data.cpu().numpy() }
        loss = xent_loss

        if cfg.LOSSES.TRG_GXENT_WEIGHT > 0:
            gxent_loss = self.gxent_loss(trg_logits, trg_labels)
            loss_info['02. General Cross Entropy Loss: '] = gxent_loss.data.cpu().numpy()
            loss += cfg.LOSSES.TRG_GXENT_WEIGHT * gxent_loss
            
        return loss, loss_info
