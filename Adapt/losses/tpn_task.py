# pytorch implementation for Transferrable Prototypical Networks for Unsupervised Domain Adaptation
# Sample-level discrepancy loss in Section 3.4 Task-specific Domain Adaptation
# https://arxiv.org/pdf/1904.11227.pdf

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg

class TpnTaskLoss(nn.Module):
    def __init__(self):
        super(TpnTaskLoss, self).__init__()

    def forward(self, src_feat, trg_feat, src_label, trg_label):
        labels = list(src_label.data.cpu().numpy())
        labels = list(set(labels))

        dim = src_feat.size(1)
        center_num = len(labels)
        u_s = torch.zeros(center_num, dim).cuda()
        u_t = torch.zeros(center_num, dim).cuda()
        u_st = torch.zeros(center_num, dim).cuda()

        for i, l in enumerate(labels):
            s_feat = src_feat[src_label == l]
            t_feat = trg_feat[trg_label == l]

            u_s[i, :] = s_feat.mean(dim=0)
            u_t[i, :] = t_feat.mean(dim=0)    
            u_st[i, :] = (s_feat.sum(dim=0) + t_feat.sum(dim=0)) / (s_feat.size(0) + t_feat.size(0))
        
        feats = torch.cat((src_feat, trg_feat), dim=0)
        P_s = torch.matmul(feats, u_s.t())
        P_t = torch.matmul(feats, u_t.t())
        P_st = torch.matmul(feats, u_st.t())

        loss_st = (F.kl_div(F.log_softmax(P_s, dim=-1), F.softmax(P_t, dim=-1), reduction='mean') + \
            F.kl_div(F.log_softmax(P_t, dim=-1), F.softmax(P_s, dim=-1), reduction='mean')) / 2
        loss_sst = (F.kl_div(F.log_softmax(P_s, dim=-1), F.softmax(P_st, dim=-1), reduction='mean') + \
            F.kl_div(F.log_softmax(P_st, dim=-1), F.softmax(P_s, dim=-1), reduction='mean')) / 2
        loss_tst = (F.kl_div(F.log_softmax(P_t, dim=-1), F.softmax(P_st, dim=-1), reduction='mean') + \
            F.kl_div(F.log_softmax(P_st, dim=-1), F.softmax(P_t, dim=-1), reduction='mean')) / 2
        tpn_task = (loss_st + loss_sst + loss_tst) / 3
        return tpn_task, ('04. tpn_task loss: ', tpn_task.data.cpu().numpy())




        