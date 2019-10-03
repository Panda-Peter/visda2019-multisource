# BSPLoss is from Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation
# Code gently borrowed from https://github.com/thuml/Batch-Spectral-Penalization

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg

class BSPLoss(nn.Module):
    def __init__(self):
        super(BSPLoss, self).__init__()

    def forward(self, s_feat, t_feat):
        _, s_s, _ = torch.svd(s_feat)
        _, s_t, _ = torch.svd(t_feat)
        loss = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        return loss, ('07. BSP loss: ', loss.data.cpu().numpy())