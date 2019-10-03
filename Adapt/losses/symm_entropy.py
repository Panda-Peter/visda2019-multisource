# Unofficial pytorch implementation for Symmetric Cross Entropy for Robust Learning with Noisy Labels
# https://arxiv.org/pdf/1908.06112.pdf

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg

class SymmEntropyLoss(nn.Module):
    def __init__(self):
        super(SymmEntropyLoss, self).__init__()
        self.alpha = cfg.LOSSES.ALPHA
        self.beta = cfg.LOSSES.BETA
        self.A = cfg.LOSSES.A

    def forward(self, x, y):
        loss_ce = F.nll_loss(F.log_softmax(x, dim=1), y)
        P = F.softmax(x, dim=1)
        P = P.scatter(1, y.unsqueeze(-1), 0.0)
        P = P * self.A
        loss_rce = -P.sum(1).mean(0)
        loss = self.alpha * loss_ce + self.beta * loss_rce
        return loss, ('03. symmetric cross entropy loss: ', loss.data.cpu().numpy())