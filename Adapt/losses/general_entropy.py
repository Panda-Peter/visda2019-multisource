# Unofficial pytorch implementation for Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels
# https://arxiv.org/pdf/1805.07836.pdf

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg

class GeneralEntropyLoss(nn.Module):
    def __init__(self):
        super(GeneralEntropyLoss, self).__init__()
        self.k = cfg.LOSSES.GXENT_K
        self.q = cfg.LOSSES.GXENT_Q
        self.trunc = (1 - pow(self.k, self.q)) / self.q

    def forward(self, x, y):
        prob = F.softmax(x, dim=1)
        y_labels = y.unsqueeze(-1)
        P = torch.gather(prob, 1, y_labels)
        mask = (P <= self.k).type(torch.cuda.FloatTensor)

        loss = ((1 - torch.pow(P, self.q)) / self.q) * (1 - mask) + self.trunc * mask
        loss = loss.mean()
        return loss, ('02. general xent loss: ', loss.data.cpu().numpy())