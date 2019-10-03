import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss = F.nll_loss(F.log_softmax(x, dim=1), y)
        return loss