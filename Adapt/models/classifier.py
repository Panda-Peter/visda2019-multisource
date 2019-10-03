import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import math
from lib.config import cfg

class FC(nn.Module):
    def __init__(self, in_dim, out_dim, distributed):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.distributed = distributed

    def forward(self, x):
        x = self.fc(x)
        #if self.distributed:
        #    x = x.unsqueeze(-1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        #if self.distributed:
        #    x = x.squeeze(-1)
        return x

class Classifier(nn.Module):
    def __init__(self, class_num=345, distributed=False):
        super(Classifier, self).__init__()
        self.fc1 = FC(cfg.MODEL.IN_DIM, cfg.MODEL.EMBED_DIM, distributed)
        self.fc2 = FC(cfg.MODEL.EMBED_DIM, cfg.MODEL.EMBED_DIM, distributed)
        self.fc3 = nn.Linear(cfg.MODEL.EMBED_DIM, class_num)

    def forward(self, x):
        fc1 = self.fc1(x)
        fc2 = self.fc2(fc1)
        cls_logit = self.fc3(fc2)
        return fc2, cls_logit