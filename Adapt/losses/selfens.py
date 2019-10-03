# Self-ensembling for visual domain adaptation
# https://arxiv.org/pdf/1706.05208.pdf
# Code gently borrowed from https://github.com/Britefury/self-ensemble-visual-domain-adapt-photo/

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg

class SelfEnsLoss(nn.Module):
    def __init__(self):
        super(SelfEnsLoss, self).__init__()

    def forward(self, stu_logits_out, tea_logits_out, confidence_thresh, n_classes):
        stu_out = F.softmax(stu_logits_out, dim=1)
        tea_out = F.softmax(tea_logits_out, dim=1)

        # Augmentation loss
        conf_tea = torch.max(tea_out, 1)[0]
        conf_mask = torch.gt(conf_tea, confidence_thresh).float()

        d_aug_loss = stu_out - tea_out
        aug_loss = d_aug_loss * d_aug_loss

        aug_loss = torch.mean(aug_loss, 1) * conf_mask
        loss = torch.mean(aug_loss)

        return loss, ('05. self ensemble loss: ', loss.data.cpu().numpy())