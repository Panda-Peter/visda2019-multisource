# Unofficial pytorch implementation for Contrastive Adaptation Network for Unsupervised Domain Adaptation
# https://arxiv.org/pdf/1901.00976.pdf

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg

class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()
        self.kernel_mul = cfg.LOSSES.KERNEL_MUL
        self.kernel_num = cfg.LOSSES.KERNEL_NUM
        self.fix_sigma = None

    def guassian_kernel(self, x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        n_samples = x_size + y_size
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        L2_distance = ((tiled_x-tiled_y)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) + cfg.EPS for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def compute_mmd(self, x, y):
        x_kernel = self.guassian_kernel(x, x, self.kernel_mul, self.kernel_num, self.fix_sigma)
        y_kernel = self.guassian_kernel(y, y, self.kernel_mul, self.kernel_num, self.fix_sigma)
        xy_kernel = self.guassian_kernel(x, y, self.kernel_mul, self.kernel_num, self.fix_sigma)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    def forward(self, x, y, x_labels, y_labels):
        if y_labels is None:            
            mmd = self.compute_mmd(x, y)
            return mmd, ('03. mmd loss: ', mmd.data.cpu().numpy())
        else:
            labels = list(x_labels.data.cpu().numpy())
            labels = list(set(labels))
            pos_num = len(labels)
            neg_num = len(labels) * (len(labels) - 1)
            
            x_c = []
            y_c = []
            n_labels = len(labels)
            for label in labels:
                x_c.append(x[x_labels == label])
                y_c.append(y[y_labels == label])

            xk_c = torch.zeros(n_labels).cuda()
            yk_c = torch.zeros(n_labels).cuda()
            xyk_c = torch.zeros(n_labels, n_labels).cuda()
            for i in range(n_labels):
                x_kernel = self.guassian_kernel(x_c[i], x_c[i], self.kernel_mul, self.kernel_num, self.fix_sigma)
                xk_c[i] = x_kernel.mean()

                y_kernel = self.guassian_kernel(y_c[i], y_c[i], self.kernel_mul, self.kernel_num, self.fix_sigma)
                yk_c[i] = y_kernel.mean()

                for j in range(n_labels):
                    xy_kernel = self.guassian_kernel(x_c[i], y_c[j], self.kernel_mul, self.kernel_num, self.fix_sigma)
                    xyk_c[i, j] = xy_kernel.mean()

            xk_c_sum = xk_c.sum()
            yk_c_sum = yk_c.sum()
            xyk_c_diag = torch.eye(n_labels, n_labels).cuda() * xyk_c
            xyk_c_antidiag = (1 - torch.eye(n_labels, n_labels).cuda()) * xyk_c

            mmd = (xk_c_sum + yk_c_sum - 2 * xyk_c_diag.sum()) / pos_num
            mmd -= cfg.LOSSES.MMD_NEG_WEIGHT * ( (n_labels - 1) * (xk_c_sum + yk_c_sum) - 2 * xyk_c_antidiag.sum() )  / neg_num
            
            return mmd, ('03. mmd loss: ', mmd.data.cpu().numpy())