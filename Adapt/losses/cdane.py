# CDANELoss is from Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation
# Code gently borrowed from https://github.com/thuml/Batch-Spectral-Penalization

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1

class CDANELoss(nn.Module):
    def __init__(self):
        super(CDANELoss, self).__init__()

    def Entropy(self, input_):
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def forward(self, s_feat, s_logit, t_feat, t_logit, ad_net):
        feature = torch.cat([s_feat, t_feat], dim=0)
        logit = torch.cat([s_logit, t_logit], dim=0)
        softmax_out = F.softmax(logit, dim=-1)
        entropy = self.Entropy(softmax_out)
        coeff = ad_net.module.calc_coeff()

        softmax_output = softmax_out.detach()
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
        batch_size = softmax_output.size(0) // 2
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()

        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        loss = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()

        return loss, ('08. CDANE loss: ', loss.data.cpu().numpy())