import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.config import cfg

class WeightEMA (object):
    def __init__(self, params, src_params, alpha=0.999):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
            
class Optimizer(nn.Module):
    def __init__(self, netG, netE):
        super(Optimizer, self).__init__()
        self.setup_optimizer(netG, netE)
        self.epoch = 0

    def setup_optimizer(self, netG, netE):
        params = []
        if netG is not None:
            for key, value in netG.named_parameters():
                if not value.requires_grad:
                    continue
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if "bias" in key:
                    lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                if lr > 0:
                    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if netE is not None:
            for key, value in netE.named_parameters():
                if not value.requires_grad:
                    continue
                lr = cfg.SOLVER.NEW_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if "bias" in key:
                    lr = cfg.SOLVER.NEW_LR * cfg.SOLVER.BIAS_LR_FACTOR
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
                if lr > 0:
                    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if cfg.SOLVER.TYPE == 'sgd':
            self.optimizer = torch.optim.SGD(params, 
                lr=cfg.SOLVER.BASE_LR, 
                momentum=cfg.SOLVER.MOMENTUM)
        elif cfg.SOLVER.TYPE == 'adam':
            self.optimizer = torch.optim.Adam(params,
                lr=cfg.SOLVER.BASE_LR)
        elif cfg.SOLVER.TYPE == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params,
                lr=cfg.SOLVER.BASE_LR, alpha=0.9, momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise NotImplementedError

        if cfg.SOLVER.LR_POLICY == 'fixed':
            self.scheduler = None
        elif cfg.SOLVER.LR_POLICY == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=cfg.SOLVER.STEP_SIZE, gamma=cfg.SOLVER.GAMMA)
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, epoch):
        if (self.scheduler is not None) and (epoch != self.epoch):
            self.epoch = epoch
            self.scheduler.step()
        self.optimizer.step()

    def get_lr(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr.append(param_group['lr'])
        lr = sorted(list(set(lr)))
        return lr