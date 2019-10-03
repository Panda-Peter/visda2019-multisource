import os
import copy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
from lib.config import cfg
from torch.autograd import Variable

def loadlines(path):
    paths = []
    labels = []
    with open(path, 'r') as fid:
        for line in fid:
            data = line.strip().split(' ')
            paths.append(data[0])
            labels.append(int(data[1]))
    return paths, labels

def sync_labels(label):
    g_labels = [torch.ones_like(label) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(g_labels, label)
    gf_labels = torch.cat(g_labels, dim=0)
    return gf_labels

def sync_tensor(vec):
    g_vec = [torch.ones_like(vec) for _ in range(dist.get_world_size())]  
    torch.distributed.all_gather(g_vec, vec)
    gf_vec = vec

    for i in range(0, dist.get_rank()):
        gf_vec = torch.cat((g_vec[dist.get_rank() - i - 1].detach(), gf_vec), dim=0)
                
    for i in range(dist.get_rank()+1, dist.get_world_size()):
        gf_vec = torch.cat((gf_vec, g_vec[i].detach()), dim=0)
    return gf_vec

def broadcast_tensor(vec):
    torch.distributed.broadcast(vec, 0)
    vec = vec.data.cpu().numpy()
    return vec

def broadcast_numpy(vec):
    vec = torch.tensor(vec, device="cuda")
    torch.distributed.broadcast(vec, 0)
    vec = vec.data.cpu().numpy()
    return vec

def load_trg_plabels():
    if len(cfg.DATA_LOADER.TRG_PSEUDOLABELS) == 0:
        return None, None
    else:
        pdata = pickle.load(open(cfg.DATA_LOADER.TRG_PSEUDOLABELS, 'rb'), encoding='bytes')
        probs = pdata['probs']
        plabels = pdata['labels']
        return probs, plabels

def filter_trg_plabels(trg_paths, probs, plabels):
    if probs is None and plabels is None:
        return trg_paths, None, None

    select = [False] * len(probs)
    select = np.array(select)
    res_plabels = copy.copy(plabels)
    res_probs = copy.copy(probs)
    for c in range(cfg.MODEL.CLASS_NUM):
        prob_c = probs[:, c]
        index_sort = np.argsort(-prob_c)[:cfg.MODEL.MIN_CLS_NUM]
        select[index_sort] = True
        probs[index_sort, :] = -1
        res_plabels[index_sort] = c
        
    max_probs = np.array([probs[i, plabels[i]] for i in range(len(plabels))])
    select[max_probs >= cfg.MODEL.MIN_CONF] = True
    res_trg_paths = [trg_paths[i] for i in range(len(select)) if select[i] == True]
    res_plabels = [res_plabels[i] for i in range(len(select)) if select[i] == True]
    res_probs = np.array([res_probs[i] for i in range(len(select)) if select[i] == True], dtype='float32')
    res_probs = Variable(torch.from_numpy(res_probs).cuda(), requires_grad=False)
    return res_trg_paths, res_probs, res_plabels