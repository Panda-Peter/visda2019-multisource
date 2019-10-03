import os
import copy
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from lib.config import cfg

def get_dim_concat(net_id):
    in_dim = 0
    for inp in cfg.MODEL.INPUTS[net_id]:
        in_dim += cfg.MODEL.NET_DIM[inp]
    return in_dim

def get_dim_array(net_id):
    in_dims = []
    for inp in cfg.MODEL.INPUTS[net_id]:
        in_dims.append(cfg.MODEL.NET_DIM[inp])
    return in_dims

def get_dim(net_id):
    if cfg.MODEL.NETS[net_id] == 'MLP':
        return get_dim_concat(net_id)
    elif cfg.MODEL.NETS[net_id] == 'Bilinear' or cfg.MODEL.NETS[net_id] == 'FBilinear':
        return get_dim_array(net_id)
    else:
        return 0

def get_feats(feats, net_id):
    if len(cfg.MODEL.INPUTS[net_id]) == 1:
        return feats[cfg.MODEL.INPUTS[net_id][0]]

    feats_arr = []
    for inp in cfg.MODEL.INPUTS[net_id]:
        feats_arr.append(feats[inp])
    if cfg.MODEL.NETS[net_id] == 'MLP':
        return torch.cat(feats_arr, 1)
    elif cfg.MODEL.NETS[net_id] == 'Bilinear' or cfg.MODEL.NETS[net_id] == 'FBilinear':
        return feats_arr

def load_probpkl(domain):
    probs = []
    plabels = []
    for i in range(len(cfg.MODEL.NET_TYPE)):
        path = os.path.join(cfg.DATA_LOADER.DATA_ROOT, cfg.MODEL.NET_TYPE[i], domain + '_probs.pkl')
        pdata = pickle.load(open(path, 'rb'), encoding='bytes')
        probs.append(pdata['probs'])
        plabels.append(pdata['labels'])
    return probs, plabels

def filter_trg_plabels(full_feats, probs_arr):
    probs = probs_arr[0] * cfg.MODEL.WEIGHTS[0]
    weight_sum = cfg.MODEL.WEIGHTS[0]
    for i in range(1, len(probs_arr)):
        probs += probs_arr[i] * cfg.MODEL.WEIGHTS[i]
        weight_sum += cfg.MODEL.WEIGHTS[i]
    probs /= weight_sum

    plabels = np.zeros((probs.shape[0],), dtype='int')
    for i, p in enumerate(probs):
        idx = np.argmax(p)
        plabels[i] = idx

    select = [False] * len(probs)
    select = np.array(select)
    res_plabels = copy.copy(plabels)
    for c in range(cfg.MODEL.CLASS_NUM):
        prob_c = probs[:, c]
        index_sort = np.argsort(-prob_c)[:cfg.MODEL.MIN_CLS_NUM]
        select[index_sort] = True
        probs[index_sort, :] = -1
        res_plabels[index_sort] = c
        
    max_probs = np.array([probs[i, plabels[i]] for i in range(len(plabels))])
    select[max_probs >= cfg.MODEL.MIN_CONF] = True
    res_plabels = np.array([res_plabels[i] for i in range(len(select)) if select[i] == True])

    res_feats = []
    for k in range(len(cfg.MODEL.NET_TYPE)):
        res_feat = np.array([full_feats[k][i] for i in range(len(select)) if select[i] == True])
        res_feats.append(res_feat)

    return res_feats, res_plabels

def eval(epoch, name, probs, labels):
    correct = 0
    tick = 0
    subclasses_correct = np.zeros(cfg.MODEL.CLASS_NUM)
    subclasses_tick = np.zeros(cfg.MODEL.CLASS_NUM)
    preds = probs.argmax(axis=1)

    for i, pred in enumerate(preds):
        subclasses_tick[labels[i]] += 1
        if pred == labels[i]:
            correct += 1
            subclasses_correct[labels[i]] += 1
        tick += 1

    correct = correct * 1.0 / tick
    subclasses_result = np.divide(subclasses_correct, subclasses_tick)
    mean_class_acc = subclasses_result.mean()
    zeors_num = subclasses_result[subclasses_result == 0].shape[0]

    mean_acc_str = "*** Epoch {:d}, {:s}; mean class acc = {:.2%}, overall = {:.2%}, missing = {:d}".format(\
        epoch, name, mean_class_acc, correct, zeors_num)
    return mean_acc_str, preds

def write_list(result_folder, epoch, domain, preds):
    with open(os.path.join(result_folder, str(epoch) + '_' + domain + '.txt'), 'w') as fid:
        for v in preds:
            fid.write(str(v) + '\n')
