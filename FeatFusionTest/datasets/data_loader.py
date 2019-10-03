import os
import re
import numpy as np
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.pkl_dataset import PklDataset
from datasets.pkl_sample_dataset import PklSampleDataset

def pkl_sample_collate(batch):
    bfeats, blabels = zip(*batch)
    feats = []
    labels = torch.from_numpy(blabels[0])
    for n in range(len(cfg.MODEL.NET_TYPE)):
        feats.append(torch.from_numpy(bfeats[0][n]))
    return feats, labels

def load_train(domain, cnt_pcls):
    dataset = PklSampleDataset(domain, 'train', cnt_pcls)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
        shuffle=True, num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        collate_fn=pkl_sample_collate)
    return loader

def load_trg_train(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
        shuffle=True, num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        collate_fn=pkl_sample_collate)
    return loader

def pkl_collate(batch):
    bfeats, blabels = zip(*batch)
    feats = []
    labels = torch.from_numpy(np.array([b for b in blabels]))
    for n in range(len(cfg.MODEL.NET_TYPE)):
        feats.append(torch.cat([torch.from_numpy(b[n]).unsqueeze(0) for b in bfeats], 0))
    return feats, labels

def load_test(domain, phase):
    dataset = PklDataset(domain, phase)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last=False, pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=pkl_collate)
    return loader