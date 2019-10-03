import os
import datasets.list_loader as list_loader
import datasets.online_loader as online_loader
import torch
from torchvision import transforms
import lib.utils as utils
from lib.config import cfg
import math
import torch.distributed as dist
import samplers.distributed
import datasets.custom_transforms as custom_transforms

def get_transform():
    trans = []

    if cfg.AUG.RESIZE[0] > 0 and cfg.AUG.RESIZE[1] > 0:
        trans.append(transforms.Resize(cfg.AUG.RESIZE))
    if cfg.AUG.V_FLIP > 0:
        trans.append(transforms.RandomVerticalFlip(p=cfg.AUG.V_FLIP))
    if cfg.AUG.H_FLIP > 0:
        trans.append(transforms.RandomHorizontalFlip(p=cfg.AUG.H_FLIP))
    if cfg.AUG.ROTATION > 0:
        trans.append(transforms.RandomRotation(cfg.AUG.ROTATION, expand=False))
    if cfg.AUG.BRIGHTNESS > 0 or cfg.AUG.CONTRAST > 0 or cfg.AUG.SATURATION > 0 or cfg.AUG.HUE > 0:
        trans.append(transforms.ColorJitter(brightness=cfg.AUG.BRIGHTNESS, 
            contrast=cfg.AUG.CONTRAST,saturation=cfg.AUG.SATURATION,hue=cfg.AUG.HUE))
    if cfg.AUG.MULTI_CROP_SIZE > 0:
        trans.append(custom_transforms.MultiScaleCrop(crop_size=cfg.AUG.MULTI_CROP_SIZE,
            scale_ratios=cfg.AUG.SCALE_RATIOS, max_distort=cfg.AUG.MAX_DISTORT))
    if cfg.AUG.RND_CROP[0] > 0 and cfg.AUG.RND_CROP[1] > 0:
        trans.append(transforms.RandomCrop(cfg.AUG.RND_CROP))

    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize(cfg.MEAN, cfg.STD))
    return trans

def load_src_trainset():
    root = cfg.DATA_LOADER.DATA_ROOT
    transform = transforms.Compose(get_transform())

    if cfg.DATA_LOADER.SOURCE_TYPE == 'online': 
        paths, labels = utils.loadlines(os.path.join(root, 'list', cfg.DATA_LOADER.SOURCE + '_train.txt'))
        image_set = online_loader.OnlineLoader(root, paths, labels, transform)
        return image_set
    else:
        image_set = list_loader.ListLoader(root, \
            os.path.join(root, 'list', cfg.DATA_LOADER.SOURCE + '_train.txt'), transform)
        return image_set

def load_trg_online_trainset(paths, labels):
    root = cfg.DATA_LOADER.DATA_ROOT
    transform = transforms.Compose(get_transform())
    image_set = online_loader.OnlineLoader(root, paths, labels, transform)
    return image_set

def load_trg_list_trainset(paths, labels):
    root = cfg.DATA_LOADER.DATA_ROOT
    transform = transforms.Compose(get_transform())

    image_set = list_loader.ListLoader(root, os.path.join(root, 'list', cfg.DATA_LOADER.TARGET + '_train.txt'),\
        transform, lists=paths, labels=labels)
    return image_set

def load_mergesrc_train(distributed, image_set):
    if cfg.DATA_LOADER.SOURCE_TYPE == 'online':
        gpu_num = dist.get_world_size() if distributed else 1
        assert cfg.TRAIN.BATCH_SIZE % cfg.DATA_LOADER.CLASS_NUM_PERBATCH == 0
        imgs_per_cls = cfg.TRAIN.BATCH_SIZE // cfg.DATA_LOADER.CLASS_NUM_PERBATCH
        cls_info = image_set.sample_cls(cfg.DATA_LOADER.ITER, cfg.DATA_LOADER.CLASS_NUM_PERBATCH * gpu_num, cfg.MODEL.CLASS_NUM)
        index = image_set.samples(imgs_per_cls, cls_info)
        index = image_set.shuffle_index(index, cfg.TRAIN.BATCH_SIZE, gpu_num)
    
        if distributed:
            index = torch.tensor(index, device="cuda")
            torch.distributed.broadcast(index, 0)
            index = index.data.cpu().numpy().tolist()

        sampler = samplers.distributed.DistributedSamplerOnline(image_set, \
            batch_size=cfg.TRAIN.BATCH_SIZE, index=index, distributed=distributed)
    
        loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
                drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY, sampler=sampler)
    else:
        cls_info = None
        loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=cfg.DATA_LOADER.SHUFFLE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
                drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
    return cls_info, loader

def load_trg_train(distributed, paths, labels, cls_info):
    if (cfg.DATA_LOADER.TARGET_TYPE == 'online') and (labels is not None) and (cls_info is not None):
        image_set = load_trg_online_trainset(paths, labels)

        assert cfg.TRAIN.BATCH_SIZE % cfg.DATA_LOADER.CLASS_NUM_PERBATCH == 0
        imgs_per_cls = cfg.TRAIN.BATCH_SIZE // cfg.DATA_LOADER.CLASS_NUM_PERBATCH
        index = image_set.samples(imgs_per_cls, cls_info)
        gpu_num = dist.get_world_size() if distributed else 1
        index = image_set.shuffle_index(index, cfg.TRAIN.BATCH_SIZE, gpu_num)

        if distributed:
            index = torch.tensor(index, device="cuda")
            torch.distributed.broadcast(index, 0)
            index = index.data.cpu().numpy().tolist()

        sampler = samplers.distributed.DistributedSamplerOnline(image_set, \
            batch_size=cfg.TRAIN.BATCH_SIZE, index=index, distributed=distributed)
        loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS, 
            drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY, sampler=sampler)
        return loader
    else:
        image_set = load_trg_list_trainset(paths, labels)
        loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=cfg.DATA_LOADER.SHUFFLE, num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            drop_last=cfg.DATA_LOADER.DROP_LAST, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
        return loader

def load_test(target_root, test_label):
    transform = transforms.Compose([
        transforms.Resize(cfg.AUG.RESIZE),
        transforms.CenterCrop(cfg.AUG.TEST_CROP),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD)
    ])
    image_set = list_loader.ListLoader(target_root, test_label, transform)
    loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
    return loader
