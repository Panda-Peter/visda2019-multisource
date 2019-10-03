import os
import datasets.list_loader as list_loader
import torch
from torchvision import transforms
import lib.utils as utils
from lib.config import cfg
import torch.distributed as dist
from os.path import join as join


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
            contrast=cfg.AUG.CONTRAST, saturation=cfg.AUG.SATURATION, hue=cfg.AUG.HUE))
    if cfg.AUG.RND_CROP[0] > 0 and cfg.AUG.RND_CROP[1] > 0:
        trans.append(transforms.RandomCrop(cfg.AUG.RND_CROP))

    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize(cfg.MEAN, cfg.STD))
    return trans


def load_test(target_root, test_label, use_mirror = False):
    if use_mirror:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize(cfg.AUG.TEST_CROP),
            transforms.ToTensor(),
            transforms.Normalize(cfg.MEAN, cfg.STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(cfg.AUG.TEST_CROP),
            transforms.ToTensor(),
            transforms.Normalize(cfg.MEAN, cfg.STD)
        ])
    image_set = list_loader.ListLoader(target_root, test_label, transform)
    loader = torch.utils.data.DataLoader(image_set, batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False, num_workers=cfg.DATA_LOADER.NUM_WORKERS)
    return loader
