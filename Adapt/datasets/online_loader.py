import torch
import torch.utils.data as data
from PIL import Image
import os
import random
import numpy as np
from lib.config import cfg

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(root, list_path, labels):
    images = {}
    for i in range(cfg.MODEL.CLASS_NUM):
        images[i] = []

    for i in range(len(list_path)):
        images[labels[i]].append((i, os.path.join(root, list_path[i])))
    return images

class OnlineLoader(data.Dataset):
    def __init__(self, root, list_path, labels, transform=None, loader=default_loader):
        imgs = make_dataset(root, list_path, labels)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        label, imgid = index
        idx, path = self.imgs[label][imgid]

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return idx, img, label
        

    def sample_cls(self, iterations, class_num_per_batch, class_num):
        cls_info = []
        for _ in range(iterations):
            cls_idxs = random.sample(range(class_num), class_num_per_batch)
            cls_info += cls_idxs
        return cls_info

    def samples(self, imgs_per_cls, cls_info):
        index = []
        for c in cls_info:
            if len(self.imgs[c]) >= imgs_per_cls:
                img_idxs = random.sample(range(len(self.imgs[c])), imgs_per_cls)
            else:
                img_idxs = []
                loop_num = imgs_per_cls // len(self.imgs[c])
                for _ in range(loop_num):
                    img_idxs += list(range(len(self.imgs[c])))
                left_num = imgs_per_cls - loop_num * len(self.imgs[c])
                img_idxs += random.sample(range(len(self.imgs[c])), left_num)
            for img_idx in img_idxs:
                index.append((c, img_idx))
        return index

    def shuffle_index(self, index, batch_size, gpu_num):
        if gpu_num == 1:
            return index
        loop_num = len(index) // (batch_size * gpu_num)
        for i in range(loop_num):
            start = i * batch_size * gpu_num
            end = start + batch_size * gpu_num
            rlist = index[start:end]
            random.shuffle(rlist)
            index[start:end] = rlist
        return index
