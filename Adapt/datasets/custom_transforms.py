import torch
from torch import nn
from torchvision import transforms

from PIL import Image
import random
import numpy as np
import numbers
import types
import collections
import cv2

class MultiScaleCrop(object):
    def __init__(self, crop_size, scale_ratios=[1, 0.875, 0.75, 0.66], 
        max_distort=1, interpolation=Image.BILINEAR):
        self.size = (crop_size, crop_size)
        self.interpolation = interpolation
        self.scale_ratios = scale_ratios
        self.max_distort = max_distort

    @staticmethod
    def get_params(img, net_input_height, net_input_width, scale_ratios, max_distort):
        base_size = min(img.size[1], img.size[0])
        crop_sizes = []
        for h in range(len(scale_ratios)):
            crop_h = int(base_size * scale_ratios[h])
            crop_h = net_input_height if abs(crop_h - net_input_height) < 3 else crop_h
            for w in range(len(scale_ratios)):
                crop_w = int(base_size * scale_ratios[w])
                crop_w = net_input_width if abs(crop_w - net_input_width) < 3 else crop_w

                if abs(h - w) <= max_distort:
                    crop_sizes.append((crop_h, crop_w))

        sel = random.randint(0, len(crop_sizes) - 1)
        crop_height, crop_width = crop_sizes[sel]
        i = random.randint(0, img.size[1] - crop_height)
        j = random.randint(0, img.size[0] - crop_width)
        return i, j, crop_height, crop_width
        
    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size[1], self.size[0], self.scale_ratios, self.max_distort)
        return transforms.functional.resized_crop(img, i, j, h, w, self.size, self.interpolation)

class AdjustBrightness(object):
    def __init__(self, min_contrast=0.8, max_contrast=1.2, max_brightness_shift=5, p=0.5):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.max_brightness_shift = max_brightness_shift
        self.p = p

    @staticmethod
    def get_params(min_contrast, max_contrast, max_brightness_shift):
        alpha = random.uniform(min_contrast, max_contrast)
        beta = random.randint(0, max_brightness_shift * 2) - max_brightness_shift
        return alpha, beta

    def __call__(self, img):
        if random.random() < self.p:
            alpha, beta = self.get_params(self.min_contrast, self.max_contrast, self.max_brightness_shift)
            return img.point(lambda i: i * alpha + beta)
        return img

class SmoothFilter(object):
    def __init__(self, max_smooth=5, p=0.5):
        self.max_smooth = max_smooth
        self.p = p

    @staticmethod
    def get_params(max_smooth):
        smooth_size = 1 + 2 * random.randint(0, int(max_smooth / 2) - 1)
        smooth_type = random.randint(0, 3)
        return smooth_type, smooth_size

    def __call__(self, img):
        if random.random() < self.p:
            smooth_type, smooth_size = self.get_params(self.max_smooth)
            opencv_img = np.array(img)
            if smooth_type == 0:
                opencv_img = cv2.GaussianBlur(opencv_img, (smooth_size, smooth_size), 0)
            elif smooth_type == 1:
                opencv_img = cv2.blur(opencv_img, (smooth_size, smooth_size))
            elif smooth_type == 2:
                opencv_img = cv2.medianBlur(opencv_img, smooth_size)
            else:
                opencv_img = cv2.boxFilter(opencv_img, -1, (smooth_size * 2, smooth_size * 2))
            return Image.fromarray(opencv_img)
        return img

class ColorShift(object):
    def __init__(self, max_color_shift=20):
        self.max_color_shift = max_color_shift

    @staticmethod
    def get_params(max_color_shift):
        color_augment = []
        for _ in range(3):
            color = random.randint(-1, 1) * random.randint(0, max_color_shift)
            color_augment.append(color / 255.0)
        return color_augment

    def __call__(self, img):
        color_augment = self.get_params(self.max_color_shift)
        img[0, :, :] += color_augment[0]
        img[1, :, :] += color_augment[1]
        img[2, :, :] += color_augment[2]
        return img
