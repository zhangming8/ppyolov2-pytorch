# -*- coding: utf-8 -*-
# @Time    : 2021/6/11 18:49
# @Author  : MingZhang
# @Email   : zm19921120@126.com

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2
import glob
import numpy as np
from PIL import Image


class Gridmask(object):
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7, upper_iter=360000):
        """
        GridMask Data Augmentation, see https://arxiv.org/abs/2001.04086
            Args:
            use_h (bool): whether to mask vertically
            use_w (boo;): whether to mask horizontally
            rotate (float): angle for the mask to rotate
            offset (float): mask offset
            ratio (float): mask ratio
            mode (int): gridmask mode
            prob (float): max probability to carry out gridmask
            upper_iter (int): suggested to be equal to global max_iter
        """
        super(Gridmask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob
        self.st_prob = prob
        self.upper_iter = upper_iter

    def __call__(self, x, curr_iter):
        self.prob = self.st_prob * min(1, 1.0 * curr_iter / self.upper_iter)
        if np.random.rand() > self.prob:
            return x
        h, w, _ = x.shape
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w].astype(np.float32)

        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask, axis=-1)
        if self.offset:
            offset = (2 * (np.random.rand(h, w) - 0.5)).astype(np.float32)
            x = (x * mask + offset * (1 - mask)).astype(x.dtype)
        else:
            x = (x * mask).astype(x.dtype)

        return x


def mix_up(img1, gt_bbox1, img2, gt_bbox2, alpha=1.5, beta=1.5):
    factor = np.random.beta(alpha, beta)
    factor = max(0.0, min(1.0, factor))
    if factor >= 1.0:
        return img1, gt_bbox1
    if factor <= 0.0:
        return img2, gt_bbox2

    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])
    img = np.zeros((h, w, img1.shape[2]), 'float32')

    img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * factor
    img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1.0 - factor)
    img = img.astype('uint8')

    gt_bbox = []
    for ann in gt_bbox1:
        ann['gt_score'] = factor
        gt_bbox.append(ann)
    for ann in gt_bbox2:
        ann['gt_score'] = 1.0 - factor
        gt_bbox.append(ann)
    return img, gt_bbox


def cut_mix(img1, gt_bbox1, img2, gt_bbox2, alpha=1.5, beta=1.5):
    # TODO, fix it, it's not right
    factor = np.random.beta(alpha, beta)
    factor = max(0.0, min(1.0, factor))
    if factor >= 1.0:
        return img1, gt_bbox1
    if factor <= 0.0:
        return img2, gt_bbox2

    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])
    cut_rat = np.sqrt(1. - factor)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    # uniform
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w - 1)
    bby1 = np.clip(cy - cut_h // 2, 0, h - 1)
    bbx2 = np.clip(cx + cut_w // 2, 0, w - 1)
    bby2 = np.clip(cy + cut_h // 2, 0, h - 1)

    img_1_pad = np.zeros((h, w, img1.shape[2]), 'float32')
    img_2_pad = np.zeros((h, w, img2.shape[2]), 'float32')
    img_1_pad[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32')
    img_2_pad[:img2.shape[0], :img2.shape[1], :] = img2.astype('float32')
    img_1_pad[bby1:bby2, bbx1:bbx2, :] = img_2_pad[bby1:bby2, bbx1:bbx2, :]
    img_1_pad = img_1_pad.astype('uint8')

    gt_bbox = []
    for ann in gt_bbox1:
        ann['gt_score'] = factor
        gt_bbox.append(ann)
    for ann in gt_bbox2:
        ann['gt_score'] = 1.0 - factor
        gt_bbox.append(ann)

    return img_1_pad, gt_bbox


if __name__ == "__main__":
    img_list = glob.glob(r"D:\work\hc\dataset\algorithm_dataset\test_dataset\*.jpeg")
    for idx, im_p in enumerate(img_list):
        img_org = cv2.imread(im_p)
        img_g = Gridmask(ratio=0.7, prob=0.7, upper_iter=360)(img_org, idx+100)

        cv2.namedWindow("img", 0)
        cv2.imshow("img", img_org)
        cv2.namedWindow("gm", 0)
        cv2.imshow("gm", img_g)

        key = cv2.waitKey(0)
        if key == 27:
            exit()
