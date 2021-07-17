# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 14:42
# @Author  : MingZhang
# @Email   : zm19921120@126.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import cv2
import os
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

from utils.utils import NpEncoder, label_color


class BaseDataset(data.Dataset):

    def __init__(self, opt, split, logger=None):
        super(BaseDataset, self).__init__()
        classes = opt.label_name
        self.num_classes = opt.num_classes
        self.class_name = classes
        self.mean = opt.mean
        self.std = opt.std

        if set(classes) == {
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'}:
            print("using origin COCO dataset...", classes)
            self._valid_ids = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                82, 84, 85, 86, 87, 88, 89, 90]  # if train origin coco dataset, use it's valid_ids
        else:
            self._valid_ids = [i + 1 for i in range(len(classes))]
            print("using my own dataset...", classes)
        print("number classes {}".format(len(classes)))

        self.logger = logger
        self.show = False
        self.data_dir = opt.data_dir.replace("\\", "/")
        if self.logger:
            self.logger.write("{} data_dir: {}\n".format(split, self.data_dir))
        if opt.annotation == "":
            # COCO style
            self.img_dir = self.data_dir + "/images/{}2017".format(split)
            self.annot_path = self.data_dir + '/annotations/instances_{}2017.json'.format(split)
        else:
            self.img_dir = self.data_dir
            self.annot_path = opt.annotation.replace("\\", "/") + '/instances_{}2014.json'.format(split)

        self.max_objs = 100
        self.epoch = 1

        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = label_color

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)

        self.split = split
        self.opt = opt

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        if split == "train":
            self.shuffle()

        if self.logger:
            self.logger.write('==> initializing {}\n'.format(self.annot_path))
            self.logger.write('Loaded {} {} samples...\n'.format(split, self.num_samples))
        print('==> initializing {}'.format(self.annot_path))
        print('Loaded {} {} samples...'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def shuffle(self):
        np.random.shuffle(self.images)
        print("shuffle {} images list...".format(self.split))
        if self.logger:
            self.logger.write("shuffle {} images list...\n".format(self.split))
        if self.opt.multi_input_size and self.split == "train":
            self.multi_shape()

    def multi_shape(self):
        multi_shapes = []
        for s in self.opt.multi_input_size:
            if isinstance(s, list) or isinstance(s, tuple):
                h, w = s
            else:
                h, w = s, s
            assert h % 32 == 0 and w % 32 == 0, "height and width should be divisible by 32: {}".format(s)
            multi_shapes.append([h, w])

        self.samples_shapes, self.max_batch_shape = [], []
        for b in range(int(np.ceil(self.num_samples / self.opt.batch_size))):
            if b < 5 and self.epoch <= 1:
                batch_shape = multi_shapes[-1]  # init with largest size in case of out of memory during training
            else:
                rand_idx = np.random.choice(list(range(len(multi_shapes))))
                batch_shape = multi_shapes[rand_idx]

            for _ in range(self.opt.batch_size):
                self.samples_shapes.append(batch_shape)
                self.max_batch_shape.append(batch_shape)
        print("multi size training: {}".format(multi_shapes))
        if self.logger:
            self.logger.write("multi size training: {}\n".format(multi_shapes))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes.keys():
            one_img_res = all_bboxes[image_id]
            for res in one_img_res:
                cls, conf, bbox = res[0], res[1], res[2]
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    'category_id': self._valid_ids[self.class_name.index(cls)],
                    'image_id': int(image_id),
                    'score': conf})
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results), open('{}/results.json'.format(save_dir), 'w'), cls=NpEncoder,
                  indent=2)

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_all, ap_0_5 = coco_eval.stats[0], coco_eval.stats[1]
        return ap_all, ap_0_5

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        return bbox

    def bbox_to_coco_box(self, box):
        bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def read_img_by_idx(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = self.img_dir + "/" + file_name
        img = cv2.imread(img_path)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        return img, anns, img_path, img_id

    def read_and_resize_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img, anns, img_path, img_id = self.read_img_by_idx(index)

        img_h, img_w = img.shape[0], img.shape[1]
        dst_shape = [self.opt.input_h, self.opt.input_w]
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]

        ratio_dst = dst_shape[0] / float(dst_shape[1])
        ratio_org = img_h / float(img_w)
        if ratio_dst > ratio_org:
            scale = dst_shape[1] / float(img_w)
        else:
            scale = dst_shape[0] / float(img_h)

        new_shape = (int(round(img_w * scale)), int(round(img_h * scale)))
        interp_method = interp_methods[np.random.randint(5)]
        img = cv2.resize(img, new_shape, interpolation=interp_method)

        return img, (img_h, img_w), img.shape[:2], anns, scale, img_path

    def mosaic(self, index):
        img4 = np.zeros((self.opt.input_h * 2, self.opt.input_w * 2, 3), dtype=np.uint8)  # base image with 4 tiles
        # xc = int(np.random.uniform(self.opt.input_w * 0.5, self.opt.input_w * 1.5))  # mosaic center x, y
        # yc = int(np.random.uniform(self.opt.input_h * 0.5, self.opt.input_h * 1.5))
        xc, yc = self.opt.input_w, self.opt.input_h
        indices = [index] + [np.random.randint(0, self.num_samples - 1) for _ in range(3)]  # 3 additional image indices
        anns4 = []
        img_paths = []
        for i, idx in enumerate(indices):
            img, _, (h, w), anns, scale, img_path = self.read_and_resize_image(idx)
            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.opt.input_w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.opt.input_h * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.opt.input_w * 2), min(self.opt.input_h * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            else:
                raise ValueError("only support combine 4 images")

            img_paths.append(img_path)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            for ann in anns:
                bbox = self._coco_box_to_bbox(ann['bbox'])
                bbox[0] = scale * bbox[0] + padw
                bbox[1] = scale * bbox[1] + padh
                bbox[2] = scale * bbox[2] + padw
                bbox[3] = scale * bbox[3] + padh
                anns4.append(dict(bbox=self.bbox_to_coco_box(bbox), category_id=ann['category_id']))

        return img4, anns4, img_paths

    def dummy_img(self, img, num_objs, img_path):
        if img is None:
            if os.path.isfile(img_path):
                print("img is damaged: {}".format(img_path))
            else:
                print("cannot find img: {}".format(img_path))
            print("input a dummy image")
            img = (np.zeros([self.opt.input_h, self.opt.input_w, 3])).astype(np.uint8)
            num_objs = 0
        return img, num_objs

    def __getitem__(self, index):
        raise NotImplementedError
