# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 11:26
# @Author  : MingZhang
# @Email   : zm19921120@126.com

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import sys
import numpy as np

sys.path.append(".")
from dataset.base import BaseDataset
from dataset.augment import Gridmask, mix_up, cut_mix
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or sample_bbox[2] <= object_bbox[0] or sample_bbox[1] >= object_bbox[3] or \
            sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


class LoadCOCO(BaseDataset):
    def __init__(self, opt, split, logger=None):
        super(LoadCOCO, self).__init__(opt, split, logger)

        self.anchors = opt.anchors
        self.anchor_masks = opt.anchor_masks
        self.downsample_ratios = opt.downsample
        self.iou_thresh = 1.

    def __getitem__(self, index):
        mosaic = self.split == 'train' and np.random.random() < self.opt.mosaic
        if self.split == 'train' and mosaic:
            img, anns, img_path = self.mosaic(index)
            num_objs = min(len(anns), self.max_objs)
            img_id = ''
        else:
            img, anns, img_path, img_id = self.read_img_by_idx(index)
            num_objs = min(len(anns), self.max_objs)
            img, num_objs = self.dummy_img(img, num_objs, img_path)

        if self.show:
            cv2.namedWindow("org", 0)
            cv2.imshow("org", img)
            print("org shape: {}".format(img.shape))

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:  # keep_res = False
            input_h = height if height % (self.opt.pad + 1) == 0 else (height | self.opt.pad) + 1
            input_w = width if width % (self.opt.pad + 1) == 0 else (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        elif self.opt.multi_input_size and self.split == 'train':
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.samples_shapes[index]
        else:
            s = max(img.shape[0], img.shape[1]) * 1.
            input_h, input_w = self.opt.input_h, self.opt.input_w

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        horizon_flip, vertical_flip, rotate90, rot_angle, interp_method = False, False, False, 0, interp_methods[0]
        if self.split == 'train':
            if not mosaic and np.random.random() < self.opt.mix_up:
                mix_up_idx = np.random.randint(0, self.num_samples - 1)
                img2, anns2, _, _ = self.read_img_by_idx(mix_up_idx)
                img, anns = mix_up(img, anns, img2, anns2, alpha=1.5, beta=1.5)
                num_objs = min(len(anns), self.max_objs)

            if self.opt.grid_mask:
                img = Gridmask(ratio=0.7, prob=0.7, upper_iter=self.opt.num_epochs)(img, self.epoch)

            # random crop
            # s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            s = s * np.random.choice(np.arange(0.3, 1.1, 0.1) if mosaic else np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, img.shape[1])
            h_border = self._get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

            if np.random.random() < self.opt.horizon_flip:
                horizon_flip = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1
            if np.random.random() < self.opt.vertical_flip:
                vertical_flip = True
                img = img[::-1, :, :]
                c[1] = height - c[1] - 1
            if np.random.random() < self.opt.rotate90:
                rotate90 = True
                img = cv2.transpose(img)
                img = cv2.flip(img, 1)
                c[[0, 1]] = c[[1, 0]]
                c[0] = height - c[0] - 1
            rot_angle = np.random.choice(list(range(-self.opt.rot_angle, self.opt.rot_angle + 1)))
            interp_method = np.random.choice(interp_methods) if self.opt.random_interp else interp_method

        trans_input = get_affine_transform(c, s, rot_angle, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=interp_method, borderValue=0)

        if self.opt.multi_input_size and self.split == 'train':
            pad_h, pad_w = self.max_batch_shape[index]
            pad_img = (np.zeros([pad_h, pad_w, 3])).astype(inp.dtype)
            pad_img[:input_h, :input_w, :] = inp
            inp = pad_img
            input_h, input_w = pad_h, pad_w

        inp = inp.astype(np.float32) / 255.
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        if self.show:
            input_img = np.clip(inp.copy() * 255., 0, 255).astype(np.uint8)
            print('warpAffine:', input_img.shape)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        gt_bbox = np.zeros((self.max_objs, 4), dtype=np.float32)
        gt_class = np.zeros(self.max_objs, dtype=np.int64)
        gt_score = np.zeros(self.max_objs, dtype=np.float32)

        b_i = 0
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])

            if horizon_flip:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            if vertical_flip:
                bbox[[1, 3]] = height - bbox[[3, 1]] - 1
            if rotate90:
                if np.ndim(bbox) == 1:
                    bbox[[0, 1, 2, 3]] = bbox[[1, 0, 3, 2]]
                    bbox[[0, 2]] = height - bbox[[0, 2]] - 1
                    bbox[[0, 2]] = bbox[[2, 0]]
                else:
                    bbox[:, [0, 1, 2, 3]] = bbox[:, [1, 0, 3, 2]]
                    bbox[:, [0, 2]] = height - bbox[:, [0, 2]] - 1
                    bbox[:, [0, 2]] = bbox[:, [2, 0]]

            bbox[:2] = affine_transform(bbox[:2], trans_input)
            bbox[2:] = affine_transform(bbox[2:], trans_input)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, input_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, input_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h < 0:
                print("error: bbox {}, y1({}) > y2({}), img_id {}, img_path {}".format(bbox.tolist(), bbox[1], bbox[3],
                                                                                       img_id, img_path))
            if w < 0:
                print("error: bbox {}, x1({}) > x2({}), img_id {}, img_path {}".format(bbox.tolist(), bbox[0], bbox[2],
                                                                                       img_id, img_path))
            if h > 0 and w > 0:
                gt_bbox[b_i] = bbox[0], bbox[1], bbox[2], bbox[3]
                gt_class[b_i] = cls_id
                gt_score[b_i] = ann.get('gt_score', 1.)

                if self.show:
                    print("box: {}, cls: {}, score: {}".format(gt_bbox[b_i], gt_class[b_i], gt_score[b_i]))
                    box_x1, box_x2 = int(bbox[0]), int(bbox[2])
                    box_y1, box_y2 = int(bbox[1]), int(bbox[3])
                    assert box_x1 < input_w and box_x2 < input_w
                    assert box_y1 < input_h and box_y2 < input_h
                    if box_x1 > box_x2:
                        print("---> error: box_x1 > box_x2, {} > {}".format(box_x1, box_x2))
                    if box_y1 > box_y2:
                        print("---> error: box_y1 > box_y2, {} > {}".format(box_y1, box_y2))
                    cv2.putText(input_img, self.class_name[cls_id], (box_x1, box_y1),
                                cv2.FONT_HERSHEY_COMPLEX, 1, self.voc_color[cls_id], 1)
                    cv2.rectangle(input_img, (box_x1, box_y1), (box_x2, box_y2), self.voc_color[cls_id], 2)

                b_i += 1
                if b_i >= self.max_objs:
                    print("==>> too many objects({}>={}), ignore img_id {} {} {}".format(b_i, self.max_objs, img_id,
                                                                                         self.class_name[cls_id], bbox))
                    continue

        if self.show:
            cv2.namedWindow("input", 0)
            cv2.imshow("input", input_img)
            key = cv2.waitKey(0)
            if key == 27:
                exit()
            # cv2.imwrite("./tmp/{}.jpg".format(index), input_img)

        # normalize bbox
        gt_bbox[:, [0, 2]] = gt_bbox[:, [0, 2]] / input_w
        gt_bbox[:, [1, 3]] = gt_bbox[:, [1, 3]] / input_h
        # x1,y1,x2,y2 to cx,cy,w,h
        gt_bbox[:, 2:4] = gt_bbox[:, 2:4] - gt_bbox[:, :2]  # w, h
        gt_bbox[:, :2] = gt_bbox[:, :2] + gt_bbox[:, 2:4] / 2.  # center x, center y

        sample = dict()
        sample["image"] = inp
        sample['gt_bbox'] = gt_bbox
        sample['gt_class'] = gt_class
        sample['gt_score'] = gt_score
        sample['im_shape'] = np.asarray([height, width], dtype=np.float32)
        an_wh = np.array(self.anchors) / np.array([[input_w, input_h]])
        for i, (mask, downsample_ratio) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
            grid_h = int(input_h / downsample_ratio)
            grid_w = int(input_w / downsample_ratio)
            target = np.zeros((len(mask), 6 + self.num_classes, grid_h, grid_w), dtype=np.float32)  # 3x86x13x13
            for b in range(gt_bbox.shape[0]):
                gt_x, gt_y, gt_w, gt_h = gt_bbox[b, :]
                cls = gt_class[b]
                score = gt_score[b]
                if gt_w <= 0. or gt_h <= 0. or score <= 0.:
                    continue

                # find best match anchor index
                best_iou = 0.
                best_idx = -1
                # for an_idx in mask:
                for an_idx in range(an_wh.shape[0]):
                    iou = jaccard_overlap([0., 0., gt_w, gt_h], [0., 0., an_wh[an_idx, 0], an_wh[an_idx, 1]])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = an_idx

                gi = int(gt_x * grid_w)
                gj = int(gt_y * grid_h)

                # gtbox should be regresed in this layes if best match
                # anchor index in anchor mask of this layer
                if best_idx in mask:
                    best_n = mask.index(best_idx)

                    # x, y, w, h, scale
                    target[best_n, 0, gj, gi] = gt_x * grid_w - gi
                    target[best_n, 1, gj, gi] = gt_y * grid_h - gj
                    target[best_n, 2, gj, gi] = np.log(gt_w * input_w / self.anchors[best_idx][0])
                    target[best_n, 3, gj, gi] = np.log(gt_h * input_h / self.anchors[best_idx][1])
                    target[best_n, 4, gj, gi] = 2.0 - gt_w * gt_h

                    # objectness record gt_score
                    target[best_n, 5, gj, gi] = score

                    # classification
                    target[best_n, 6 + cls, gj, gi] = 1.

                # For non-matched anchors,
                # calculate the target if the iou between anchor and gt is larger than iou_thresh
                if self.iou_thresh < 1:
                    for idx, mask_i in enumerate(mask):
                        if mask_i == best_idx:
                            continue
                        iou = jaccard_overlap([0., 0., gt_w, gt_h], [0., 0., an_wh[mask_i, 0], an_wh[mask_i, 1]])
                        if iou > self.iou_thresh and target[idx, 5, gj, gi] == 0.:
                            # x, y, w, h, scale
                            target[idx, 0, gj, gi] = gt_x * grid_w - gi
                            target[idx, 1, gj, gi] = gt_y * grid_h - gj
                            target[idx, 2, gj, gi] = np.log(gt_w * input_w / self.anchors[mask_i][0])
                            target[idx, 3, gj, gi] = np.log(gt_h * input_h / self.anchors[mask_i][1])
                            target[idx, 4, gj, gi] = 2.0 - gt_w * gt_h

                            # objectness record gt_score
                            target[idx, 5, gj, gi] = score

                            # classification
                            target[idx, 6 + cls, gj, gi] = 1.
            sample['target{}'.format(i)] = target

        if "ap" in self.opt.metric.lower():
            sample["img_id"] = img_id
        return sample


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from config import opt

    # opt.keep_res = True
    opt.mosaic = 0.3
    opt.mix_up = 0.5
    opt.multi_input_size = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    opt.horizon_flip = 0
    # opt.vertical_flip = 0.5
    opt.rotate90 = 0
    opt.rot_angle = 0
    # opt.grid_mask = True
    # opt.input_h, opt.input_w = 800, 1344
    opt.input_h, opt.input_w = 640, 640
    opt.batch_size = 2
    opt.num_workers = 0
    # opt.data_dir = r"D:\work\public_dataset\coco2017"
    opt.data_dir = "/media/ming/DATA1/dataset/coco2017"
    opt.annotation = ""
    print(opt)

    # dataset = LoadCOCO(opt, "val")
    dataset = LoadCOCO(opt, "train")
    dataset.show = True
    dataset.epoch = 70
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    for _ in range(10):
        for batch_i, t in enumerate(dataloader):
            for ke, v in t.items():
                print(ke, v.size())
            print("------------- batch: {} ---------------".format(batch_i))
