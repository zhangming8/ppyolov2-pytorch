# -*- coding: utf-8 -*-
# @Time    : 2021/6/3 17:48
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import copy
import uuid
from numbers import Number, Integral
from collections.abc import Sequence
import numpy as np
import cv2
import math

from dataset.base import BaseDataset
from utils.image import draw_umich_gaussian, draw_msra_gaussian, truncate_radius, draw_truncate_gaussian
from utils.image import get_affine_transform, affine_transform, color_aug
from utils.image import gaussian_radius_v2 as gaussian_radius
from utils.image import draw_dense_reg


def viz_hm(hm):
    tmp_hms = []
    for i, tmp_h in enumerate(hm):
        hm_h, hm_w = tmp_h.shape
        one_hm = np.zeros([hm_h + hm_h // 2, hm_w + 20]) + 1.0
        one_hm[:hm_h, :hm_w] = tmp_h
        cv2.putText(one_hm, "{}".format(i), (hm_w // 2, hm_h + hm_h // 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
        tmp_hms.append(one_hm)
    tmp_hms = np.hstack(tmp_hms)
    tmp_hms = (tmp_hms * 255).astype(np.uint8)
    return tmp_hms


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample

    def __str__(self):
        return str(self._id)


def is_poly(segm):
    assert isinstance(segm, (list, dict)), "Invalid segm type: {}".format(type(segm))
    return isinstance(segm, list)


class Pad(BaseOperator):
    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, Sequence): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """
        super(Pad, self).__init__()

        if not isinstance(size, (int, Sequence)):
            raise TypeError("Type of target_size is invalid when random_size is True. Must be List, now is {}"
                            "".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [-1, 0, 1, 2], 'currently only supports four modes [-1, 0, 1, 2]'
        assert pad_mode == -1 and offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_segm(self, segms, offsets, im_size, size):
        def _expand_poly(poly, x, y):
            expanded_poly = np.array(poly)
            expanded_poly[0::2] += x
            expanded_poly[1::2] += y
            return expanded_poly.tolist()

        def _expand_rle(rle, x, y, height, width, h, w):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            expanded_mask = np.full((h, w), 0).astype(mask.dtype)
            expanded_mask[y:y + height, x:x + width] = mask
            rle = mask_util.encode(
                np.array(
                    expanded_mask, order='F', dtype=np.uint8))
            return rle

        x, y = offsets
        height, width = im_size
        h, w = size
        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [_expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                expanded_segms.append(
                    _expand_rle(segm, x, y, height, width, h, w))
        return expanded_segms

    def apply_bbox(self, bbox, offsets):
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_keypoint(self, keypoints, offsets):
        n = len(keypoints[0]) // 2
        return keypoints + np.array(offsets * n, dtype=np.float32)

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def apply(self, sample, context=None):
        im = sample['image']
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert (im_h < h and im_w < w), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = np.ceil(im_h // self.size_divisor) * self.size_divisor
            w = np.ceil(im_w / self.size_divisor) * self.size_divisor

        if h == im_h and w == im_w:
            return sample

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        sample['image'] = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            return sample
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)

        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], offsets, im_size, size)

        if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
            sample['gt_keypoint'] = self.apply_keypoint(sample['gt_keypoint'], offsets)

        return sample


class RandomDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings. in [lower, upper, probability] format.
        saturation (list): saturation settings. in [lower, upper, probability] format.
        contrast (list): contrast settings. in [lower, upper, probability] format.
        brightness (list): brightness settings. in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD) order.
        count (int): the number of doing distrot
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False,
                 no_color_aug=False):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel
        self.no_color_aug = no_color_aug

    def apply_hue(self, img):
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img += delta
        return img

    def apply(self, sample, context=None):
        if self.no_color_aug:
            return sample
        img = sample['image']
        if self.random_apply:
            functions = [self.apply_brightness, self.apply_contrast, self.apply_saturation, self.apply_hue]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = self.apply_contrast(img)

        img = self.apply_saturation(img)
        img = self.apply_hue(img)

        if not mode:
            img = self.apply_contrast(img)

        if self.random_channel:
            if np.random.randint(0, 2):
                img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample


class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
    """

    def __init__(self, ratio=4., prob=0.5, fill_value=(127.5, 127.5, 127.5)):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def apply(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        im = sample['image']
        height, width = im.shape[:2]
        ratio = np.random.uniform(1., self.ratio)
        h = int(height * ratio)
        w = int(width * ratio)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        offsets, size = [x, y], [h, w]

        pad = Pad(size, pad_mode=-1, offsets=offsets, fill_value=self.fill_value)
        pad_res = pad(sample, context=context)

        return pad_res


class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    def apply(self, sample, context=None):
        h, w = sample['image'].shape[:2]
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(max(min_ar, scale ** 2), min(max_ar, scale ** -2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]

                iou = self._iou_matrix(gt_bbox, np.array([crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(gt_bbox,
                                                                               np.array(crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                sample['image'] = self._crop_image(sample['image'], crop_box)
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(sample['gt_class'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]


class RandomHorizonFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomHorizonFlip, self).__init__()
        self.prob = float(prob)
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image(self, image):
        return image[:, ::-1, :]

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - 1 - oldx2
        bbox[:, 2] = width - 1 - oldx1
        return bbox

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            sample['flipped'] = True
            sample['image'] = im
        return sample


class RandomVerticalFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of RandomVerticalFlip flipping image
        """
        super(RandomVerticalFlip, self).__init__()
        self.prob = float(prob)
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image(self, image):
        return image[::-1, :, :]

    def apply_bbox(self, bbox, height):
        oldy1 = bbox[:, 1].copy()
        oldy2 = bbox[:, 3].copy()
        bbox[:, 1] = height - 1 - oldy2
        bbox[:, 3] = height - 1 - oldy1
        return bbox

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], height)
            sample['flipped'] = True
            sample['image'] = im
        return sample


class RandomRote90(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of RandomVerticalFlip flipping image
        """
        super(RandomRote90, self).__init__()
        self.prob = float(prob)
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image(self, image):
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)
        return image

    def apply_bbox(self, bbox, width):
        bbox_old = bbox.copy()
        bbox[:, [0, 2]] = bbox_old[:, [1, 3]]
        bbox[:, [1, 3]] = width - 1 - bbox_old[:, [2, 0]]
        return bbox

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            im = sample['image']
            height, width = im.shape[:2]
            im = self.apply_image(im)
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
            sample['flipped'] = True
            sample['image'] = im
        return sample


class Cutmix(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, see https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(Cutmix, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def apply_image(self, img1, img2, factor):
        """ _rand_bbox """
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
        img_1_pad[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32')
        img_2_pad = np.zeros((h, w, img2.shape[2]), 'float32')
        img_2_pad[:img2.shape[0], :img2.shape[1], :] = img2.astype('float32')
        img_1_pad[bby1:bby2, bbx1:bbx2, :] = img_2_pad[bby1:bby2, bbx1:bbx2, :]
        return img_1_pad

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'cutmix need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        img1 = sample[0]['image']
        img2 = sample[1]['image']
        img = self.apply_image(img1, img2, factor)
        gt_bbox1 = sample[0]['gt_bbox']
        gt_bbox2 = sample[1]['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample[0]['gt_class']
        gt_class2 = sample[1]['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = np.ones_like(sample[0]['gt_class'])
        gt_score2 = np.ones_like(sample[1]['gt_class'])
        gt_score = np.concatenate((gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        result = copy.deepcopy(sample[0])
        result['image'] = img
        result['gt_bbox'] = gt_bbox
        result['gt_score'] = gt_score
        result['gt_class'] = gt_class
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate((is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult
        return result


class Resize(BaseOperator):
    def __init__(self, target_size, interp=cv2.INTER_LINEAR, random_interp=False, down_ratio=4):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.interp = interp
        self.random_interp = random_interp
        self.down_ratio = down_ratio

        self.out_h, self.out_w = self.target_size
        self.feat_h, self.feat_w = self.out_h // self.down_ratio, self.out_w // self.down_ratio

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        if self.random_interp:
            interp = np.random.choice(
                [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4])
        else:
            interp = self.interp

        img = sample['image']
        gt_bbox = sample['gt_bbox']

        rot_angle = 0
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.

        trans_input = get_affine_transform(c, s, rot_angle, [self.out_w, self.out_h])
        trans_output = get_affine_transform(c, s, rot_angle, [self.feat_w, self.feat_h])

        img = cv2.warpAffine(img, trans_input, (self.out_w, self.out_h), flags=interp)

        bbox_affine = []
        for gt_b in gt_bbox:
            # resize to feature map size
            gt_b[:2] = affine_transform(gt_b[:2], trans_output)
            gt_b[2:] = affine_transform(gt_b[2:], trans_output)
            gt_b[[0, 2]] = np.clip(gt_b[[0, 2]], 0, self.feat_w - 1)
            gt_b[[1, 3]] = np.clip(gt_b[[1, 3]], 0, self.feat_h - 1)

            bbox_affine.append(gt_b)

        sample['gt_bbox'] = bbox_affine
        sample['image'] = img
        return sample


def convert_ann(coco, img_id, catid2clsid):
    img_anno = coco.loadImgs([img_id])[0]
    ins_anno_ids = coco.getAnnIds(imgIds=[img_id])
    instances = coco.loadAnns(ins_anno_ids)

    im_fname = img_anno['file_name']
    im_w = float(img_anno['width'])
    im_h = float(img_anno['height'])

    bboxes = []
    for inst in instances:
        x1, y1, box_w, box_h = inst['bbox']
        x2 = x1 + box_w
        y2 = y1 + box_h
        inst['clean_bbox'] = [round(float(x), 3) for x in [x1, y1, x2, y2]]
        bboxes.append(inst)

    num_bbox = len(bboxes)
    gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
    gt_class = np.zeros((num_bbox, 1), dtype=np.int32)

    for i, box in enumerate(bboxes):
        gt_class[i][0] = box['category_id']
        gt_bbox[i, :] = box['clean_bbox']

    coco_rec = {}
    coco_rec['im_id'] = np.array([img_id])
    coco_rec['w'] = im_w
    coco_rec['h'] = im_h
    coco_rec['gt_class'] = gt_class
    coco_rec['gt_bbox'] = gt_bbox
    return coco_rec, im_fname


def vis_res(image, res):
    for idx, bbox in enumerate(res['gt_bbox']):
        x1, y1, x2, y2 = bbox
        txt = str(res['gt_class'][idx])
        if 'gt_score' in res:
            txt = "{}:{}".format(txt, res['gt_score'][idx])
        cv2.putText(image, txt, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.rectangle(image, (x1, y1), (int(x2 - 2), int(y2 - 2)), (255, 0, 0), 2)
    return image


class LoadCOCO(BaseDataset):
    def __init__(self, opt, split, logger=None):
        super(LoadCOCO, self).__init__(opt, split, logger)

    def __getitem__(self, index):
        img_id = self.images[index]
        coco_rec, file_name = convert_ann(self.coco, img_id, self.cat_ids)
        img_path = self.img_dir + "/" + file_name
        img = cv2.imread(img_path)

        if self.show:
            image_org = img.copy()
            image_org = vis_res(image_org, coco_rec)
            cv2.imshow("org_image", image_org)

        coco_rec["image"] = img[:, :, ::-1]  # bgr to rgb
        height, width = img.shape[0], img.shape[1]

        if self.opt.keep_res:  # keep_res = False
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
        elif self.opt.multi_input_size and self.split == 'train':
            input_h, input_w = self.samples_shapes[index]
        else:
            input_h, input_w = self.opt.input_h, self.opt.input_w

        feat_h = input_h // self.opt.down_ratio
        feat_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes

        if self.split == 'train':
            augments = [RandomExpand(fill_value=[0, 0, 0]),
                        RandomCrop(aspect_ratio=None, cover_all_box=True, thresholds=[.3, .4, .5, .6, .7, .9]),
                        RandomHorizonFlip(prob=self.opt.horizon_flip),
                        RandomVerticalFlip(prob=self.opt.vertical_flip),
                        RandomRote90(prob=self.opt.rotate90),
                        Resize(target_size=[input_h, input_w], down_ratio=self.opt.down_ratio)]
            # aug = Cutmix(alpha=1.5, beta=1.5)
        else:
            augments = [Resize(target_size=[input_h, input_w], down_ratio=self.opt.down_ratio)]

        for aug in augments:
            coco_rec = aug(coco_rec)

        inp = coco_rec["image"][:, :, ::-1]  # rgb to bgr
        gt_bbox = coco_rec['gt_bbox']
        gt_class = coco_rec['gt_class']
        num_objs = len(gt_bbox)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        if self.show:
            input_img = np.clip(inp.copy() * 255., 0, 255).astype(np.uint8)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        hm = np.zeros((num_classes, feat_h, feat_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, feat_h, feat_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        ids = np.zeros(self.max_objs, dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            bbox = gt_bbox[k]
            cls_id = self.cat_ids[int(gt_class[k])]
            tracking_id = cls_id
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if self.show:
                box_x1, box_x2 = int(bbox[0] * self.opt.down_ratio), int(bbox[2] * self.opt.down_ratio)
                box_y1, box_y2 = int(bbox[1] * self.opt.down_ratio), int(bbox[3] * self.opt.down_ratio)
                if box_x1 > box_x2:
                    print("---> error: box_x1 > box_x2, {} > {}".format(box_x1, box_x2))
                if box_y1 > box_y2:
                    print("---> error: box_y1 > box_y2, {} > {}".format(box_y1, box_y2))
                if h > 0 and w > 0:
                    cv2.putText(input_img, self.class_name[int(gt_class[k])], (box_x1, box_y1),
                                cv2.FONT_HERSHEY_COMPLEX, 1, self.voc_color[cls_id], 1)
                    cv2.rectangle(input_img, (box_x1, box_y1), (box_x2, box_y2), self.voc_color[cls_id], 2)
            if h < 0:
                print("error: bbox {}, y1({}) >= y2({})".format(bbox.tolist(), bbox[1], bbox[3]))
            if w < 0:
                print("error: bbox {}, x1({}) >= x2({})".format(bbox.tolist(), bbox[0], bbox[2]))
            if h > 0 and w > 0:
                if self.opt.fix_radius == -1:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                    if self.opt.truncate_radius:
                        w_radius, h_radius = truncate_radius((w, h))
                else:
                    radius = self.opt.fix_radius  # 3  # 5
                    # print("fix radius:", radius)
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                if self.opt.fix_radius == -1 and self.opt.truncate_radius:
                    draw_truncate_gaussian(hm[cls_id, :, :], ct_int, h_radius, w_radius)
                else:
                    draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * feat_w + ct_int[0]  # ind[k]: 0~128*128-1, object index in 128*128
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = tracking_id
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        if self.show:
            v_hm = viz_hm(hm)
            cv2.namedWindow("input", 0)
            cv2.imshow("input", input_img)
            cv2.namedWindow("heatmap", 0)
            cv2.imshow("heatmap", v_hm)
            key = cv2.waitKey(0)
            cv2.imwrite("./tmp/{}_g.jpg".format(index), input_img)
            cv2.imwrite("./tmp/{}_hm.jpg".format(index), v_hm)
            if key == 27:
                exit()

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ids': ids}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if not self.split == 'train':
            meta = {'img_id': img_id, 'org_img_h': height, 'org_img_w': width}
            ret['meta'] = meta
        return ret


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from opts import opts

    opt = opts().parse()
    opt = opts().update_dataset_info_and_set_heads(opt, LoadCOCO)
    opt.fix_radius = -1  # 3 # 5
    opt.truncate_radius = True
    # opt.keep_res = True
    # opt.mosaic = True
    # opt.multi_input_size = True
    opt.horizon_flip = 0.5
    opt.vertical_flip = 0.5
    opt.rotate90 = 0
    opt.input_h, opt.input_w = 384, 640  # 544, 960
    opt.batch_size = 1
    opt.num_workers = 0
    opt.data_dir = r"D:\work\hc\dataset\algorithm_dataset"
    opt.annotation = r"D:\work\hc\dataset\algorithm_dataset"

    print(opt)
    # dataset = LoadCOCO(opt, "val")
    dataset = LoadCOCO(opt, "train")
    dataset.show = True
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers
                            )  # if num_workers>=1, use multi-process to load image
    for _ in range(10):
        for idx, target in enumerate(dataloader):
            for k, v in target.items():
                if type(v) == dict:
                    print(k, v)
                else:
                    print(k, v.size())
            print("------------- batch: {} ---------------".format(idx))
