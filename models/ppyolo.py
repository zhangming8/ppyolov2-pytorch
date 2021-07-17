# -*- coding: utf-8 -*-
# @Time    : 2021/6/25 11:00
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.backbone import Darknet, ResNet, ResNeXt, ResNeSt, Res2Net, CSPDarknet53, ResNetVd
from models.neck.yolo_fpn import PPYOLOPAN, PPYOLOFPN
from models.head.yolo_head import YOLOv3Head
from models.losses import YOLOv3Loss, IouLoss, IouAwareLoss
from models.post_process import bbox_post_process
from utils.model_utils import load_model, de_sigmoid
from utils.image import get_affine_transform
from utils.data_parallel import DataParallel


def get_model(opt):
    # define backbone
    backbones = {"resnet": {"net": ResNet,
                            "out_index": [1, 2, 3],
                            "channel": {18: [128, 256, 512], 34: [128, 256, 512], 50: [512, 1024, 2048],
                                        101: [512, 1024, 2048], 152: [512, 1024, 2048]}},
                 "resnetvd": {"net": ResNetVd,
                              "out_index": [1, 2, 3],
                              "channel": {18: [128, 256, 512], 34: [128, 256, 512], 50: [512, 1024, 2048],
                                          101: [512, 1024, 2048], 152: [512, 1024, 2048]}},
                 "resnext": {"net": ResNeXt,
                             "out_index": [1, 2, 3],
                             "channel": {50: [512, 1024, 2048], 101: [512, 1024, 2048], 152: [512, 1024, 2048]}},
                 "darknet": {"net": Darknet,
                             "out_index": (3, 4, 5),
                             "channel": {53: [256, 512, 1024]}},
                 "cspdarknet": {"net": CSPDarknet53,
                                "out_index": (3, 4, 5),
                                "channel": {53: [256, 512, 1024]}}}

    name, num_layer = opt.backbone.lower().split("_")
    num_layer = int(num_layer)
    assert name in backbones, "backbone should be in {}, your setting {}".format(backbones.keys(), name)

    in_channels = backbones[name]['channel'][num_layer]
    out_indices = backbones[name]['out_index']
    backbone = backbones[name]['net'](depth=num_layer, out_indices=out_indices, pretrained=opt.pretrained)

    # define neck
    if "pan" == opt.neck.lower():
        act = "leaky"  # "mish"
        neck = PPYOLOPAN(in_channels=in_channels, norm_type='bn', act=act, conv_block_num=3,
                         drop_block=opt.drop_block, block_size=3, keep_prob=0.9, spp=opt.spp)
    elif "fpn" == opt.neck.lower():
        neck = PPYOLOFPN(in_channels=in_channels, coord_conv=True, drop_block=opt.drop_block, block_size=3,
                         keep_prob=0.9, spp=opt.spp)
    else:
        raise ValueError("not support opt.neck {}".format(opt.neck))

    # define head
    head = YOLOv3Head(in_channels=[1024, 512, 256], anchors=opt.anchors, anchor_masks=opt.anchor_masks,
                      num_classes=opt.num_classes, iou_aware=opt.iou_aware, iou_aware_factor=0.5)

    # define loss
    loss = YOLOv3Loss(num_classes=opt.num_classes, ignore_thresh=0.7, downsample=opt.downsample,
                      label_smooth=opt.label_smooth, scale_x_y=1.05,
                      iou_loss=IouLoss(loss_weight=2.5, loss_square=True) if opt.iou_loss else None,
                      iou_aware_loss=IouAwareLoss(loss_weight=1.0, call_grad=False) if opt.iou_aware else None)

    model = PPYOLO(opt, backbone=backbone, neck=neck, yolo_head=head, loss=loss)
    return model


class PPYOLO(nn.Module):
    def __init__(self,
                 opt,
                 backbone=ResNet(depth=50),
                 neck=PPYOLOPAN(),
                 yolo_head=YOLOv3Head(),
                 loss=YOLOv3Loss(),
                 for_mot=False):
        super(PPYOLO, self).__init__()
        self.opt = opt
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.loss = loss
        self.for_mot = for_mot
        self.input_shape = [self.opt.input_h, self.opt.input_w]

        self.backbone.init_weights()
        self.neck.init_weights()
        self.yolo_head.init_weights()

    def forward(self, inputs, return_loss=False, return_pred=True, vis_thresh=0.001):
        assert return_loss or return_pred

        loss, predicts = "", ""
        with torch.cuda.amp.autocast(enabled=self.opt.use_amp):
            body_feats = self.backbone(inputs['image'])
            neck_feats = self.neck(body_feats, self.for_mot)
            yolo_outputs = self.yolo_head(neck_feats)
            # print('yolo_outputs:', [i.dtype for i in yolo_outputs])  # float16 when use_amp=True
            if return_loss:
                loss = self.loss(yolo_outputs, inputs, self.yolo_head.anchors)
                # for k, v in loss.items():
                #     print(k, v, v.dtype)  # always float32

        if return_pred:
            if self.opt.iou_aware:
                y_outputs = []
                for i, out in enumerate(yolo_outputs):
                    na = len(self.yolo_head.anchors[i])
                    ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                    b, c, h, w = x.shape
                    no = c // na
                    x = x.view((b, na, no, h * w))
                    ioup = ioup.view((b, na, 1, h * w))
                    obj = x[:, :, 4:5, :]
                    ioup = torch.sigmoid(ioup)
                    obj = torch.sigmoid(obj)
                    obj_t = (obj ** (1 - self.yolo_head.iou_aware_factor)) * (ioup ** self.yolo_head.iou_aware_factor)
                    obj_t = de_sigmoid(obj_t)
                    loc_t = x[:, :, :4, :]
                    cls_t = x[:, :, 5:, :]
                    y_t = torch.cat([loc_t, obj_t, cls_t], dim=2)
                    y_t = y_t.view((b, c, h, w))
                    y_outputs.append(y_t)
            else:
                y_outputs = yolo_outputs

            predicts = bbox_post_process(y_outputs, self.yolo_head.anchors, self.loss.downsample,
                                         self.opt.label_name, inputs['im_shape'], self.input_shape,
                                         conf_thres=vis_thresh, iou_thres=0.5, max_det=300, multi_label=True)
        if return_loss:
            return predicts, loss
        else:
            return predicts


def set_device(model_with_loss, optimizer, opt):
    if len(opt.gpus) > 1:
        model_with_loss = DataParallel(model_with_loss, device_ids=opt.gpus, chunk_sizes=opt.chunk_sizes).to(opt.device)
    else:
        model_with_loss = model_with_loss.to(opt.device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=opt.device, non_blocking=True)
    return model_with_loss, optimizer


class Detector(object):
    def __init__(self, cfg):
        self.opt = cfg
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.opt.pretrained = None
        self.model = get_model(self.opt)
        self.model = load_model(self.model, self.opt.load_model)
        self.model.to(self.opt.device)
        self.model.eval()

    @staticmethod
    def letterbox(img, dst_shape, color=(0, 0, 0)):
        if type(dst_shape) == int:
            dst_shape = [dst_shape, dst_shape]

        ratio_dst = dst_shape[0] / float(dst_shape[1])
        img_h, img_w = img.shape[0], img.shape[1]

        ratio_org = img_h / float(img_w)
        if ratio_dst > ratio_org:
            scale = dst_shape[1] / float(img_w)
        else:
            scale = dst_shape[0] / float(img_h)

        new_shape = (int(round(img_w * scale)), int(round(img_h * scale)))

        dw = (dst_shape[1] - new_shape[0]) / 2  # width padding
        dh = (dst_shape[0] - new_shape[1]) / 2  # height padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
        return img

    @staticmethod
    def letterbox2(img, dst_shape):
        input_h, input_w = dst_shape
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        return inp

    def run(self, images, vis_thresh=0.001):
        batch_img = True
        if np.ndim(images) == 3:
            images = [images]
            batch_img = False

        with torch.no_grad():
            img_shapes = []
            inp_imgs = np.zeros([len(images), 3, self.opt.input_h, self.opt.input_w], dtype=np.float32)
            for b_i, image in enumerate(images):
                img_h, img_w = image.shape[0], image.shape[1]
                resize_img = self.letterbox(image, [self.opt.input_h, self.opt.input_w])
                # resize_img = self.letterbox2(image, [self.opt.input_h, self.opt.input_w])
                img = ((resize_img / 255. - self.opt.mean) / self.opt.std).astype(np.float32)
                img = img.transpose(2, 0, 1).reshape(3, self.opt.input_h, self.opt.input_w)

                inp_imgs[b_i] = img
                img_shapes.append([img_h, img_w])

            inp_imgs = torch.from_numpy(inp_imgs).to(self.opt.device)
            predicts = self.model(dict(image=inp_imgs, im_shape=img_shapes), vis_thresh=vis_thresh)

        if batch_img:
            return predicts
        else:
            return predicts[0]
