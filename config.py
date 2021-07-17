# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 17:24
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import sys
import numpy as np
from easydict import EasyDict
from utils.utils import merge_opt

opt = EasyDict()

opt.backbone = "resnet_50"  # resnet_50 resnet_101 resnetvd_50 darknet_53 resnext_50
opt.pretrained = 'torchvision://resnet50'  # None 'open-mmlab://darknet53' 'open-mmlab://resnext50_32x4d' 'path/to/.pth'
opt.neck = "pan"  # "fpn" "pan"
opt.num_epochs = 300
opt.gpus = "0"  # "-1" "0,1,2,3,4,5,6,7"
opt.lr_type = "warmup_multistep"  # "warmup_cosine"
opt.lr_decay_epoch = "240,270"  # set one small value when lr_type="warmup_cosine", eg: 50
opt.batch_size = 24
opt.master_batch_size = -1  # batch size in first gpu
opt.lr = 0.005
opt.warmup_iters = 4000
opt.exp_id = "res50_512x512_coco"
opt.input_h = 512  # be divisible by 32
opt.input_w = 512
opt.vis_thresh = 0.3  # inference confidence
opt.num_workers = 10
# opt.data_dir = "/media/ming/DATA1/dataset/coco2017"  # r"D:\work\public_dataset\coco2017"
opt.data_dir = "/data/dataset/coco_dataset"
opt.annotation = ""  # default: annotation=opt.data_dir+"/annotations"; images=opt.data_dir+"/images"
opt.load_model = ''
opt.ema = True  # False
opt.grad_clip = dict(max_norm=35, norm_type=2)  # None
opt.print_iter = 1
opt.metric = "loss"  # 'mAp' or 'loss', slowly when set 'mAp'
opt.val_intervals = 1
opt.save_epoch = 1
opt.resume = False  # resume from 'model_last.pth'
opt.keep_res = False
opt.no_color_aug = False
opt.multi_input_size = None  # [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
opt.mosaic = 0  # 0.5
opt.mix_up = 0  # 0.5
opt.horizon_flip = 0.5
opt.vertical_flip = 0
opt.rotate90 = 0
opt.rot_angle = 0  # 2
opt.grid_mask = False
opt.spp = False
opt.drop_block = False
opt.iou_loss = True
opt.iou_aware = True
opt.use_amp = False  # True
opt.accumulate = 1  # real batch size = accumulate * batch_size
opt.cuda_benchmark = True
opt.random_interp = True
opt.downsample = [32, 16, 8]
opt.label_smooth = False

opt.label_name = [
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
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']
opt.seed = 317
opt.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
opt.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
if "darknet" in opt.backbone:
    opt.mean_value = [0, 0, 0]
    opt.std_value = [1, 1, 1]
else:
    opt.mean_value = [0.485, 0.456, 0.406][::-1]
    opt.std_value = [0.229, 0.224, 0.225][::-1]

opt = merge_opt(opt, sys.argv[1:])
if opt.multi_input_size is not None:
    opt.cuda_benchmark = False
assert opt.input_h % 32 == 0 and opt.input_w % 32 == 0
opt.mean = np.array(opt.mean_value, dtype=np.float32).reshape([1, 1, 3])
opt.std = np.array(opt.std_value, dtype=np.float32).reshape([1, 1, 3])
opt.num_classes = len(opt.label_name)
opt.gpus_str = opt.gpus
opt.metric = opt.metric.lower()
opt.lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
opt.gpus = [int(i) for i in opt.gpus.split(',')]
if opt.master_batch_size == -1:
    opt.master_batch_size = opt.batch_size // len(opt.gpus)
rest_batch_size = opt.batch_size - opt.master_batch_size
opt.chunk_sizes = [opt.master_batch_size]
for i in range(len(opt.gpus) - 1):
    slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
    if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
    opt.chunk_sizes.append(slave_chunk_size)
opt.pad = 31
opt.root_dir = os.path.dirname(__file__)
opt.save_dir = os.path.join(opt.root_dir, 'exp', opt.exp_id)
if opt.resume and opt.load_model == '':
    opt.load_model = os.path.join(opt.save_dir, 'model_last.pth')
