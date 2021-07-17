#!/usr/bin/env bash

#python evaluate.py gpus='0' backbone="resnet50" vis_thresh=0.001 load_model="exp/res50_512x512_coco/model_best.pth" input_h=512 input_w=512
python evaluate.py gpus='0' backbone="resnet50" pretrained=None vis_thresh=0.001 load_model="exp/res50_512x512_coco_dropBlock_spp/model_best.pth" exp_id="res50_512x512_coco_dropBlock_spp" drop_block=True spp=True input_h=512 input_w=512
