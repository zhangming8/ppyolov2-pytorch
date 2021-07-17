#!/usr/bin/env bash

#python predict.py gpus='0' backbone="resnet50" img_dir="/data/dataset/coco_dataset/images/val2017" vis_thresh=0.3 load_model="exp/res50_512x512_coco/model_best.pth" input_h=512 input_w=512
python predict.py gpus='0' backbone="resnet50" pretrained=None img_dir="/data/dataset/coco_dataset/images/val2017" vis_thresh=0.3 load_model="exp/res50_512x512_coco_dropBlock_spp/model_best.pth" exp_id="res50_512x512_coco_dropBlock_spp" drop_block=True spp=True input_h=512 input_w=512