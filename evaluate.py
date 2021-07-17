# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 14:59
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import os
import cv2
import tqdm

from config import opt
from models.ppyolo import Detector
from dataset.coco_dataset import LoadCOCO


def evaluate():
    detector = Detector(opt)
    val_dataset = LoadCOCO(opt, "val")

    img_dir = val_dataset.img_dir
    gt_annot_path = val_dataset.annot_path
    result_p = "./"

    coco = val_dataset.coco
    images = coco.getImgIds()
    num_samples = len(images)
    print('find {} samples in {}'.format(num_samples, gt_annot_path))

    results = {}
    for index in tqdm.tqdm(range(num_samples)):
        img_id = images[index]
        file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
        image_path = img_dir + "/" + file_name
        assert os.path.isfile(image_path), "not find {}".format(image_path)
        img = cv2.imread(image_path)
        pred = detector.run(img, vis_thresh=opt.vis_thresh)
        results[img_id] = pred

    val_dataset.run_eval(results, save_dir=result_p)
    os.remove(result_p + "/results.json")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.load_model = opt.load_model if opt.load_model != "" else os.path.join(opt.save_dir, "model_best.pth")

    evaluate()
