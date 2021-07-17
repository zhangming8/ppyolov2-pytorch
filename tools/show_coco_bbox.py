# -*- coding: utf-8 -*-
# @Time    : 2021/7/9 11:24
# @Author  : MingZhang
# @Email   : zm19921120@126.com

'''
可视化coco目标检测的框
'''

import json
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt


colors_hp = [(255, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),
             (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0),
             (0, 0, 255)]


label_color = [[31, 0, 255], [0, 159, 255], [255, 95, 0], [255, 19, 0], [255, 0, 0], [255, 38, 0], [0, 255, 25],
               [255, 0, 133],
               [255, 172, 0], [108, 0, 255], [0, 82, 255], [0, 255, 6], [255, 0, 152], [223, 0, 255], [12, 0, 255],
               [0, 255, 178],
               [108, 255, 0], [184, 0, 255], [255, 0, 76], [146, 255, 0], [51, 0, 255], [0, 197, 255], [255, 248, 0],
               [255, 0, 19],
               [255, 0, 38], [89, 255, 0], [127, 255, 0], [255, 153, 0], [0, 255, 255], [0, 255, 216], [0, 255, 121],
               [255, 0, 248],
               [70, 0, 255], [0, 255, 159], [0, 216, 255], [0, 6, 255], [0, 63, 255], [31, 255, 0], [255, 57, 0],
               [255, 0, 210],
               [0, 255, 102], [242, 255, 0], [255, 191, 0], [0, 255, 63], [255, 0, 95], [146, 0, 255], [184, 255, 0],
               [255, 114, 0],
               [0, 255, 235], [255, 229, 0], [0, 178, 255], [255, 0, 114], [255, 0, 57], [0, 140, 255], [0, 121, 255],
               [12, 255, 0],
               [255, 210, 0], [0, 255, 44], [165, 255, 0], [0, 25, 255], [0, 255, 140], [0, 101, 255], [0, 255, 82],
               [223, 255, 0],
               [242, 0, 255], [89, 0, 255], [165, 0, 255], [70, 255, 0], [255, 0, 172], [255, 76, 0], [203, 255, 0],
               [204, 0, 255],
               [255, 0, 229], [255, 133, 0], [127, 0, 255], [0, 235, 255], [0, 255, 197], [255, 0, 191], [0, 44, 255],
               [50, 255, 0],
               [31, 0, 255], [0, 159, 255], [255, 95, 0], [255, 19, 0], [255, 0, 0], [255, 38, 0], [0, 255, 25],
               [255, 0, 133],
               [255, 172, 0], [108, 0, 255], [0, 82, 255], [0, 255, 6], [255, 0, 152], [223, 0, 255], [12, 0, 255],
               [0, 255, 178],
               [108, 255, 0], [184, 0, 255], [255, 0, 76], [146, 255, 0], [51, 0, 255], [0, 197, 255], [255, 248, 0],
               [255, 0, 19]]


edges = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
         [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]

ec = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 0, 0),
      (255, 0, 0), (0, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 0, 0), (255, 0, 0),
      (0, 0, 255), (0, 0, 255)]


def add_coco_hp(img, keypoints, categories):
    keypoints = np.reshape(keypoints, (-1, 3)).astype(np.int)
    for keypoints_index in range(len(keypoints)):
        x, y, visable = keypoints[keypoints_index, :]
        if visable == 0:
            # not labeled
            continue
        elif visable == 1:
            # labeled but not visable
            color = colors_hp[keypoints_index]
        elif visable == 2:
            # labeled and visable
            color = colors_hp[keypoints_index]
        else:
            raise ValueError
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
        cv2.putText(img, categories["keypoints"][keypoints_index], (max(0, int(x) - 40), int(y)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

    for j, e in enumerate(edges):
        if keypoints[e].min() > 0:
            cv2.line(img, (keypoints[e[0], 0], keypoints[e[0], 1]), (keypoints[e[1], 0], keypoints[e[1], 1]), ec[j], 2,
                     lineType=cv2.LINE_AA)


def x1y1wh_x1y1x2y2(box):
    x1, y1, w, h = box
    assert w > 0 and h > 0
    return [x1, y1, x1 + w, y1 + h]


def convert_annotation(anns):
    converted_ann = {}
    for ann in anns:
        if ann["image_id"] in converted_ann:
            converted_ann[ann["image_id"]].append(ann)
        else:
            converted_ann[ann["image_id"]] = [ann]
    return converted_ann


def main():
    data = json.load(open(json_file, 'r'))

    categories = data['categories']
    images = data['images']
    annotations = data['annotations']

    print("categories:", categories)
    # print("images:", images)
    print("images length:", len(images))
    # print("annotations:", annotations)
    categories_new = {}
    for cat in categories:
        categories_new[cat["id"]] = cat["name"]
    print("categories_new:", categories_new)

    annotations = convert_annotation(annotations)
    img_num = len(images)
    img_size = []
    for i in tqdm.tqdm(range(img_num)):
        image = images[i]
        filename = image["file_name"]
        height = image["height"]
        width = image["width"]
        img_id = image["id"]
        img_size.append([height, width])

        print("-" * 50)
        img_path = img_dir + '/' + filename
        img = cv2.imread(img_path)
        print("{}/{}, reading img {}".format(i, img_num, img_path))

        if img_id not in annotations:
            print("{} not labeled".format(img_id))
            continue
        if img is None:
            print("img is none {}".format(img_path))
            continue

        ann_i = annotations[img_id]
        for idx, ann in enumerate(ann_i):
            # segmentation = ann["segmentation"]
            # num_keypoints = ann["num_keypoints"]
            area = ann["area"]
            iscrowd = ann["iscrowd"]
            # keypoints = ann["keypoints"]
            bbox = ann["bbox"]
            bbox = x1y1wh_x1y1x2y2(bbox)
            category_id = ann["category_id"]

            # draw bounding box
            color = label_color[category_id]
            # color = colors_hp[category_id]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color)
            cv2.putText(img, str(categories_new[category_id]), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX,
                        1, color)
            cv2.circle(img, ((int(bbox[0]) + int(bbox[2])) // 2, (int(bbox[1]) + int(bbox[3])) // 2), 4, color, -1)

            # draw keypoints
            # keypoints = np.reshape(keypoints, (-1, 3))
            # add_coco_hp(img, keypoints, categories)

        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == 27:
            exit()

    hs = [i[0] for i in img_size]
    ws = [i[1] for i in img_size]

    plt.figure(1)
    plt.hist(hs, bins=50)
    plt.title('height(mean {})'.format(int(np.mean(hs))))

    plt.figure(2)
    plt.hist(ws, bins=50)
    plt.title('width(mean {})'.format(int(np.mean(ws))))
    plt.show()


if __name__ == "__main__":
    # json_file = r"D:\work\public_dataset\coco2017\annotations\instances_val2017.json"
    # img_dir = r"D:\work\public_dataset\coco2017\images\val2017"
    json_file = r"D:\work\public_dataset\coco2017\annotations\instances_train2017.json"
    # json_file = "/media/ming/DATA1/dataset/coco2017/annotations/instances_train2017_org.json"
    img_dir = r"D:\work\public_dataset\coco2017\images\train2017"

    # json_file = "person_keypoints_val2017.json"
    # img_dir = "../images/val2017"

    main()
