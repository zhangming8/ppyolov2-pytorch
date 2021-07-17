import cv2
import tqdm
import numpy as np
import pycocotools.coco as coco_

from utils.image import get_affine_transform, affine_transform


def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
    return bbox


def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i


def x1y1wh_x1y1x2y2(box):
    x1, y1, w, h = box
    assert w > 0 and h > 0
    return [x1, y1, x1 + w, y1 + h]


def main():
    coco = coco_.COCO(annot_path)
    images = coco.getImgIds()
    num_samples = len(images)

    input_w, input_h = 512, 512
    max_objs = 100
    _valid_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90]
    cat_ids = {v: i for i, v in enumerate(_valid_ids)}

    for index in tqdm.tqdm(range(num_samples)):
        img_id = images[index]
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ids=ann_ids)
        image = coco.loadImgs(ids=[img_id])[0]

        num_objs = min(len(anns), max_objs)
        # print(image)
        file_name = image['file_name']
        height = image["height"]
        width = image["width"]

        # img_path = img_dir + "/" + file_name
        # img = cv2.imread(img_path)

        horizon_flip, rot_angle = False, 0

        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.
        # random crop
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = _get_border(128, width)
        h_border = _get_border(128, height)
        c[0] = np.random.randint(low=w_border, high=width - w_border)
        c[1] = np.random.randint(low=h_border, high=height - h_border)

        if np.random.random() < 0.5:
            horizon_flip = True
            c[0] = width - c[0] - 1
        rot_angle = np.random.choice(list(range(-2, 2 + 1)))

        trans_input = get_affine_transform(c, s, rot_angle, [input_w, input_h])
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        # inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=interp_method, borderValue=0)

        for k in range(num_objs):
            ann = anns[k]
            # if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
            #     print("img_id {}, w={}, h={}".format(img_id, ann['bbox'][2], ann['bbox'][3]))

            bbox = _coco_box_to_bbox(ann['bbox'])
            cls_id = int(cat_ids[ann['category_id']])

            if horizon_flip:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[:2] = affine_transform(bbox[:2], trans_input)
            bbox[2:] = affine_transform(bbox[2:], trans_input)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, input_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, input_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h < 0:
                print("error: bbox {}, y1({}) > y2({}), img_id {}".format(bbox.tolist(), bbox[1], bbox[3], img_id))
                print("org box: {}, rot {}".format(ann['bbox'], rot_angle))
            if w < 0:
                print("error: bbox {}, x1({}) > x2({}), img_id {}".format(bbox.tolist(), bbox[0], bbox[2], img_id))
                print("org box: {}, rot {}".format(ann['bbox'], rot_angle))


if __name__ == "__main__":
    annot_path = "/media/ming/DATA1/dataset/coco2017/annotations/instances_train2017.json"
    img_dir = ""
    main()
