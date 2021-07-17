import time
import numpy as np
import torch
import torchvision

from models.bbox_utils import make_grid


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def bbox_post_process(yolo_head_outs, anchors, downsamples, label_name, org_shapes, input_shape, conf_thres, iou_thres,
                      max_det, multi_label):
    outs = []
    for p, anchor, downsample in zip(yolo_head_outs, anchors, downsamples):
        bs, c, grid_h, grid_w = p.shape
        na = len(anchor)
        anchor = torch.tensor(anchor, dtype=p.dtype, device=p.device)
        anchor = anchor.view((1, na, 1, 1, 2))
        grid = make_grid(grid_h, grid_w, p.dtype, p.device).view((1, 1, grid_h, grid_w, 2))
        p = p.view((bs, na, -1, grid_h, grid_w)).permute((0, 1, 3, 4, 2))

        x, y = torch.sigmoid(p[:, :, :, :, 0:1]), torch.sigmoid(p[:, :, :, :, 1:2])
        w, h = torch.exp(p[:, :, :, :, 2:3]), torch.exp(p[:, :, :, :, 3:4])
        obj, pcls = torch.sigmoid(p[:, :, :, :, 4:5]), torch.sigmoid(p[:, :, :, :, 5:])

        # rescale to input size, eg: 512x512
        x = (x + grid[:, :, :, :, 0:1]) * downsample
        y = (y + grid[:, :, :, :, 1:2]) * downsample
        w = w * anchor[:, :, :, :, 0:1]
        h = h * anchor[:, :, :, :, 1:2]

        out = torch.cat([x, y, w, h, obj, pcls], -1).view(bs, na * grid_h * grid_w, -1)
        outs.append(out)
    outs = torch.cat(outs, 1)

    outs = non_max_suppression(outs, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
                               multi_label=multi_label)

    # rescale to original image size
    predict = []
    for batch_i, pred in enumerate(outs):
        one_img_res = []
        if pred.shape[0]:
            org_h, org_w = org_shapes[batch_i]
            pred = transform(pred, org_shapes[batch_i], input_shape)
            for p in pred:
                x1, y1, x2, y2, conf, cls_id = p
                label = label_name[int(cls_id)]
                x1 = int(max(0, min(x1, org_w - 1)))
                y1 = int(max(0, min(y1, org_h - 1)))
                x2 = int(max(0, min(x2, org_w - 1)))
                y2 = int(max(0, min(y2, org_h - 1)))
                one_img_res.append([label, float(conf), [x1, y1, x2, y2]])

        predict.append(one_img_res)
    return predict


def transform(box, org_img_shape, img_size):
    x1, y1, x2, y2 = box[:, 0:1], box[:, 1:2], box[:, 2:3], box[:, 3:4]
    org_h, org_w = org_img_shape
    model_h, model_w = img_size
    org_ratio = org_w / float(org_h)
    model_ratio = model_w / float(model_h)
    if org_ratio >= model_ratio:
        # pad h
        scale = org_w / float(model_w)
        x1 = x1 * scale
        x2 = x2 * scale
        pad = (scale * model_h - org_h) / 2
        y1 = scale * y1 - pad
        y2 = scale * y2 - pad
    else:
        # pad w
        scale = org_h / float(model_h)
        y1 = y1 * scale
        y2 = y2 * scale
        pad = (scale * model_w - org_w) / 2
        x1 = x1 * scale - pad
        x2 = x2 * scale - pad
    box[:, 0:1], box[:, 1:2], box[:, 2:3], box[:, 3:4] = x1, y1, x2, y2
    return box


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


if __name__ == "__main__":
    yolo_outs = torch.rand([1, 3 * 2 * 2, 5 + 2])
    print(yolo_outs)
    print(non_max_suppression(yolo_outs, conf_thres=0.7))
