from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from models.losses.iou_loss import IouLoss
from models.bbox_utils import bbox_iou


class IouAwareLoss(IouLoss):
    """
    iou aware loss, see https://arxiv.org/abs/1912.05992
    Args:
        loss_weight (float): iou aware loss weight, default is 1.0
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self, loss_weight=1.0, giou=False, diou=False, ciou=False, call_grad=True):
        super(IouAwareLoss, self).__init__(loss_weight=loss_weight, giou=giou, diou=diou, ciou=ciou)
        self.call_grad = call_grad

    def __call__(self, ioup, pbox, gbox):
        if self.call_grad:
            iou = bbox_iou(pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        else:
            with torch.no_grad():
                iou = bbox_iou(pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
                # assert iou.requires_grad is False

        loss_iou_aware = F.binary_cross_entropy_with_logits(ioup, iou, reduction='none')
        loss_iou_aware = loss_iou_aware * self.loss_weight
        return loss_iou_aware
