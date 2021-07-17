# -*- coding: utf-8 -*-
# @Time    : 2021/7/5 12:48
# @Author  : MingZhang
# @Email   : zm19921120@126.com

from bisect import bisect_right
from typing import List
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

__all__ = ["get_lr_scheduler"]


def get_lr_scheduler(optimizer, total_iteration, lr_decay_step=None, warmup_iters=None, lr_type="multistep",
                     delay_iters=None):
    lr_type = lr_type.lower()

    if lr_type == "cosine":
        return CosineAnnealingLR(optimizer, total_iteration, eta_min=0.00000077)
    elif lr_type == "warmup_cosine":
        assert warmup_iters is not None, "please set 'warmup_iters'"
        assert delay_iters is not None, "please set 'delay_iters'"
        return DelayedCosineAnnealingLR(optimizer, delay_iters=delay_iters, max_iters=total_iteration,
                                        eta_min_lr=optimizer.param_groups[0]['lr']/1e3, warmup_factor=0.001,
                                        warmup_iters=warmup_iters, warmup_method="linear")
    elif lr_type == "multistep":
        assert lr_decay_step is not None, "please set 'lr_decay_step'"
        return MultiStepLR(optimizer, lr_decay_step, gamma=0.1)
    elif lr_type == "warmup_multistep":
        assert lr_decay_step is not None, "please set 'lr_decay_step'"
        assert warmup_iters is not None, "please set 'warmup_iters'"
        return WarmupMultiStepLR(optimizer, lr_decay_step, warmup_iters=warmup_iters, warmup_factor=0.001)
    else:
        raise ValueError("'lr_type' should be in ['cosine', 'warmup_cosine', 'multistep', 'warmup_multistep'], "
                         "your set {}".format(lr_type))


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


class DelayedScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Starts with a flat lr schedule until it reaches N epochs the applies a scheduler
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        delay_iters: number of epochs to keep the initial lr until starting applying the scheduler
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, delay_iters, after_scheduler, warmup_factor, warmup_iters, warmup_method):
        self.delay_epochs = delay_iters
        self.after_scheduler = after_scheduler
        self.finished = False
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [base_lr * warmup_factor for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
        else:
            return super(DelayedScheduler, self).step(epoch)


def DelayedCosineAnnealingLR(optimizer, delay_iters, max_iters, eta_min_lr, warmup_factor,
                             warmup_iters, warmup_method, **kwargs, ):
    cosine_annealing_iters = max_iters - delay_iters
    base_scheduler = CosineAnnealingLR(optimizer, cosine_annealing_iters, eta_min_lr)
    return DelayedScheduler(optimizer, delay_iters, base_scheduler, warmup_factor, warmup_iters, warmup_method)


def main():
    import matplotlib.pyplot as plt
    import torch.optim as optim
    from torchvision.models import resnet18

    base_lr = 0.01  # 0.01
    num_epoch = 300
    batch_size = 32
    dataset_num = 115200
    lr_decay_epoch = [5]
    warmup_iters = 9000

    one_epoch_iteration = dataset_num // batch_size
    total_iteration = num_epoch * one_epoch_iteration
    lr_decay_step = [i * one_epoch_iteration for i in lr_decay_epoch]

    model = resnet18(num_classes=10, pretrained=False)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True)

    # ['cosine', 'warmup_cosine', 'multistep', 'warmup_multistep']
    # lr_scheduler = get_lr_scheduler(optimizer, total_iteration, lr_type="cosine")
    lr_scheduler = get_lr_scheduler(optimizer, total_iteration, warmup_iters=warmup_iters, delay_iters=lr_decay_step[0],
                                    lr_type="warmup_cosine")
    # lr_scheduler = get_lr_scheduler(optimizer, total_iteration, lr_decay_step=lr_decay_step, lr_type="multistep")
    # lr_scheduler = get_lr_scheduler(optimizer, total_iteration, warmup_iters=warmup_iters, lr_decay_step=lr_decay_step,
    #                                 lr_type="warmup_multistep")

    y = []
    y2 = []
    x2 = []
    for epoch in range(1, num_epoch + 1):
        for batch_i in range(1, one_epoch_iteration + 1):
            # optimizer.zero_grad()
            # optimizer.step()
            y.append(optimizer.param_groups[0]['lr'])
            iteration = (epoch - 1) * one_epoch_iteration + batch_i
            print(
                'total epoch: [{}/{}], total iter[{}/{}], epoch iter[{}/{}] lr: {}'.format(epoch, num_epoch, iteration,
                                                                                           total_iteration, batch_i,
                                                                                           one_epoch_iteration,
                                                                                           optimizer.param_groups[0][
                                                                                               'lr']))
            lr_scheduler.step()
        y2.append(optimizer.param_groups[0]['lr'])
        x2.append(epoch)

    plt.figure(1)
    plt.plot(y, c='r', label='warmup step_lr', linewidth=1)
    plt.figure(2)
    plt.plot(x2, y2, c='b', label='warmup epoch_lr', linewidth=1)
    plt.show()


if __name__ == '__main__':
    main()
