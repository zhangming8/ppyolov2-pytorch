# -*- coding: utf-8 -*-
# @Time    : 2021/6/24 14:59
# @Author  : MingZhang
# @Email   : zm19921120@126.com

from __future__ import print_function, division
import os
import shutil

import cv2
import time
import numpy as np
from progress.bar import Bar
import torch
from torch.utils.data import DataLoader

from dataset.coco_dataset import LoadCOCO
from models.ppyolo import get_model, set_device
from utils.lr_scheduler import get_lr_scheduler
from utils.utils import AverageMeter, write_log
from utils.model_utils import EMA, save_model, load_model, ensure_same, clip_grads
from config import opt


def run_epoch(model_with_loss, optimizer, scaler, ema, phase, epoch, data_loader, one_epoch_iter, total_iter,
              lr_scheduler=None, accumulate=1):
    if phase == 'train':
        model_with_loss.train()
    else:
        model_with_loss.eval()
        torch.cuda.empty_cache()

    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {}
    num_iters = len(data_loader)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    end = time.time()
    last_opt_iter = 0
    optimizer.zero_grad() if phase == 'train' else ""
    for batch_i, batch in enumerate(data_loader):
        iter_id = batch_i + 1
        iteration = (epoch - 1) * one_epoch_iter + iter_id
        data_time.update(time.time() - end)
        for k in batch:
            batch[k] = batch[k].to(device=opt.device, non_blocking=True)

        return_pred = phase != 'train' and "ap" in opt.metric.lower()
        preds, loss_stats = model_with_loss(batch, return_loss=True, return_pred=return_pred)
        lr, shapes = optimizer.param_groups[0]['lr'], "x".join([str(i) for i in batch["image"].shape])
        if phase == 'train':
            scaler.scale(loss_stats["loss"]).backward()

            if iteration - last_opt_iter >= accumulate or iter_id == num_iters:
                if opt.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    grad_normal = clip_grads(model_with_loss, opt.grad_clip)
                    if not opt.use_amp:
                        loss_stats['grad_normal'] = grad_normal
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update_params() if ema else ''
                last_opt_iter = iteration

            lr_scheduler.step()
            if (iteration - 1) % 50 == 0 and epoch <= 10:
                logger.scalar_summary("lr_iter_before_10_epoch", lr, iteration)

        batch_time.update(time.time() - end)
        end = time.time()
        if return_pred:
            for img_id, pred in zip(batch['img_id'].cpu().numpy().tolist(), preds):
                results[img_id] = pred

        Bar.suffix = '{phase}: total epoch[{0}/{1}] total batch[{2}/{3}] batch[{4}/{5}] |size: {6} |lr: {7} |Tot: ' \
                     '{total:} |ETA: {eta:} '.format(epoch, opt.num_epochs, iteration, total_iter, iter_id, num_iters,
                                                     shapes, "{:.8f}".format(lr), phase=phase, total=bar.elapsed_td,
                                                     eta=bar.eta_td)
        for l in loss_stats:
            if l not in avg_loss_stats:
                avg_loss_stats[l] = AverageMeter()
            avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['image'].size(0))
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if opt.print_iter > 0 and iter_id % opt.print_iter == 0:
            print('{}| {}'.format(opt.exp_id, Bar.suffix))
            logger.write('{}| {}\n'.format(opt.exp_id, Bar.suffix))
        else:
            bar.next()
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results


def train(model, scaler, train_loader, val_loader, optimizer, lr_scheduler, start_epoch, one_epoch_iteration,
          total_iteration, accumulate):
    ema = EMA(model) if opt.ema else None
    best = 1e10 if opt.metric == "loss" else -1
    train_loader.dataset.epoch = start_epoch
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        train_loader.dataset.shuffle()  # shuffle image each epoch
        logger.scalar_summary("lr_epoch", optimizer.param_groups[0]['lr'], epoch)
        loss_dict_train, _ = run_epoch(model, optimizer, scaler, ema, "train", epoch, train_loader, one_epoch_iteration,
                                       total_iteration, lr_scheduler, accumulate)
        logger.write('train epoch: {} |'.format(epoch))
        write_log(loss_dict_train, logger, epoch, "train")

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            if ema is not None:
                ema.apply_shadow()
                # ensure_same(ema.model, model)
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model)
            logger.write('----------epoch {} evaluating----------\n'.format(epoch))
            with torch.no_grad():
                loss_dict_val, preds = run_epoch(model, optimizer, scaler, ema, "val", epoch, val_loader,
                                                 one_epoch_iteration, total_iteration)
            logger.write('----------epoch {} evaluate done----------\n'.format(epoch))
            logger.write('val epoch: {} |'.format(epoch))
            write_log(loss_dict_val, logger, epoch, "val")

            if "ap" in opt.metric.lower():
                ap_all, ap_0_5 = val_loader.dataset.run_eval(preds, opt.save_dir)
                logger.write(
                    "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n".format(ap_all))
                logger.write(
                    "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}\n".format(ap_0_5))
                logger.scalar_summary("val_AP", ap_all, epoch)
                logger.scalar_summary("val_AP_05", ap_0_5, epoch)
                if ap_all >= best:
                    best = ap_all
                    save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
            elif opt.metric == "loss":
                if loss_dict_val['loss'] <= best:
                    best = loss_dict_val['loss']
                    save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
            if ema is not None:
                ema.restore()

        if epoch % opt.save_epoch == 0:
            if ema is not None:
                ema.apply_shadow()
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model)
            if ema is not None:
                ema.restore()
        train_loader.dataset.epoch += 1
        save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer, scaler)
    logger.close()


def main():
    train_dataset = LoadCOCO(opt, "train", logger=logger)
    val_dataset = LoadCOCO(opt, "val", logger=logger)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                              pin_memory=True, drop_last=True)
    one_epoch_iteration = len(train_loader)
    total_iteration = opt.num_epochs * one_epoch_iteration
    lr_decay_step = [i * one_epoch_iteration for i in opt.lr_decay_epoch]
    model = get_model(opt)
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp, init_scale=2.**16)
    lr_scheduler = get_lr_scheduler(optimizer, total_iteration, warmup_iters=opt.warmup_iters,
                                    lr_decay_step=lr_decay_step, delay_iters=lr_decay_step[0], lr_type=opt.lr_type)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch, scaler = load_model(model, opt.load_model, optimizer, scaler, opt.resume)
        for _ in range(one_epoch_iteration * start_epoch):
            lr_scheduler.step()
        print('start epoch {} with lr {}'.format(start_epoch, optimizer.param_groups[0]['lr']))

    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    model, optimizer = set_device(model, optimizer, opt)
    train(model, scaler, train_loader, val_loader, optimizer, lr_scheduler, start_epoch, one_epoch_iteration,
          total_iteration, opt.accumulate)


if __name__ == "__main__":
    cv2.setNumThreads(0)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = opt.cuda_benchmark
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    from utils.logger import Logger

    logger = Logger(opt)
    shutil.copyfile("./config.py", logger.log_path + "/config.py")
    main()
