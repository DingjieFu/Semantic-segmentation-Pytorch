#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:main.py
@Author:夜卜小魔王
@Date:2023/02/26 15:05:11
    Note:
        训练模型 数据集6:2:2划分
        最后保存在验证集上性能最好的模型参数
'''

import os
import json
import tqdm
import time
import torch
import logging
import warnings
import numpy as np
import torch.nn as nn
from torch import optim
from MyData import MyDataset
from datetime import datetime
from typing import List, Optional
from torch.optim import Optimizer
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from parameter import parse_args
from Load_Model import ChooseModel
from utils.measure import SegmentationMetric
warnings.filterwarnings("ignore")
args = parse_args()  # load parameters


def WarmupDecorator(lr_scheduler: type, warmup: int) -> type:
    """
        预热装饰器
        - scheduler = WarmupDecorator(CosineAnnealingLR, warmup)(optimizer, T_max, eta_min)
    """
    class WarmupLRScheduler(lr_scheduler):
        def __init__(self, optimizer: Optimizer, *args, **kwargs) -> None:
            self.warmup = warmup
            self.lrs_ori: Optional[List[int]] = None
            super().__init__(optimizer, *args, **kwargs)

        def get_lr(self) -> List[float]:
            # recover
            if self.lrs_ori is not None:
                for p, lr in zip(self.optimizer.param_groups, self.lrs_ori):
                    p["lr"] = lr
            #
            last_epoch = self.last_epoch
            lrs = super().get_lr()
            self.lrs_ori = lrs
            # warmup
            scale = 1
            if last_epoch < self.warmup:
                scale = (last_epoch + 1) / (self.warmup + 1)
            return [lr * scale for lr in lrs]
    return WarmupLRScheduler


# ---------- GPU set ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# ---------- create results file ----------
if not os.path.exists(args.log):
    os.mkdir(args.log)
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())

args.log = args.log + t + '.txt'

# ---------- Run logging Set ----------
logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')
logger = logging.getLogger(__name__)


def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)


# ---------- set seed for random number ----------
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


setup_seed(args.seed)

# ---------- load data ----------
printlog('Loading data')

# 加载训练集、验证集、测试集
train_data = MyDataset(args.root_path, "Train")
train_dataloader = DataLoader(train_data, args.batch_size, shuffle=True)

val_data = MyDataset(args.root_path, "Val")
val_dataloader = DataLoader(val_data, args.batch_size, shuffle=True)

# test_data = MyDataset(args.root_path, "Test")
# test_dataloader = DataLoader(test_data, args.batch_size, shuffle=True)

# train_size = len(train_data)
# val_size = len(val_data)
# test_size = len(test_data)

print('Data loaded')

# ---------- network ------------
net = ChooseModel().to(device)

# ---------- loss & optimizer ------------
optimizer = optim.Adam(net.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
metric = SegmentationMetric(args.num_class)


# ---------- load weight ------------
if os.path.exists(args.model_path):
    net.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    print('successful load weight!')
else:
    print('not successful load weight')

# ---------- load Pre Score ------------
if os.path.exists(args.score_train):
    with open(args.score_train, encoding='utf-8') as f:
        score_dict = json.load(f)
    print('successful load pre best score!')
else:
    score_dict = {
        "Model": args.model_name,
        "Dataset": args.dataset_name,
        "OA": 0,
        "AA": 0,
        "Kappa": 0,
        "mIoU": 0,
        "FWIoU": 0,
        "F1": 0
    }
    print('not successful load pre best score')

# ---------- print parameters ------------
breakout = 0
best_score = score_dict["mIoU"]

printlog('model_name:{}'.format(args.model_name))
printlog('model_name:{}'.format(args.dataset_name))
printlog('batch_size:{}'.format(args.batch_size))
printlog('num_epoch: {}'.format(args.num_epoch))
printlog('initial_lr: {}'.format(args.lr))
printlog('seed: {}'.format(args.seed))
printlog('num_class: {}'.format(args.num_class))

printlog('Start training ...')


train_loss_list = []
dev_loss_list = []
time_list = []
# ----------  epoch  ----------
for epoch in range(args.num_epoch):
    printlog('=' * 40)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()

    start = time.time()

# ----------  train  ----------
    print("----------  Train  ----------")
    net.train()
    scheduler = WarmupDecorator(CosineAnnealingLR, warmup=1)(
        optimizer, 10, args.lr * 0.01)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 10, eta_min=args.lr * 0.01, last_epoch=-1)
    train_loss = 0.0

    progress = tqdm.tqdm(total=len(train_dataloader), ncols=75,
                         desc='Train {}'.format(epoch))

    for batch_id, (batch_initial_image, batch_semantic_image) in enumerate(train_dataloader):
        progress.update(1)

        batch_initial_image, batch_semantic_image = batch_initial_image.to(
            device), batch_semantic_image.to(device)

        # fed data into network
        batch_output = net(batch_initial_image)
        batch_semantic_pred = batch_output.argmax(
            dim=1).squeeze().data.view(-1, 256, 256).to(device)  # 得到0-5的标签值

        # loss
        loss = criterion(batch_output, batch_semantic_image)
        train_loss += loss.to(device).item()

        # metric.addBatch(batch_semantic_pred, batch_semantic_image)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # report train evaluation(训练集其实可以不用看性能)
    train_loss /= len(train_dataloader)
    train_loss_list.append(train_loss)
    # overallAcc = metric.pixelAccuracy()
    # mIoU = metric.meanIntersectionOverUnion()
    # printlog("\n\tepoch_loss:{:.4f}, OA:{:.4f}, K:{:.4f}, mIoU:{:.4f}".format(
    #     epoch_loss, overallAcc, mIoU))
    # metric.reset()

    end = time.time()
    printlog('Training Time: {:.2f}'.format(end - start))
    time_list.append(end-start)
    progress.close()


# ----------  dev  ----------
    print("----------  Eval  ----------")

    with torch.no_grad():
        net.eval()
        dev_loss = 0.0
        progress = tqdm.tqdm(total=len(val_dataloader), ncols=75,
                             desc='Eval {}'.format(epoch))

        for batch_id, (batch_initial_image, batch_semantic_image) in enumerate(val_dataloader):
            progress.update(1)

            batch_initial_image, batch_semantic_image = batch_initial_image.to(
                device), batch_semantic_image.to(device)

            # fed data into network
            batch_output = net(batch_initial_image)
            batch_semantic_pred = batch_output.argmax(
                dim=1).squeeze().data.view(-1, 256, 256).to(device)  # 得到0-5的标签值

            # loss
            loss = criterion(batch_output, batch_semantic_image)
            dev_loss += loss.to(device).item()

            metric.addBatch(batch_semantic_pred.cpu(),
                            batch_semantic_image.cpu())

    # report dev evaluation
    dev_loss /= len(val_dataloader)
    dev_loss_list.append(dev_loss)
    dev_overallAcc = metric.pixelAccuracy()
    dev_averageAcc = metric.meanPixelAccuracy()
    dev_kappa = metric.Kappa_score()
    dev_mIoU = metric.meanIntersectionOverUnion()
    dev_FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    dev_F1 = metric.F1_score()
    print("\n\tdev_loss:{:.4f}, OA:{:.4f}, AA:{:.4f}, K:{:.4f}, mIoU:{:.4f}, FWIoU:{:.4f}, F1:{:.4f}".format(
        dev_loss, dev_overallAcc, dev_averageAcc, dev_kappa, dev_mIoU, dev_FWIoU, dev_F1))
    metric.reset()

    progress.close()
    breakout += 1

    # ---------- record the best result & early stop - ---------
    print('=' * 40)
    printlog('Now epoch: {}'.format(epoch))
    printlog('Now dev loss: {}'.format(dev_loss))
    printlog('Now dev OA: {}'.format(dev_overallAcc))
    printlog('Now dev AA: {}'.format(dev_averageAcc))
    printlog('Now dev Kappa: {}'.format(dev_kappa))
    printlog('Now dev mIoU: {}'.format(dev_mIoU))
    printlog('Now dev FWIoU: {}'.format(dev_FWIoU))
    printlog('Now dev F1: {}'.format(dev_F1))

    score = dev_mIoU
    if score > best_score:
        best_score = score
        breakout = 0
        torch.save(net.state_dict(), args.model_path)

        # ---------- create json file ----------
        score_dict["OA"] = dev_overallAcc
        score_dict["AA"] = dev_averageAcc
        score_dict["Kappa"] = dev_kappa
        score_dict["mIoU"] = dev_mIoU
        score_dict["FWIoU"] = dev_FWIoU
        score_dict["F1"] = dev_F1
        printlog("*"*40)
        printlog('Best dev OA: {}'.format(score_dict["OA"]))
        printlog('Best dev AA: {}'.format(score_dict["AA"]))
        printlog('Best dev Kappa: {}'.format(score_dict["Kappa"]))
        printlog('Best dev mIoU: {}'.format(score_dict["mIoU"]))
        printlog('Best dev FWIoU: {}'.format(score_dict["FWIoU"]))
        printlog('Best dev F1: {}'.format(score_dict["F1"]))
        score_json = json.dumps(score_dict, indent=4)
        with open(args.score_train, 'w', newline='\n') as f:
            f.write(score_json)

    printlog('Breakout: {}'.format(breakout))
    # 若连续训练十个epoch精度没有提升(刚好是一次学习率调整周期),就退出训练
    if breakout == 10:
        break

printlog("#"*40)
printlog('Average train time per epoch: {}'.format(np.mean(time_list)))

# ---------- Plot&Save loss curve ----------
plt.title('Loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.tick_params(labelsize=10)
plt.plot(range(len(train_loss_list)), train_loss_list)
plt.plot(range(len(dev_loss_list)), dev_loss_list)
plt.legend(['train_loss', 'dev_loss'])
plt.savefig("{}/Loss_curve.png".format(args.log))
plt.show()
