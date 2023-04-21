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
from parameter import parse_args
from Load_Model import ChooseModel
from utils.measure import SegmentationMetric
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
args = parse_args()  # load parameters


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

# 加载训练集(2964)、验证集(988)、测试集(988)
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
optimizer = optim.AdamW(net.parameters(), lr=args.lr)
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
        "OA": 0,
        "AA": 0,
        "mIoU": 0,
        "FWIoU": 0,
    }
    print('not successful load pre best score')

# ---------- print parameters ------------
breakout = 0
best_score = score_dict["mIoU"]

printlog('model_name:{}'.format(args.model_name))
printlog('batch_size:{}'.format(args.batch_size))
printlog('num_epoch: {}'.format(args.num_epoch))
printlog('initial_lr: {}'.format(args.lr))
printlog('seed: {}'.format(args.seed))
printlog('num_class: {}'.format(args.num_class))

printlog('Start training ...')


# ----------  epoch  ----------
for epoch in range(args.num_epoch):
    printlog('=' * 40)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()

    start = time.time()

# ----------  train  ----------
    print("----------  Train  ----------")
    net.train()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=args.lr * 0.01, last_epoch=-1)
    # epoch_loss = 0.0

    progress = tqdm.tqdm(total=len(train_dataloader), ncols=75,
                         desc='Train {}'.format(epoch))

    for batch_id, (batch_initial_image, batch_semantic_image) in enumerate(train_dataloader):
        progress.update(1)

        batch_initial_image, batch_semantic_image = batch_initial_image.to(
            device), batch_semantic_image.to(device)

        # fed data into network
        batch_output = net(batch_initial_image)
        batch_output = nn.Softmax(dim=1)(batch_output)  # 采用softmax得到(0,1)的概率分布
        batch_semantic_pred = batch_output.argmax(
            dim=1).squeeze().data.view(-1, 256, 256).to(device)  # 得到0-5的标签值

        # loss
        loss = criterion(batch_output, batch_semantic_image)
        # epoch_loss += loss.to(device).item()

        # metric.addBatch(batch_semantic_pred, batch_semantic_image)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    # report train evaluation(训练集其实可以不用看性能)
    # epoch_loss /= len(train_dataloader)
    # overallAcc = metric.pixelAccuracy()
    # mIoU = metric.meanIntersectionOverUnion()
    # printlog("\n\tepoch_loss:{:.4f}, OA:{:.4f}, mIoU:{:.4f}".format(
    #     epoch_loss, overallAcc, mIoU))
    # metric.reset()

    end = time.time()
    printlog('Training Time: {:.2f}'.format(end - start))
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
            batch_output = nn.Softmax(dim=1)(
                batch_output)  # 采用softmax得到(0,1)的概率分布
            batch_semantic_pred = batch_output.argmax(
                dim=1).squeeze().data.view(-1, 256, 256).to(device)  # 得到0-5的标签值

            # loss
            loss = criterion(batch_output, batch_semantic_image)
            dev_loss += loss.to(device).item()

            metric.addBatch(batch_semantic_pred.cpu(),
                            batch_semantic_image.cpu())

    # report dev evaluation
    dev_loss /= len(val_dataloader)
    dev_overallAcc = metric.pixelAccuracy()
    dev_averageAcc = metric.meanPixelAccuracy()
    dev_mIoU = metric.meanIntersectionOverUnion()
    dev_FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
    print("\n\tdev_loss:{:.4f}, OA:{:.4f}, AA:{:.4f}, mIoU:{:.4f}, FWIoU:{:.4f}".format(
        dev_loss, dev_overallAcc, dev_averageAcc, dev_mIoU, dev_FWIoU))
    metric.reset()

    progress.close()
    breakout += 1

    # ---------- record the best result & early stop - ---------
    score = dev_mIoU
    if score > best_score:
        best_score = score
        breakout = 0
        torch.save(net.state_dict(), args.model_path)

        # ---------- create json file ----------
        score_dict["OA"] = dev_overallAcc
        score_dict["AA"] = dev_averageAcc
        score_dict["mIoU"] = dev_mIoU
        score_dict["FWIoU"] = dev_FWIoU
        score_json = json.dumps(score_dict, indent=4)
        with open(args.score_train, 'w', newline='\n') as f:
            f.write(score_json)

    print('=' * 40)
    printlog('Now epoch: {}'.format(epoch))
    printlog('Now dev loss: {}'.format(dev_loss))
    printlog('Now dev OA: {}'.format(dev_overallAcc))
    printlog('Now dev AA: {}'.format(dev_averageAcc))
    printlog('Now dev mIoU: {}'.format(dev_mIoU))
    printlog('Now dev FWIoU: {}'.format(dev_FWIoU))
    printlog("__________"*4)
    printlog('Best dev OA: {}'.format(score_dict["OA"]))
    printlog('Best dev AA: {}'.format(score_dict["AA"]))
    printlog('Best dev mIoU: {}'.format(score_dict["mIoU"]))
    printlog('Best dev FWIoU: {}'.format(score_dict["FWIoU"]))
    printlog('Breakout: {}'.format(breakout))

    # 若连续训练十个epoch精度没有提升,就退出训练
    if breakout == 10:
        break
