#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:predict.py
@Author:夜卜小魔王
@Date:2023/03/07 09:38:00
    Note:
        模型测试
'''

import json
import tqdm
import torch
import warnings
import numpy as np
import torch.nn as nn
from MyData import MyDataset
from parameter import parse_args
from Load_Model import ChooseModel
from utils.measure import SegmentationMetric
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")
args = parse_args()


# ---------- GPU set ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# ---------- Model set ----------
net = ChooseModel().to(device)

# ---------- loss & metric ------------
criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
metric = SegmentationMetric(args.num_class)

# ---------- Data set ----------
test_data = MyDataset(args.root_path, "Test")
test_dataloader = DataLoader(test_data, args.batch_size, shuffle=True)

# ---------- Load weight ----------
net.load_state_dict(torch.load(args.model_path, map_location='cpu'))


print("----------  Test  ----------")
with torch.no_grad():
    net.eval()
    progress = tqdm.tqdm(total=len(test_dataloader), ncols=75)
    for batch_id, (batch_initial_image, batch_semantic_image) in enumerate(test_dataloader):
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
        metric.addBatch(batch_semantic_pred.cpu(), batch_semantic_image.cpu())

progress.close()

# report dev evaluation
overallAcc = metric.pixelAccuracy()
averageAcc = metric.meanPixelAccuracy()
mIoU = metric.meanIntersectionOverUnion()
FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
print("\n\tOA:{:.4f}, AA:{:.4f}, mIoU:{:.4f}, FWIoU:{:.4f}".format(
    overallAcc, averageAcc, mIoU, FWIoU))
metric.reset()


# ---------- create json file ----------
score_dict = {
    "Model": args.model_name,
    "OA": 0,
    "AA": 0,
    "mIoU": 0,
    "FWIoU": 0,
}
score_dict["OA"] = overallAcc
score_dict["AA"] = averageAcc
score_dict["mIoU"] = mIoU
score_dict["FWIoU"] = FWIoU
score_json = json.dumps(score_dict, indent=4)
with open(args.score_test, 'w', newline='\n') as f:
    f.write(score_json)
