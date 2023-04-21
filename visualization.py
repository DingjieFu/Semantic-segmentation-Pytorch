#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:visualization.py
@Author:夜卜小魔王
@Date:2023/03/21 10:04:33
    Note:
        可视化结果
'''

import torch
import random
import warnings
import numpy as np
import torch.nn as nn
from MyData import MyDataset, label2image
import matplotlib.pyplot as plt
from Load_Model import ChooseModel
from parameter import parse_args
from utils.measure import SegmentationMetric
warnings.filterwarnings("ignore")
args = parse_args()


# ---------- GPU set ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# ---------- Model set ----------
net = ChooseModel().to(device)

# ---------- Data set ----------
test_data = MyDataset(args.root_path, mode="Test")

# ---------- Load weight ----------
net.load_state_dict(torch.load(args.model_path, map_location='cpu'))

metric = SegmentationMetric(args.num_class)
# ---------- Get drawings ----------
# 随机选择一张
# index = random.randint(0, len(test_data)-1)
# print(index)
index = 959
test_image, test_label = test_data[index]

output = net(test_image.unsqueeze(0).to(device))
output = nn.Softmax(dim=1)(output)
pred_label = output.argmax(dim=1).squeeze().data.cpu().numpy()

semantic_pred = label2image()(pred_label)
semantic_label = label2image()(test_label)


# ---------- measures ----------
metric.addBatch(pred_label, test_label)
# report dev evaluation
overallAcc = metric.pixelAccuracy()
averageAcc = metric.meanPixelAccuracy()
mIoU = metric.meanIntersectionOverUnion()
FWIoU = metric.Frequency_Weighted_Intersection_over_Union()
print("\n\tOA:{:.4f}, AA:{:.4f}, mIoU:{:.4f}, FWIoU:{:.4f}".format(
    overallAcc, averageAcc, mIoU, FWIoU))
metric.reset()


temp = test_image.numpy()
temp = (temp-np.min(temp)) / (np.max(temp)-np.min(temp))*255

# ---------- plot ----------
fig, ax = plt.subplots(2, 2)

ax[0, 0].imshow(temp.transpose(1, 2, 0).astype("uint8"))
# ax[0, 0].imshow(test_label)
ax[0, 1].imshow(semantic_label)
ax[1, 0].imshow(pred_label)
ax[1, 1].imshow(semantic_pred)

ax[0, 0].set_title('true_picture', fontsize=10)
ax[0, 1].set_title('semantic_true', fontsize=10)
ax[1, 0].set_title('pred_label', fontsize=10)
ax[1, 1].set_title('semantic_pred', fontsize=10)

fig.suptitle('{} Visualization'.format(args.model_name), fontsize=20)
plt.show()
