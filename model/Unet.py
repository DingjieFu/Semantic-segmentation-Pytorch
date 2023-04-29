#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:Unet.py
@Author:夜卜小魔王
@Date:2023/02/25 19:23:06
    Note: Unet模型
    - 对称结构 下采样后上采样 跳跃连接
    - <转置卷积>
    - Total params: 7,703,302
    - channels = [64, 128, 256, 512]
    - 三次最大池化 三次上采样
'''


import torch
import torch.nn as nn
from torchsummary import summary


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        channels = [64, 128, 256, 512]
        # Encoder
        self.conv1 = ConvBlock(in_ch, channels[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(channels[0], channels[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(channels[1], channels[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(channels[2], channels[3])
        # Decoder
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.upConv3 = ConvBlock(channels[3], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.upConv2 = ConvBlock(channels[2], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.upConv1 = ConvBlock(channels[1], channels[0])
        self.output = nn.Conv2d(channels[0], out_ch, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        # Decoder
        up3 = self.up3(conv4)
        merge3 = torch.cat([conv3, up3], dim=1)
        upConv3 = self.upConv3(merge3)
        up2 = self.up2(upConv3)
        merge2 = torch.cat([conv2, up2], dim=1)
        upConv2 = self.upConv2(merge2)
        up1 = self.up1(upConv2)
        merge1 = torch.cat([conv1, up1], dim=1)
        upConv1 = self.upConv1(merge1)
        out = self.output(upConv1)
        return out


# 一个卷积块包括两次卷积操作
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# # 两倍上采样 使用双线性插值
# class UpSampling(nn.Module):
#     def __init__(self, in_ch):
#         super(UpSampling, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(in_ch, in_ch//2, 1))

#     def forward(self, x):
#         out = self.up(x)
#         return out


if __name__ == "__main__":
    model = UNet(3, 6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print(model)
    summary(model, input_size=(3, 256, 256), batch_size=1)
