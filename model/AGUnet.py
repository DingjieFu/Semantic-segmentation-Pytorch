#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:AGUnet.py
@Author:夜卜小魔王
@Date:2023/03/26 18:46:21
    Note: Attention Gate Unet
    - 跳跃连接处加入Attention Gate, 对编码阶段的特征赋以注意力权重
    - <双线性内插>
    - Total params: 7,790,895
'''

import torch
import torch.nn as nn
from torchsummary import summary


class AGUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AGUNet, self).__init__()
        channels = [64, 128, 256, 512]
        # Encoder
        self.conv1 = ConvBlock(in_ch, channels[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(channels[0], channels[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(channels[1], channels[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(channels[2], channels[3])
        # # Decoder
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)

        self.upConv3 = ConvBlock(channels[3], channels[2])
        self.upConv2 = ConvBlock(channels[2], channels[1])
        self.upConv1 = ConvBlock(channels[1], channels[0])

        self.att3 = AGBlock(
            F_g=channels[2], F_l=channels[2], F_int=channels[2]//2)
        self.att2 = AGBlock(
            F_g=channels[1], F_l=channels[1], F_int=channels[1]//2)
        self.att1 = AGBlock(
            F_g=channels[0], F_l=channels[0], F_int=channels[0]//2)

        self.output = nn.Conv2d(channels[0], out_ch, 1)

    def forward(self, x):
        # encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        # decoder
        up3 = self.up3(conv4)
        x3 = self.att3(up3, conv3)
        merge3 = torch.cat([x3, up3], dim=1)
        upConv3 = self.upConv3(merge3)

        up2 = self.up2(upConv3)
        x2 = self.att2(up2, conv2)
        merge2 = torch.cat([x2, up2], dim=1)
        upConv2 = self.upConv2(merge2)

        up1 = self.up1(upConv2)
        x1 = self.att1(up1, conv1)
        merge1 = torch.cat([x1, up1], dim=1)
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


# 注意力模块
class AGBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AGBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


if __name__ == "__main__":
    model = AGUNet(3, 6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print(model)
    summary(model, input_size=(3, 256, 256))
