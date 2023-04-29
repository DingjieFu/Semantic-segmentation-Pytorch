#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:SDAUnet.py
@Author:夜卜小魔王
@Date:2023/03/28 14:21:13
    Note: Sequential-Double-Attention Unet
    - 最后<一次>卷积后加入串行通过PAM CAM
    - Total params: 8,819,590
'''

import torch
import torch.nn as nn
from torchsummary import summary


class SDAUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SDAUNet, self).__init__()
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

        self.PAM = PositionAttention(channels[3])
        self.CAM = ChannelAttention(channels[3])

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

        conv4 = self.CAM(self.PAM(conv4))

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


# 位置注意力层
class PositionAttention(nn.Module):
    def __init__(self, in_dim):
        super(PositionAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            x : 输入特征图 (batch_szie · channel_size · Width · Height)
            out : 注意力值 + 输入特征图
        """
        batch_size, channel_size, width, height = x.size()
        query = self.query_conv(x).view(
            batch_size, -1, width*height).permute(0, 2, 1)  # B·(W·H)·C
        key = self.key_conv(x).view(
            batch_size, -1, width*height)  # B·C·(W·H)

        energy = torch.bmm(query, key)
        attention = self.softmax(energy)  # B·(W·H)·(W·H)
        value = self.value_conv(x).view(
            batch_size, -1, width*height)  # B·C·(W·H)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel_size, width, height)
        out = self.gamma * out + x
        return out


# 通道注意力层
class ChannelAttention(nn.Module):
    def __init__(self, in_dim):
        super(ChannelAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            x : 输入特征图 (batch_szie · channel_size · Width · Height)
            out : 注意力值 + 输入特征图
        """
        batch_size, channel_size, width, height = x.size()
        query = self.query_conv(x).view(
            batch_size, -1, width*height)  # B·C·(W·H)
        key = self.key_conv(x).view(
            batch_size, -1, width*height).permute(0, 2, 1)  # B·(W·H)·C
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)  # B·C·C
        value = self.value_conv(x).view(
            batch_size, -1, width*height)  # B·C·(W·H)
        out = torch.bmm(attention, value)
        out = out.view(batch_size, channel_size, width, height)
        out = self.gamma * out + x
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


if __name__ == "__main__":
    model = SDAUNet(3, 6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print(model)
    summary(model, input_size=(3, 256, 256), batch_size=1)
