#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:Load_Model.py
@Author:夜卜小魔王
@Date:2023/04/05 13:56:26
    Note: 
        用来选择模型
'''


import torch
import warnings
from model.Unet import UNet
from model.ResUnet import ResUNet
from model.BottleNeckUnet import BottleNeckUNet


from model.AGUnet import AGUNet
from model.CAUnet import CAUNet
from model.PAUnet import PAUNet
from model.DAUnet import DAUNet
from model.SDAGUnet import SDAGUNet
from model.SDAGResUnet import SDAGResUNet
from model.AGResUnet import AGResUNet
from model.DAResUnet import DAResUNet

from parameter import parse_args
warnings.filterwarnings("ignore")
args = parse_args()  # load parameters


def ChooseModel():
    if args.model_name == "Unet":  # Unet模型
        net = UNet(3, args.num_class)

    elif args.model_name == "ResUnet":  # 残差卷积层
        net = ResUNet(3, args.num_class)

    elif args.model_name == "BNUnet":  # 瓶颈卷积层
        net = BottleNeckUNet(3, args.num_class)

    elif args.model_name == "AGUnet":  # 注意力门控
        net = AGUNet(3, args.num_class)

    elif args.model_name == "AGResUnet":  # 注意力门控 + Residual
        net = AGResUNet(3, args.num_class)

    elif args.model_name == "PAUnet":  # 位置注意力机制
        net = PAUNet(3, args.num_class)

    elif args.model_name == "CAUnet":  # 通道注意力机制
        net = CAUNet(3, args.num_class)

    elif args.model_name == "DAUnet":   # 双重注意力机制
        net = DAUNet(3, args.num_class)

    elif args.model_name == "DAResUnet":   # 双重注意力机制 + residual
        net = DAResUNet(3, args.num_class)

    elif args.model_name == "SDAGUnet":   # 串行双重注意力机制 + 注意力门控
        net = SDAGUNet(3, args.num_class)

    elif args.model_name == "SDAGResUnet":   # 串行双重注意力机制 + 注意力门控
        net = SDAGResUNet(3, args.num_class)

    else:
        print("\033[1;31;40m[ERROR]无效的模型! \033[0m")
        exit()
    return net


if __name__ == '__main__':
    model = ChooseModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model)
