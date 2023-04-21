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

from parameter import parse_args
warnings.filterwarnings("ignore")
args = parse_args()  # load parameters


def ChooseModel():
    if args.model_name == "Unet":  # Unet模型
        net = UNet(3, args.num_class)
    else:
        print("\033[1;31;40m[ERROR]无效的模型! \033[0m")
        exit()
    return net


if __name__ == '__main__':
    model = ChooseModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(model)
