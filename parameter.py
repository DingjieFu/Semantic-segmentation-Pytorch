#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:parameter.py
@Author:夜卜小魔王
@Date:2023/02/25 19:20:39
    Note:
        定义了一些参数
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='GraduationDesign')
    model = "DAUnet"
    parser.add_argument('--model_name',       default=model,
                        type=str,     help='Model name')
    parser.add_argument('--root_path',        default='dataset/WHDLD',
                        type=str,     help='dataset root path')
    parser.add_argument('--num_class',        default=6,
                        type=int,     help='number of total classes')
    parser.add_argument('--seed',             default=2023,
                        type=int,     help='seed for reproducibility')
    parser.add_argument('--batch_size',       default=32,
                        type=int,     help='batchsize for optimizer updates')
    parser.add_argument('--num_epoch',        default=200,
                        type=int,     help='number of total epochs to run')
    parser.add_argument('--lr',               default=3e-4,
                        type=float,   help='initial learning rate')
    parser.add_argument('--log',              default='out/{}/'.format(model),
                        type=str,     help='Log result file name')
    parser.add_argument('--model_path',              default='out/{}/best_model.pth'.format(model),
                        type=str,     help='model weight path')
    parser.add_argument('--score_train',              default='out/{}/best_train.json'.format(model),
                        type=str,     help='model val score')
    parser.add_argument('--score_test',              default='out/{}/best_test.json'.format(model),
                        type=str,     help='model val score')
    args = parser.parse_args()
    return args
