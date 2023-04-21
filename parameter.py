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
    model = "Unet"
    dataset = "WHDLD"
    parser.add_argument('--model_name',       default=model,
                        type=str,     help='Model name')
    parser.add_argument('--dataset_name',     default=dataset,
                        type=str,     help='dataset name')
    parser.add_argument('--root_path',        default='dataset/{}'.format(dataset),
                        type=str,     help='dataset root path')
    parser.add_argument('--num_class',        default=6,
                        type=int,     help='number of total classes')
    parser.add_argument('--seed',             default=2023,
                        type=int,     help='seed for reproducibility')
    parser.add_argument('--batch_size',       default=4,
                        type=int,     help='batchsize for optimizer updates')
    parser.add_argument('--num_epoch',        default=200,
                        type=int,     help='number of total epochs to run')
    parser.add_argument('--lr',               default=3e-4,
                        type=float,   help='initial learning rate')
    parser.add_argument('--log',              default='out/{}_{}/'.format(model, dataset),
                        type=str,     help='Log result file name')
    parser.add_argument('--model_path',              default='out/{}_{}/best_model.pth'.format(model, dataset),
                        type=str,     help='model weight path')
    parser.add_argument('--score_train',              default='out/{}_{}/best_train.json'.format(model, dataset),
                        type=str,     help='model val score')
    parser.add_argument('--score_test',              default='out/{}_{}/best_test.json'.format(model, dataset),
                        type=str,     help='model val score')
    args = parser.parse_args()
    return args
