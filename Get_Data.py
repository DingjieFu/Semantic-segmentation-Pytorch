#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:Get_Data.py
@Author:夜卜小魔王
@Date:2023/03/16 09:34:14
    Note:
        运行一次即可 会在dataset中生成相应的.txt文件
'''


import os
import random
import numpy as np
from parameter import parse_args
args = parse_args()


def write_imgname_to_txtfile(imgdir, txtdir):
    """
        write_imgname_to_txtfile():将指定文件夹下的所有文件名写入指定的txt文件中。
        imgdir:包含图像文件的文件夹路径。
        txtdir:需要写入的txt文件路径。 
    """
    f = open(txtdir, 'w')
    for file in os.listdir(imgdir):
        file = os.path.join(args.root_path, "Images") + "/{}".format(file)
        f.write("{}\n".format(file))
    f.close()
    print("write to txt finished...")


def load_filename_to_list(dir):
    """
        load_filename_to_list():从文本文件txt中读取每行数据,组成列表返回
        dir:文本文件txt的路径
    """
    img_name_list = np.loadtxt(dir, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def step1():
    """
        step1(): 将存储在指定文件夹下的所有图像名称写入指定txt文件
        imgdir: 存放所有图像的路径
        txtdir: txt文件的输出路径
    """
    imgdir = args.root_path + "/Images"
    txtdir = args.root_path + "/total_list.txt"
    write_imgname_to_txtfile(imgdir, txtdir)


def step2():
    """
        step2(): 将step1中存储的txt文件中的所有图像明细,按照6:2:2的比例划分为训练集,验证集和测试集
        imgdir: 存放所有图像的路径
        txtdir: txt文件的输出路径
    """
    dir = args.root_path + "/total_list.txt"
    totallist = load_filename_to_list(dir)
    n = totallist.size
    m = int(n * 0.4)
    print("n is:%s, m is: %s" % (n, m))
    print(type(totallist))
    split1 = set(random.sample(list(totallist), m))
    split1_1 = set(random.sample(list(split1), m // 2))

    train = set(totallist) - split1
    val = split1 - split1_1
    test = split1_1

    traindir = r"dataset\WHDLD\train_list.txt"
    valdir = r"dataset\WHDLD\val_list.txt"
    testdir = r"dataset\WHDLD\test_list.txt"

    f = open(traindir, 'w')
    for file in train:
        f.write("{}\n".format(file))
    f.close()
    print("write to traintxt finished...")

    f = open(valdir, 'w')
    for file in val:
        f.write("{}\n".format(file))
    f.close()
    print("write to valtxt finished...")

    f = open(testdir, 'w')
    for file in test:
        f.write("{}\n".format(file))
    f.close()
    print("write to testtxt finished...")


if __name__ == "__main__":
    step1()
    step2()
