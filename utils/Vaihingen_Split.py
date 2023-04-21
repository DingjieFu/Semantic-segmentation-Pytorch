#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:Vaihingen_Split.py
@Author:夜卜小魔王
@Date:2023/04/18 12:38:46
    Note:
    - 切分Vaihingen数据集 将.tif图像划分为多张256*256的小块
    - 由于数据量较小 加入了数据增广 共计得到 4360 张
'''
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T


def savePicture(idx, image, label):
    name = str(idx).rjust(4, "0")
    cv2.imwrite('dataset/Vaihingen/Images/' + name +
                '.jpg', image)  # 所有的RGB图像都放到jpg文件夹下
    cv2.imwrite('dataset/Vaihingen/ImagesPNG/' + name +
                '.png', label)  # 所有的标签图像都放到png文件夹下


transform_V = T.Compose([
    T.RandomVerticalFlip(1),  # 随机垂直翻转
])
transform_H = T.Compose([
    T.RandomHorizontalFlip(1),  # 随机水平翻转
])
transform_VH = T.Compose([
    T.RandomVerticalFlip(1),  # 随机垂直翻转
    T.RandomHorizontalFlip(1),  # 随机水平翻转
])


index = 0
for k in range(1, 40):
    image_path = './dataset/Vaihingen/image/top_mosaic_09cm_area' + \
        str(k) + '.tif'
    label_path = './dataset/Vaihingen/label/top_mosaic_09cm_area' + \
        str(k) + '.tif'
    if not os.path.exists(image_path):
        print("\033[1;31;40m[ERROR]路径{}不存在! \033[0m".format(image_path))
        continue
    img1 = cv2.imread(image_path)  # 读取RGB原图像
    img2 = cv2.imread(label_path)  # 读取Labels图像
    # cv2.imread函数会把图片读取为（B，G，R）顺序，一定要注意
    # cv2.imwrite函数最后会将通道调整回来，所以成对使用cv2.imread与cv2.imwrite不会改变通道顺序
    # Vaihingen数据集的图片尺寸不相同 将每一张图片分为很多256X256的小块 舍去边缘
    for i in range(10):
        for j in range(10):
            img1_ = img1[256*i: 256*(i+1), 256*j: 256*(j+1), :]
            img2_ = img2[256*i: 256*(i+1), 256*j: 256*(j+1), :]
            # 如果不是256*256*3 就舍去
            if img1_.size != 256*256*3:
                break

            index += 1
            savePicture(index, img1_, img2_)
            # index += 1
            # savePicture(index, np.array(transform_V(Image.fromarray(img1_))),
            #             np.array(transform_V(Image.fromarray(img2_))))
            # index += 1
            # savePicture(index, np.array(transform_H(Image.fromarray(img1_))),
            #             np.array(transform_H(Image.fromarray(img2_))))
            index += 1
            savePicture(index, np.array(transform_VH(Image.fromarray(img1_))),
                        np.array(transform_VH(Image.fromarray(img2_))))

print("Finish!")
