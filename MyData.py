#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:MyData.py
@Author:夜卜小魔王
@Date:2023/03/07 09:36:56
    Note:
        将划分好的数据 以(data,label)的形式返回
        data: 原图像 转化为tensor格式
        label: 已经将原本的标注图变为0-5标签图 转为tensor格式
'''

import os
import torch
import warnings
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from parameter import parse_args
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
args = parse_args()


# ---------- WHDLD ----------
bare_soil = [128, 128, 128]
building = [255, 0, 0]
pavement = [192, 192, 0]
road = [255, 255, 0]
vegetation = [0, 255, 0]
water = [0, 0, 255]
colorMap_WHDLD = [bare_soil, building, pavement, road, vegetation, water]

# ---------- Vaihingen ----------
impervious_surface = [255, 255, 255]
structure = [0, 0, 255]
low_vegetation = [0, 255, 255]
tree = [0, 255, 0]
car = [255, 255, 0]
background = [255, 0, 0]
colorMap_Vaihingen = [impervious_surface, structure,
                      low_vegetation, tree, car, background]

Map_dictionary = {"WHDLD": colorMap_WHDLD, "Vaihingen": colorMap_Vaihingen}

color_map = Map_dictionary[args.dataset_name]


class MyDataset(Dataset):
    def __init__(self, root, mode):
        """
            通过mode来选择载入的数据:
                mode == "Train" : 加载训练集
                mode == "Val"   : 加载验证集
                mode == "Test"  : 加载测试集
                others : 回复错误
        """
        self.root = root
        self.label_path = os.path.join(root, "ImagesPNG")
        train_path_list = np.loadtxt(
            os.path.join(root, "train_list.txt"), dtype=str)
        val_path_list = np.loadtxt(
            os.path.join(root, "val_list.txt"), dtype=str)
        test_path_list = np.loadtxt(
            os.path.join(root, "test_list.txt"), dtype=str)

        if mode == "Train":
            self.path_list = train_path_list
        elif mode == "Val":
            self.path_list = val_path_list
        elif mode == "Test":
            self.path_list = test_path_list
        else:
            print("\033[1;31;40m[ERROR]无效的模式输入! \033[0m")
            exit()

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index: int):
        # 得到图像
        img_path = self.path_list[index]
        image = Image.open(img_path).convert("RGB")
        data = self.transforms(image)
        # 得到标签
        (_, filename) = os.path.split(img_path)
        filename = filename.split('.')[0]
        label = Image.open(self.label_path + "/"+filename +
                           ".png").convert("RGB")

        label = image2label()(label)

        label = torch.from_numpy(label).long()
        return data, label

    def __len__(self):
        return len(self.path_list)


class image2label():
    def __init__(self):
        """
            根据color_map,对每一种种类的颜色赋上0-5的标签
        """
        self.color_map = color_map
        # 创建256^3 次方空数组，颜色的所有组合
        cm2lb = np.zeros(256 ** 3)
        for i, cm in enumerate(self.color_map):
            cm2lb[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 符合这种组合的标记这一类
        self.cm2lb = cm2lb

    def __call__(self, image):
        """
            对ImagesPNG中的标注图片,将其标注也改为0-5的标签
        """
        image = np.array(image, dtype=np.int64)
        idx = (image[:, :, 0] * 256 + image[:, :, 1]) * 256 + image[:, :, 2]
        label = np.array(self.cm2lb[idx], dtype=np.int64)  # 根据颜色条找到这个label的标号
        return label


class label2image():
    def __init__(self):
        """
            self.colormap -> (6,3)
        """
        self.colormap = np.array(color_map).astype(np.uint8)

    def __call__(self, label):
        """
            逆操作,将0-5的标签还原为原来的三通道色彩
        """
        pred = self.colormap[label]
        return pred


if __name__ == "__main__":
    root = args.root_path
    train_data = MyDataset(root, "Train")
    val_data = MyDataset(root, "Val")
    test_data = MyDataset(root, "Test")
    train_dataloader = DataLoader(train_data, batch_size=8)
    val_dataloader = DataLoader(val_data, batch_size=8)
    test_dataloader = DataLoader(test_data, batch_size=8)
    print(len(train_data), len(train_dataloader))
    print(len(val_data), len(val_dataloader))
    print(len(test_data), len(test_dataloader))

    test_image, test_label = test_data[0]
    temp = test_image.numpy()
    temp = (temp-np.min(temp)) / (np.max(temp)-np.min(temp))*255
    true_label = label2image()(test_label)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(temp.transpose(1, 2, 0).astype("uint8"))
    ax[1].imshow(test_label)
    ax[2].imshow(true_label)
    ax[0].set_title('Image', fontsize=10)
    ax[1].set_title('i2l_label', fontsize=10)
    ax[2].set_title('true_label', fontsize=10)
    plt.show()
