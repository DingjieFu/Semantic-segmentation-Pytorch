#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File:utils.py
@Author:夜卜小魔王
@Date:2023/02/26 10:00:31
    Note:
        此文件为一些评价指标
        - OA
        - AA
        - mIoU
        - FWIoU
'''
import numpy as np


class SegmentationMetric(object):
    def __init__(self, numClass):
        """
            - numClass: 类别总数6
            - confusionMatrix: (6,6)矩阵
        """
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, imgPredict, imgLabel):
        """
            计算混淆矩阵
            - imgPredict: 经过model输出的label; imgLabel: 原来的label经过img2label
            此处横着代表预测值,竖着代表真实值
            - confusionMetric:
            - P\L     P    N
            - P      TP    FP
            - N      FN    TN
        """
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def pixelAccuracy(self):
        """
            准确率(Accuracy): 预测结果中正确的占总预测值的比例(对角线元素和/混淆矩阵元素总和)
            all class overall pixel accuracy -> PA/OA
            - Accuracy = (TP + TN) / (TP + TN + FP + TN)

            这里返回需要的 总体正确率(OA)
        """
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def meanPixelAccuracy(self):
        """
            精确率(Precision): 预测结果中,某类别预测正确的概率
            - Precision = TP / (TP + FP)

            这里返回需要的 平均正确率(AA)
        """
        # 返回一个list, 如[0.90, 0.80, 0.96], 表示类别1 2 3各类别的预测精确率
        classAcc = np.diag(self.confusionMatrix) / \
            self.confusionMatrix.sum(axis=1)
        # 返回值, 如np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96) / 3 =  0.89
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值, nan表示遇到Nan类型, 其值取为0
        return meanAcc

    def meanIntersectionOverUnion(self):
        """
            交并比: 某一类别预测结果和真实值的交集与并集的比值
            - IoU = TP / (TP + FP + FN)

            这里返回需要的 平均交并比mIoU
        """
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        # axis = 1表示混淆矩阵行的值, 返回列表; axis = 0表示取混淆矩阵列的值,返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - \
            np.diag(self.confusionMatrix)
        IoU = intersection / union  # 返回列表,其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        """
            加权交并比: 根据每个类出现的频率设置权重
            - FWIoU =   [(TP+FN)/(TP+FP+TN+FN)] * [TP/(TP+FP+FN)]
        """
        freq = np.sum(self.confusionMatrix, axis=1) / \
            np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
            np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
            np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        """
            对每一个batch进行一次这个操作,混淆矩阵加和
        """
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        """
            reset操作,将混淆矩阵置零
        """
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
