# -*- coding: utf-8 -*-
from __future__ import division
from config import cfg
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
from scipy.cluster.vq import vq

class Vocabulary:

    # 构建词袋模型词典，
    def __init__(self, sift_des, nclusters):
        # 将所有图片的sift特征点聚集成一个集合
        sift_sets = np.concatenate(sift_des)
        kmeans = KMeans(init='k-means++', n_clusters=nclusters, n_init=10)
        kmeans.fit(sift_sets)
        # 获得词典/n个聚类中心
        self.vocabulary = kmeans.cluster_centers_
        self.size = self.vocabulary.shape[0]

    # 将一副图片以及设定的金字塔层数level，生成最终的特征向量
    def buildHistogramForEachImageAtDifferentLevels(self, image_info, level):
        img_shape = image_info['image_shape']
        width = img_shape[1]
        height = img_shape[0]
        widthStep = np.ceil(width / 4)
        heightStep = np.ceil(height / 4)
        descriptors = image_info['image_des']
        keypoints = image_info['image_kps']

        # level2一共有4x4个网格，16个bins
        histogramOfLevelTwo = np.zeros((16, self.size))
        for index in xrange(len(descriptors)):
            # 获得特征点的(x,y)坐标，在kpt.pt里面
            x = int(keypoints[index].pt[0])
            y = int(keypoints[index].pt[1])
            # 对应16个bins的序号
            boundaryIndex = int(x / widthStep)  + int(y / heightStep) *4
            # 依次获取图片中每个sift特征点
            sift_feat = descriptors[index]
            # 进行词袋投放, vq的2个参数一定要2维列表或者矩阵，如果是一维需要外面加[]
            codes, distance = vq([sift_feat], self.vocabulary)
            # 对应bins的codes[0]号袋子计数加1
            histogramOfLevelTwo[boundaryIndex][codes[0]] += 1

        # level1一共2x2个网格，4个bins， 只要对level2每4个网格特征相加合并成一个大网格即可
        histogramOfLevelOne = np.zeros((4, self.size))
        histogramOfLevelOne[0] = histogramOfLevelTwo[0] + histogramOfLevelTwo[1] + histogramOfLevelTwo[4] + histogramOfLevelTwo[5]
        histogramOfLevelOne[1] = histogramOfLevelTwo[2] + histogramOfLevelTwo[3] + histogramOfLevelTwo[6] + histogramOfLevelTwo[7]
        histogramOfLevelOne[2] = histogramOfLevelTwo[8] + histogramOfLevelTwo[9] + histogramOfLevelTwo[12] + histogramOfLevelTwo[13]
        histogramOfLevelOne[3] = histogramOfLevelTwo[10] + histogramOfLevelTwo[11] + histogramOfLevelTwo[14] + histogramOfLevelTwo[15]

        # level0一共1x1个网格，1个bins，只要对level1的4个网格特征相加合并成一个大网格即可
        histogramOfLevelZero = histogramOfLevelOne[0] + histogramOfLevelOne[1] + histogramOfLevelOne[2] + histogramOfLevelOne[3]

        # 不同金字塔层次的bow特征加权拼接
        if level == 0:
            return histogramOfLevelZero
        elif level == 1:
            tempZero = histogramOfLevelZero.flatten() * 0.5
            tempOne = histogramOfLevelOne.flatten() * 0.5
            result = np.concatenate((tempZero, tempOne))
            return result
        elif level == 2:
            tempZero = histogramOfLevelZero.flatten() * 0.25
            tempOne = histogramOfLevelOne.flatten() * 0.25
            tempTwo = histogramOfLevelTwo.flatten() * 0.5
            result = np.concatenate((tempZero, tempOne, tempTwo))
            return result
        else:
            return None