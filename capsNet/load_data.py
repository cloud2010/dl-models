# -*- coding: utf-8 -*-
"""
A Tensorflow implementation of CapsNet for bio datasets
Author: liumin@shmtu.edu.cn
Date: 2018-04-07
Tested under: Python 3.5 / Python 3.6 and TensorFlow 1.6+
Reference: https://github.com/naturomics/CapsNet-Tensorflow
"""
import numpy as np
import pandas as pd

DATASETS = "d://datasets//peptide_caps_net.csv"  # 数据集地址
df = pd.read_csv(DATASETS)  # 读取数据

labels = df.values[:, 0]  # 第一列为分类标注名称
features = df.values[:, 2:].reshape((labels.size, 20, 20))  # 第二列之后为特征矩阵（完成 reshape）

print(features.shape)
