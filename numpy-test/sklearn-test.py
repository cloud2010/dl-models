# -*- coding: utf-8 -*-
"""
sklearn learning
Transforming the prediction target (y)
Label binarization
"""
import os
import numpy as np
from sklearn import preprocessing  # 数据预处理模块调用

# 导入CSV数据
TRAIN_CSV = os.path.join(os.path.dirname(__file__), "d:\\train_nitration_941_standard.csv")
# TRAIN_CSV = os.path.join(os.path.dirname(__file__), "d:\\train.csv")
train_set = np.genfromtxt(
    TRAIN_CSV, delimiter=',', skip_header=1)

# 分类矩阵为第一列数据
n_target = train_set[:, 0]

# 特征矩阵
n_features = train_set[:, 1:]

# 原始分类矩阵型
# print("\nOriginal Target:")
# print(n_target)

# Class 二进制化模块
lb = preprocessing.LabelBinarizer(sparse_output=False)

# 原始分类转换（fit）
n_classes = lb.fit(n_target).classes_

print("\nClasses info:")
print(n_classes)

# 验证输出新分类矩阵
sample = np.array([1, 2, 2, 2, 1])
sample_transform = lb.transform(sample)

print("\nClass Sample:", sample)
print("\nClass Sample Transform:\n", sample_transform, sample_transform.shape)
# new_target = lb.transform(n_target)
# print("\nNew Target:\n", new_target)

# print("\nArgmax output:", np.argmax(sample_transform, axis=1))
