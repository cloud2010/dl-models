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
# TRAIN_CSV = os.path.join(os.path.dirname(__file__), "d:\\datasets\\train_nitration_941_standard.csv")
TRAIN_CSV = os.path.join(os.path.dirname(__file__), "d:\\datasets\\train_multi_class.csv")
train_set = np.genfromtxt(TRAIN_CSV, delimiter=',', skip_header=1)

# 分类矩阵为第一列数据
n_target = train_set[:, 0]

# 特征矩阵
n_features = train_set[:, 1:]

# 原始分类矩阵型
print("\nOriginal Target:")
print(n_target)

# Class 二进制化模块
lb = preprocessing.LabelBinarizer()
# One-hot 编码
enc = preprocessing.OneHotEncoder(sparse=True, dtype=np.int8)

# LabelBinarizer 转换
n_classes = lb.fit(n_target).classes_
# One-hot 转换
one_hot_mat = enc.fit(n_target.reshape(-1, 1))

print("\nClasses info:")
print(n_classes)
# print(one_hot_mat.active_features_)

# 验证输出新分类矩阵
# sample = np.array([2., 1., 1., 1., 2.])
sample = np.array([3., 4., 5., 1., 2., 1.])
sample_transform = lb.transform(sample)
one_hot_transform = enc.transform(sample.reshape(-1, 1)).toarray()

print("\nClass Sample:", sample)
print("\nLabelBinarizer Class Sample Transform:\n", sample_transform)
# print("\nSample Shape:", sample_transform.shape)
print("\nOneHotEncoder Class Sample Transform:\n", one_hot_transform)
# new_target = lb.transform(n_target)
# print("\nNew Target:\n", new_target)

# print("\nArgmax output:", np.argmax(sample_transform, axis=1))
