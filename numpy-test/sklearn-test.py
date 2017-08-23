# -*- coding: utf-8 -*-
"""
sklearn learning
Transforming the prediction target (y)
Label binarization
One-hot Encoder
Model evaluation: quantifying the quality of predictions
Source: http://scikit-learn.org/stable/modules/model_evaluation.html
"""
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing  # 数据预处理模块调用
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef

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
# print(n_classes)
print(one_hot_mat.active_features_)
for i in one_hot_mat.active_features_:
    print("Sum of Class {0} : {1}".format(i, np.sum(n_target == i)))

# 验证输出新分类矩阵
# sample = np.array([2., 1., 1., 1., 2.])
sample = np.array([3., 4., 5., 1., 2., 1.])
sample_transform = lb.transform(sample)
one_hot_transform = enc.transform(sample.reshape(-1, 1)).toarray()

print("\nClass Sample:", sample)
print("\nLabelBinarizer Class Sample Transform:\n", sample_transform)
# print("\nSample Shape:", sample_transform.shape)
print("\nOneHotEncoder Class Sample Transform:\n", one_hot_transform)

# 几种模型评价指标测试
y_true = [1, 2, 1, 2, 3, 4, 1]
y_pred = [1, 1, 1, 3, 4, 1, 1]
# 混淆矩阵
con_mat = confusion_matrix(y_true, y_pred)
# ACC Score
acc = accuracy_score(y_true, y_pred)
print("\nConfusion Matrix:\n{0}".format(con_mat))
# MCC Score
# mcc = matthews_corrcoef(y_true, y_pred)

print("\nACC=", acc)

target_names = []
pred_names = []
for i in range(4):
    target_names.append('Class ' + str(i + 1))
    pred_names.append('Pred C' + str(i + 1))
# print(target_names, len(target_names))
con_df = pd.DataFrame(data=con_mat, index=target_names, columns=pred_names)
print("\nNew Confusion Matrix:\n{0}".format(con_df))
print("\nSum of each Class:\n", con_df.loc[:, ].sum())
# print(con_df.describe())
# 输出数据报告
print(classification_report(y_true, y_pred, target_names=target_names))
# print("\nMCC=", mcc)
