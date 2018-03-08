""" CSV Reader and data summary """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np

# 导入CSV数据
TRAIN_CSV = os.path.join(os.path.dirname(
    __file__), "../datasets/train_nitration_941_standard.csv")
# 去掉CSV文件标题行
train_set = np.genfromtxt(TRAIN_CSV, delimiter=',', skip_header=1)
# 分类矩阵为第一列数据
n_target = train_set[:, 0]
# 特征矩阵为去第一列之后数据
n_features = train_set[:, 1:]
# 样本数
nums_samples = n_target.shape[0]
# 特征数
nums_features = n_features.shape[1]

print("Sum of samples:", nums_samples)
print("Sum of vector length:", nums_features)

# 不同 Class 统计
for i in [1, 2]:
    print("Sum of Class {0}: {1}".format(i, np.sum(n_target == i)))
