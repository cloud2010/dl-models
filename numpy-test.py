"""
Numpy learning
"""
import numpy as np
import os
# 导入CSV数据
TRAIN_NITRATION = os.path.join(os.path.dirname(
    __file__), "d:\\train_nitration_941_standard.csv")
train_nitration_set = np.genfromtxt(
    TRAIN_NITRATION, delimiter=',', skip_header=1)

# 分类矩阵为第一列数据
n_target = train_nitration_set[:, 0]

# 特征矩阵 212*941
n_features = train_nitration_set[:, 1:]

# 输出两个矩阵的Shape
print("Target Shape: ", n_target.shape)
print("Features Shape: ", n_features.shape)

# 随机放回抽样
instance_num = np.arange(1, n_target.size + 1)
# 随机生成抽样行
sample_row = np.random.choice(instance_num, size=10, replace=True)

# 输出抽样行矩阵切片
sample_set = np.empty(shape=[942, ])
print(sample_set.shape)
# print(sample_set)
for i in sample_row:
    # print(train_nitration_set[i, :])
    # print(train_nitration_set[i, :].shape)
    # print(n_target[i])
    sample_set = np.vstack((sample_set, train_nitration_set[i, :]))

# 剔除第一行初始值
sample_set = np.delete(sample_set, 0, 0)
print(sample_set.shape)
print(sample_set)
