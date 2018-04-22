# -*- coding: utf-8 -*-
"""
A PyTorch implementation of CapsNet for peptide datasets
Author: liumin@shmtu.edu.cn
"""
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os
import torch
import torch.utils.data as Data

FILE = "peptide_caps_net.csv"  # 数据集名称
DATASETS = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, "datasets", FILE))  # 构建路径
# print(os.getcwd())

print("Data Name:", DATASETS)

df = pd.read_csv(DATASETS)
# 第1列为分类标注名称
labels = df.values[:, 0].astype(np.int32)
# 调整 label 2为1(N样本), 1为0(P样本), 便于转换为 One-hot Vector
labels[labels == 1] = 0
labels[labels == 2] = 1
t_labels = torch.from_numpy(labels).view(-1, 1)
print(t_labels)
# 转换为 One-hot Vector
t_onehot = torch.zeros(labels.size, 2)
t_onehot.scatter_(1, t_labels.type(torch.LongTensor), 1)
print(t_onehot)

# 第3列之后为特征矩阵
features = df.values[:, 2:].astype(np.int32)
"""
返回一个内存连续的有相同数据的tensor，否则无法做reshape
 -Input:  (batch, channels, width, height), optional (batch, classes)
 -Output: ((batch, classes), (batch, channels, width, height))
"""
t_features = torch.from_numpy(features).contiguous().view(-1, 1, 20, 20)  # 4-D Tensor

# print(t_features)
# print(t_labels)
# print(type(t_labels))
# print(type(t_features))
# print(t_features[1])

# Using PyTorch DataLoader
# 构建 pytorch 数据集
t_dataset = Data.TensorDataset(data_tensor=t_features, target_tensor=t_labels)
# remove num_workers settings
t_loader = Data.DataLoader(dataset=t_dataset, shuffle=True, batch_size=1000)

# 迭代输出DataLoader相关信息
# for epoch in range(2):
#     for step, (batch_x, batch_y) in enumerate(t_loader):
#         # print('Epoch: ', epoch, '| Step: ', step, '| Batch x: ', batch_x.numpy(), '| Batch y: ', batch_y.numpy())
#         print('Epoch: ', epoch, '| Step: ', step, '| Batch y: ', batch_y.numpy())
