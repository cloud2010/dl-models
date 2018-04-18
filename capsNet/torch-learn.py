import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

FILE = "peptide_caps_net.csv"  # 数据集名称
DATASETS = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir, "datasets", FILE))  # 构建路径
# print(os.getcwd())

print("Datsets: ", DATASETS)

df = pd.read_csv(DATASETS)
labels = df.values[:, 0].astype(np.int32)  # 第一列为分类标注名称
features = df.values[:, 2:].astype(np.int32)

t_labels = torch.from_numpy(labels)
# 返回一个内存连续的有相同数据的tensor，否则无法做reshape
t_features = torch.from_numpy(features).contiguous().view(-1, 20, 20)

print(t_features.size())
print(t_labels.size())
# print(type(t_labels))
# print(type(t_features))
# print(t_labels[1:3])
