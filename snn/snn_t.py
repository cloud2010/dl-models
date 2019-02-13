# -*- coding: utf-8 -*-
"""
Pytorch Implementation of the Scaled ELU function and Alpha Dropout.

Author: Liu Min
Date: 2019-1-27
Github: `https://github.com/cloud2010/dl-models`
"""
# import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score  # 计算 ACC
# from sklearn.metrics import roc_curve, roc_auc_score  # 计算 ROC
from .utils import model_evaluation, bi_model_evaluation


class SNNs(nn.Module):
    """Fully connected neural network with 2 hidden layer and SELU activation

    Args:
        input_size: size of each input sample
        hidden_size: size of hidden layer units
        num_classes: size of each output sample
        dropout_rate: Hidden layer dropout rate
    """

    def __init__(self, input_size, hidden_size, dropout_rate, num_classes):
        super(SNNs, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Alpha Dropout is a type of Dropout that maintains the self-normalizing property.
        # Ref: `https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.AlphaDropout`
        self.dp1 = nn.AlphaDropout(dropout_rate)
        self.selu1 = nn.SELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dp2 = nn.AlphaDropout(dropout_rate)
        self.selu2 = nn.SELU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dp3 = nn.AlphaDropout(dropout_rate)
        self.selu3 = nn.SELU()
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dp1(out)
        out = self.selu1(out)
        out = self.fc2(out)
        out = self.dp2(out)
        out = self.selu2(out)
        out = self.fc3(out)
        out = self.dp3(out)
        out = self.selu3(out)
        out = self.output(out)
        return out


def init_model(m):
    """Model weights and bias initialization.
    """
    # print(m)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def run(inputFile, h_units, epochs, folds, l_rate, d_rate, w_rate, random_s):
    """RNN主程序
    参数
    ----
    inputFile: 训练集文件路径
    h_units: 隐藏层单元数
    epochs: 每个fold的训练次数
    folds: k-fold折数
    l_rate: Learning rate
    d_rate: Dropout rate
    w_rate: weight decay (L2 penalty) (default: 0)
    random_s: 随机种子
    """
    try:
        # 导入CSV数据
        df = pd.read_csv(inputFile, encoding='utf8')
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)

    # 特征矩阵为去第一列之后数据
    X = df.iloc[:, 1:].values
    # 分类矩阵为第一列数据 (labels值减1)
    y = df.iloc[:, 0].values - 1
    # 读取特征名称
    features = df.columns[1:]
    f_size = features.size
    print("\nDataset shape: ", df.shape, " Number of features: ", f_size)
    # 不同 Class 统计 (根据 Target 列)
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Sets the seed for generating random numbers to all devices
    torch.manual_seed(random_s)
    print('\nTraining Device:', device)

    # numpy dataset trasform to Tensor
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)

    # Create SNN Neural Net (Depth: 2)
    model = SNNs(f_size, h_units, d_rate, num_categories).to(device)
    print("\nModel Parameters:", model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # add the weights and bias with L2 weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=w_rate)
    print("\nOptimizer Parameters:", optimizer)

    # 交叉验证
    rs = KFold(n_splits=folds, shuffle=True, random_state=random_s)
    # 生成 k-fold 训练集、测试集索引
    cv_index_set = rs.split(y)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in cv_index_set:
        print("\nFold:", k_fold_step)
        # 每个 fold 模型参数初始化
        model.apply(init_model)
        # Start Training
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X_t[train_index])
            _, pred = outputs.max(1)
            loss = criterion(outputs, y_t[train_index])
            correct = (pred == y_t[train_index]).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                print('Epoch [{}/{}], Train size: {}, Acc: {:.4%}, Loss: {:.6f}'.format(
                    epoch+1, epochs, train_index.size, correct/train_index.size, loss.item()))
        # 测试集验证
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            test_outputs = model(X_t[test_index])
            _, y_pred = test_outputs.max(1)
            # ensure copy the tensor from cuda to host memory
            y_pred = y_pred.cpu()
            # 计算测试集 ACC
            accTest = accuracy_score(y[test_index], y_pred.data.numpy())
            print("\nFold:", k_fold_step, "Test Accuracy:",
                  "{:.4%}".format(accTest), "Test Size:", test_index.size)

        # 暂存每次选中的测试集和预测结果
        test_cache = np.concatenate((test_cache, y[test_index]))
        pred_cache = np.concatenate((pred_cache, y_pred.data.numpy()))
        print("\n=========================================================================")
        # 每个fold训练结束后次数 +1
        k_fold_step += 1

    # 输出统计结果
    if(num_categories > 2):
        model_evaluation(num_categories, test_cache, pred_cache)
    else:
        bi_model_evaluation(test_cache, pred_cache)
