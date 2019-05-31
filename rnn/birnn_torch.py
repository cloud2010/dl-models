# -*- coding: utf-8 -*-
"""
A Bidirectional Recurrent Neural Network with SMOTE implementation based on Pytorch and scikit-learn library.
Author: liumin@shmtu.edu.cn
Date: 2019-05-30
Tested under: Python 3.6 / Pytorch 1.1+
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score  # 计算 ACC
from sklearn.model_selection import KFold

from .utils import model_evaluation, bi_model_evaluation

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiRNN(nn.Module):
    """Bidirectional recurrent neural network (many-to-one)
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def init_model(m):
    """Model weights and bias initialization.
    """
    if type(m) == nn.LSTM:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def run(input_file, h_units, s_length, n_layers, l_rate, epochs, folds, random_s):
    """Bi-RNN主程序
    参数
    ----
    input_file: 训练集文件路径
    h_units: 隐藏层单元数
    s_length: sequence 长度
    n_layers: 隐藏层数
    l_rate: Learning rate
    epochs: 每个fold的训练次数
    folds: k-fold折数
    random_s: 随机种子
    """
    try:
        # 导入CSV数据
        df = pd.read_csv(input_file, encoding='utf8')
    except OSError as e:
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

    # Sets the seed for generating random numbers to all devices
    torch.manual_seed(random_s)
    print('\nTraining Device:', device)

    # numpy dataset transform to Tensor
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)

    # Create Bi-RNN
    model = BiRNN(f_size, h_units, n_layers, num_categories)
    print("\nModel Parameters:", model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

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
            x_train = X_t[train_index].reshape(-1, s_length, f_size).to(device)
            
            # Forward pass
            outputs = model(x_train)
            _, pred = outputs.max(1)
            loss = criterion(outputs, y_t[train_index])
            correct = (pred == y_t[train_index]).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print('Epoch [{}/{}], Train size: {}, Acc: {:.4%}, Loss: {:.6f}'.format(
                    epoch + 1, epochs, train_index.size, correct / train_index.size, loss.item()))

        # 测试集验证
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            x_test = X_t[test_index].reshape(-1, s_length, f_size).to(device)
            test_outputs = model(x_test)
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
    if num_categories > 2:
        model_evaluation(num_categories, test_cache, pred_cache)
    else:
        bi_model_evaluation(test_cache, pred_cache)
