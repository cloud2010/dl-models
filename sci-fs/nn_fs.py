# -*- coding: utf-8 -*-
"""
The FNN with ReLU integrates the feature selection based on PyTorch for classification.
Date: 2020-06-24
"""
import os
import sys
import time
import torch
import torch.nn as nn
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


class NNs(nn.Module):
    r"""Fully connected neural network with 1 hidden layer and ReLU activation

    Args:
        input_size: size of each input sample
        hidden_size: size of hidden layer units
        dropout_rate: Hidden layer dropout rate
        num_classes: size of each output sample
    """

    def __init__(self, input_size, hidden_size, dropout_rate, num_classes):
        super(NNs, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dp1 = nn.Dropout(dropout_rate)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dp2 = nn.Dropout(dropout_rate)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dp1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.dp2(out)
        out = self.relu2(out)
        out = self.output(out)
        return out


def init_model(m):
    r"""Model weights and bias initialization.

    Args:
        m: nn model
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="The FNN with ReLU integrates the feature selection based on PyTorch for classification.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-u", "--nunits", type=int,
                        help="Number of hidden layer units.", default=1024)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=200)
    parser.add_argument("-d", "--dropout", type=float,
                        help="Hidden layer dropout rate.", default=5e-2)
    parser.add_argument("-l", "--learningrate", type=float,
                        help="Learning rate.", default=1e-2)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=88)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("-f", "--feature", type=str,
                        help="The feature selection algorithm.(f_classif, chi2, mutual_info_classif)", default="f_classif")

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from tqdm import tqdm, trange
    from sklearn.model_selection import KFold
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
    from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    # y = df.iloc[:, 0].map(lambda x: x if x == 1 else 0)  # 调整正负样本标注(1正样本/0负样本)
    f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)
    # 特征排序，根据函数名，获得相应算法函数对象
    fs = getattr(sys.modules[__name__], args.feature)
    # 特征排序
    model_fs = SelectKBest(fs, k="all").fit(X, y)
    # 降序排列特征权重
    fs_idxs = np.argsort(-model_fs.scores_)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Label向量读入GPU
    y_t = torch.from_numpy(y).long().to(device)
    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True,
               random_state=args.randomseed)
    # 评估数据保存列表
    mcc_nn_ifs = []
    acc_nn_ifs = []
    f1_nn_ifs = []
    print('Starting cross validating...\n')
    # 神经网络 IFS 曲线
    for ifs_i in trange(0, X.shape[1]):
        # 根据输入特征维度动态建立神经网络模型
        model = NNs(ifs_i+1, args.nunits, args.dropout, 2).to(device)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate)
        k_fold_step = 1  # 初始化折数
        # 生成 k-fold 训练集、测试集索引
        cv_index_set = rs.split(y)
        # 暂存每次选中的测试集和对应预测结果
        test_cache = pred_cache = np.array([], dtype=np.int)
        # 初始化模型参数
        model.apply(init_model)
        # IFS 增量放入特征子集
        X_t = torch.from_numpy(X[:, fs_idxs[0:ifs_i+1]]).float().to(device)
        # 迭代训练 k-fold 交叉验证
        for train_index, test_index in cv_index_set:
            # print("\nFold:", k_fold_step)
            # 每个fold之初进行参数初始化
            model.apply(init_model)
            # Start Training
            for epoch in trange(args.epochs):
                # Forward pass
                outputs = model(X_t[train_index])
                _, pred = outputs.max(1)
                loss = criterion(outputs, y_t[train_index])
                correct = (pred == y_t[train_index]).sum().item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (epoch+1) % 100 == 0:
                #    print('Epoch [{}/{}], Train size: {}, Acc: {:.4%}, Loss: {:.6f}'.format(
                #        epoch+1, num_epochs, train_index.size, correct/train_index.size, loss.item()))
            # 测试集验证
            # In test phase, we don't need to compute gradients (for memory efficiency)
            with torch.no_grad():
                test_outputs = model(X_t[test_index])
                _, y_pred = test_outputs.max(1)
                # ensure copy the tensor from cuda to host memory
                y_pred = y_pred.cpu()
                # 计算测试集 ACC
                accTest = accuracy_score(y[test_index], y_pred.data.numpy())
                # print("\nFold:", k_fold_step, "Test Accuracy:",
                #      "{:.4%}".format(accTest), "Test Size:", test_index.size)

            # 暂存每次选中的测试集和预测结果
            test_cache = np.concatenate((test_cache, y[test_index]))
            pred_cache = np.concatenate((pred_cache, y_pred.data.numpy()))
            # print("\n=========================================================================")
            # 每个fold训练结束后次数 +1
            k_fold_step += 1
        mcc_cache = matthews_corrcoef(test_cache, pred_cache)
        acc_cache = accuracy_score(test_cache, pred_cache)
        f1_cache = precision_recall_fscore_support(
            test_cache, pred_cache, average='binary')
        acc_nn_ifs.append(acc_cache)
        mcc_nn_ifs.append(mcc_cache)
        f1_nn_ifs.append(f1_cache)
        # print('F{0} => ACC={1:.4f}, MCC={2:.4f}'.format(ifs_i, acc_cache, mcc_cache))

    # 测试输出正确性
    print('\nACC Max after FS:', np.argmax(acc_nn_ifs), np.max(acc_nn_ifs))
    print('\nMCC Max after FS:', np.argmax(mcc_nn_ifs), np.max(mcc_nn_ifs))

    # 导出评估指标数据到 CSV
    df_fs = pd.DataFrame(data=list(zip(acc_nn_ifs, mcc_nn_ifs)), columns=[
                         'acc_fs', 'mcc_fs'])
    df_f1 = pd.DataFrame(data=f1_nn_ifs, columns=[
                         'precision', 'recall', 'f-score', 'support'])
    # 合并指标输出
    df_fs = pd.concat([df_fs, df_f1.iloc[:, 0:3]], axis=1)
    # 提取文件名
    filepath = os.path.basename(args.datapath).split('.')[0]
    # 写入 CSV
    df_fs.to_csv('{0}_{1}_{2}.csv'.format('nn', args.feature, filepath))

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
