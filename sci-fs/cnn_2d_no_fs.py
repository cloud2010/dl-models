# -*- coding: utf-8 -*-
"""
The 2D-CNN without the feature selection based on PyTorch for classification.
Date: 2020-07-17
"""
import os
import sys
import time
import torch
import torch.nn as nn
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support, confusion_matrix, classification_report

__author__ = 'Min'


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    r"""2层二维卷积神经网络模型(输入：W*H，输出：2)
        2层卷积层+2层批归一化+1层Dropout+1层池化+1层全连接

    Args:
        f_size: width or height of the feature matrix
        dropout_rate: hidden layer dropout rate
        num_classes: size of each output sample
    """

    def __init__(self, f_size, dropout_rate, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # 第1层16个卷积核，核大小3*3，步长1，有效填充不补0，输出大小 W-3+1
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 第1层池化，步长2，输出长宽压缩一半
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # 第1层 Dropout，随机丢失5%
            nn.Dropout2d(0.05))
        self.layer2 = nn.Sequential(
            # 第2层32个卷积核，核大小3*3，步长1，有效填充不补0，输出大小 W-3+1
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 第2层池化，步长2，输出长宽压缩一半
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten())
        # 全连接层，输入大小例如 (22-3+1)/2 -> (10-3+1)/2 -> 4
        # input_size_w =np.int(((f_size-3+1)/2-3+1)/2)
        input_size_w = np.int(((f_size-3+1)-3+1)/2)
        self.fc = nn.Linear(input_size_w*input_size_w*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def init_model(m):
    r"""Model weights and bias initialization.

    Args:
        m: Conv2d or Linear model
    """
    if (type(m) == nn.Conv2d) or (type(m) == nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def bi_model_evaluation(y_true, y_pred):
    """二分类问题每个fold测试结束后计算Precision、Recall、ACC、MCC等统计指标

    Args:
    num_classes : 分类数
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    """
    class_names = ["Negative", "Positive"]
    pred_names = ["Pred Negative", "Pred Positive"]

    # 混淆矩阵生成
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(data=cm, index=class_names, columns=pred_names)

    # 混淆矩阵添加一列代表各类求和
    df['Sum'] = df.sum(axis=1).values

    print("\n=== Model evaluation ===")
    print("\n=== Accuracy classification score ===")
    print("\nACC = {:.6f}".format(accuracy_score(y_true, y_pred)))
    print("\n=== Matthews Correlation Coefficient ===")
    print("\nMCC = {:.6f}".format(matthews_corrcoef(y_true, y_pred)))
    print("\n=== Confusion Matrix ===\n")
    print(df.to_string())
    print("\n=== Detailed Accuracy By Class ===\n")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, digits=6))


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="The 2D-CNN without the feature selection based on PyTorch for classification.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--fsize", type=int,
                        help="Width or height of the feature matrix.", default=30)
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

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from tqdm import tqdm, trange
    from sklearn.model_selection import KFold

    torch.manual_seed(args.randomseed)
    np.random.seed(args.randomseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.randomseed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Label向量读入GPU
    y_t = torch.from_numpy(y).long().to(device)
    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True,
               random_state=args.randomseed)

    print('Starting cross validating...\n')

    # 根据输入特征维度动态建立神经网络模型
    model = ConvNet(args.fsize, args.dropout, num_classes=2).to(device)
    # Loss and optimizer
    # nn.logSoftmax()和nn.NLLLoss()的整合
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate)

    k_fold_step = 1  # 初始化折数
    # 生成 k-fold 训练集、测试集索引
    cv_index_set = rs.split(y)
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    # 输出模型参数
    print("\nModel Parameters:", model)
    print("\nOptimizer Parameters:", optimizer)
    # 本例面向2维卷积神经网络，默认取前484维(22^2 or 22*22)
    f_nums = np.square(args.fsize)
    X = X[:, 0:f_nums].reshape(len(y), 1, args.fsize, args.fsize)
    X_t = torch.from_numpy(X).float().to(device)
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in cv_index_set:
        print("\nFold:", k_fold_step)
        # 每个fold之初进行参数初始化
        model.apply(init_model)
        # Start Training
        for epoch in range(args.epochs):
            # 前向传递
            outputs = model(X_t[train_index])
            _, pred = outputs.max(1)
            # 计算误差
            loss = criterion(outputs, y_t[train_index])
            correct = (pred == y_t[train_index]).sum().item()
            # 清空上一次梯度
            optimizer.zero_grad()
            # 误差反向传递
            loss.backward()
            # 优化器参数更新
            optimizer.step()
            if (epoch+1) % 50 == 0:
                print('Epoch [{}/{}], Train size: {}, Acc: {:.4%}, Loss: {:.6f}'.format(
                    epoch+1, args.epochs, train_index.size, correct/train_index.size, loss.item()))
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
    bi_model_evaluation(test_cache, pred_cache)

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
