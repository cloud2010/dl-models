# -*- coding: utf-8 -*-
"""
This program implements Quadratic Discriminant Analysis classifier for predicting biological sequence datasets.
Date: 2024-04-11
Author: Min
"""
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import pandas as pd
from lgb.utils import model_evaluation, bi_model_evaluation


def parse_args():
    """
    解析命令行参数。

    参数:
    无

    返回:
    返回解析后的命令行参数，包括：
    - kfolds: 折叠的数量，必须大于等于2，默认为10。
    - randomseed: 用于洗牌的伪随机数生成器状态，默认为0。
    - knighbors: 用于构建合成样本的最近邻数量，默认为3。
    - datapath: 数据集的路径，此参数为必填项。
    """

    # 初始化命令行参数解析器
    parser = ArgumentParser(
        description="This program implements Quadratic Discriminant Analysis classifier for predicting biological sequence datasets.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    # 添加-k或--kfolds参数，用于指定折叠数量
    parser.add_argument(
        "-k",
        "--kfolds",
        type=int,
        help="Number of folds. Must be at least 2.",
        default=10,
    )
    # 添加-r或--randomseed参数，用于指定随机种子
    parser.add_argument(
        "-r",
        "--randomseed",
        type=int,
        help="pseudo-random number generator state used for shuffling.",
        default=0,
    )
    # 添加-n或--kneighbors参数，用于指定最近邻数量
    parser.add_argument(
        "-n",
        "--kneighbors",
        type=int,
        help="number of nearest neighbours to used to construct synthetic samples.",
        default=3,
    )
    # 添加--datapath参数，用于指定数据集路径，此参数为必填项
    parser.add_argument(
        "--datapath", type=str, help="The path of dataset.", required=True
    )

    # 解析并返回命令行参数
    return parser.parse_args()


def load_and_prepare_data(args):
    """
    加载数据并进行必要的预处理。

    参数:
    - args: 一个包含数据路径(datapath)的命名空间对象。

    返回值:
    - X: 数据集的特征矩阵，去除第一列。
    - y: 数据集的目标值，索引从 0 开始。
    """
    try:
        df = pd.read_csv(args.datapath)  # 尝试从指定路径读取数据集
    except Exception as e:
        print(f"Error loading the dataset: {e}")  # 加载数据集时出现异常打印错误信息
        exit(1)

    # 检查数据完整性
    if df.isnull().values.any():
        print(
            "Data contains missing values. Please preprocess the data."
        )  # 数据中存在缺失值，提示预处理数据
        exit(1)

    X = df.iloc[:, 1:].values  # 提取特征数据，排除第一列
    y = df.iloc[:, 0].values - 1  # 提取目标值，并将索引从 1 调整为 0 开始
    return X, y


def resample_data(X, y, args):
    """
    使用SMOTE（合成少数类过采样技术）对数据进行重采样。

    参数:
    - X: 特征矩阵，其中包含了待处理的特征数据。
    - y: 目标变量向量，包含了对应的标签信息。
    - args: 包含各种参数的对象，其中需要使用到k_neighbors和randomseed参数。

    返回值:
    - x_resampled: 重采样后的特征矩阵。
    - y_resampled: 重采样后的目标变量向量。
    """
    from imblearn.over_sampling import SMOTE

    # 初始化SMOTE对象，设置邻居数、随机种子和并行任务数
    sm = SMOTE(k_neighbors=args.kneighbors, random_state=args.randomseed)
    try:
        # 尝试对数据进行重采样
        x_resampled, y_resampled = sm.fit_resample(X, y)
    except Exception as e:
        # 如果重采样过程中出现异常，则打印错误信息并退出程序
        print(f"Error during resampling: {e}")
        exit(1)
    return x_resampled, y_resampled


def train_and_evaluate(X, y, args):
    """
    训练分类器并评估其性能。

    参数:
    - X: 特征矩阵，用于训练分类器。
    - y: 目标变量，表示每个样本的类别。
    - args: 包含各种设置的对象，例如k-fold交叉验证的折数。

    返回值:
    无。该函数直接打印评估结果或在出现异常时退出。
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.model_selection import cross_val_predict

    # 创建线性判别分析器实例
    clf = QuadraticDiscriminantAnalysis()
    print("\nClassifier parameters:")
    print(clf.get_params())

    try:
        # 使用k-fold交叉验证进行预测
        y_pred_resampled = cross_val_predict(
            clf, X, y, cv=args.kfolds, n_jobs=-1, verbose=1
        )
    except Exception as e:
        # 打印预测过程中的错误并退出
        print(f"Error during prediction: {e}")
        exit(1)

    # 根据目标变量的类别数量，选择合适的评估方法
    if len(np.unique(y)) > 2:
        # 对多分类问题进行模型评估
        model_evaluation(np.unique(y), y, y_pred_resampled)
    else:
        # 对二分类问题进行模型评估
        bi_model_evaluation(y, y_pred_resampled)


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    if args.kfolds < 2 or args.kneighbors < 2:
        print("kfolds and kneighbors must be at least 2.")
        exit(1)

    X, y = load_and_prepare_data(args)
    x_resampled, y_resampled = resample_data(X, y, args)
    train_and_evaluate(x_resampled, y_resampled, args)

    end_time = time.time()  # 程序结束时间
    print(
        f"\n[Finished in: { ((end_time - start_time) / 60):.3f} mins = {(end_time - start_time):.3f} seconds]"
    )
