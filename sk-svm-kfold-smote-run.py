# -*- coding: utf-8 -*-
"""
A SVM classifier based on scikit-learn with SMOTE apply it to classify bio datasets.
Date: 2020-04-01
"""
# import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="A SVM classifier based on scikit-learn with SMOTE apply it to classify bio datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--kernel", type=str,
                        help="Specifies the kernel type to be used in the algorithm. \
                        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or \
                        a callable.",
                        default='sigmoid')
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-n", "--kneighbors", type=int,
                        help="number of nearest neighbours to used to construct synthetic samples.", default=3)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)

    args = parser.parse_args()
    # logdir_base = os.getcwd()  # 获取当前目录

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn import svm
    from sklearn.model_selection import cross_val_predict
    from imblearn.over_sampling import SMOTE
    from lgb.utils import model_evaluation, bi_model_evaluation

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values - 1
    # f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # Apply SMOTE 生成 fake data
    sm = SMOTE(k_neighbors=args.kneighbors,
               random_state=args.randomseed, n_jobs=-1)
    x_resampled, y_resampled = sm.fit_sample(X, y)
    # after over sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled, return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['Class', 'Sum'])
    print("\nNumber of samples after over sampleing:\n{0}\n".format(
        df_resampled_y))

    # 初始化 classifier
    clf = svm.SVC(kernel=args.kernel, random_state=args.randomseed)
    print("\nClassifier parameters:")
    print(clf.get_params())
    print("\nSMOTE parameters:")
    print(sm.get_params())
    print("\n")
    # 交叉验证
    y_pred_resampled = cross_val_predict(
        clf, x_resampled, y_resampled, cv=args.kfolds, n_jobs=-1, verbose=1)
    # 通过 index 去除 fake data
    y_pred = y_pred_resampled[0:X.shape[0]]

    # 输出统计结果
    if(num_categories > 2):
        model_evaluation(num_categories, y, y_pred)
    else:
        bi_model_evaluation(y, y_pred)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
