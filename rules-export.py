# -*- coding: utf-8 -*-
"""
This program is implementation of rules export as text.
Date: 2019-06-17
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description='This program is implementation of rules export as text.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--randomseed', type=int,
                        help='The seed of the pseudo random number generator used when shuffling the data for probability estimates.', default=0)
    parser.add_argument('--datapath', type=str, help='The path of dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print('\nDataset shape: ', X.shape, ' Number of features: ', X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # 初始化 classifier 并完成数据集训练
    clf = DecisionTreeClassifier().fit(X, y)
    print('\nClassifier parameters:')
    print(clf.get_params())

    # 输出分类规则
    r = tree.export_text(clf, feature_names=f_names.tolist())
    print('\nBuild a text report showing the rules of the classifier:\n')
    print(r)

    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
