# -*- coding: utf-8 -*-
"""
This program is implementation of the Boruta all-relevant feature selection method.
Ref: `https://github.com/scikit-learn-contrib/boruta_py`
Date: 2019-02-27
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description='This program is implementation of the Boruta all-relevant feature selection method.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--ntrees', type=int,
                        help='The number of trees in the forest to fit.', default=100)
    parser.add_argument('-d', '--depth', type=int,
                        help='The maximum depth of the tree.', default=7)
    parser.add_argument('-n', '--iter', type=int,
                        help='The number of maximum iterations to perform.', default=100)
    parser.add_argument('-r', '--randomseed', type=int,
                        help='The seed of the pseudo random number generator used when shuffling the data for probability estimates.', default=0)
    parser.add_argument('--datapath', type=str, help='The path of dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy

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

    # 初始化 classifier
    clf = RandomForestClassifier(n_estimators=args.ntrees, n_jobs=-1,
                                 class_weight=None, max_depth=args.depth, random_state=args.randomseed)
    print('\nClassifier parameters:')
    print(clf.get_params())
    # Define Boruta feature selection method
    feat_selector = BorutaPy(clf, n_estimators='auto', verbose=2,
                             max_iter=args.iter, random_state=args.randomseed)
    # Once built, we can use this object to identify the relevant features in our dataset.
    feat_selector.fit(X, y)
    # Select the chosen features from our dataframe.
    selected = X[:, feat_selector.support_]
    print('\nSelected Feature Matrix Shape:', selected.shape)
    # 构建新的特征组名称
    new_f_names = np.insert(f_names[feat_selector.support_], 0, df.columns[0])
    # 合并新的特征矩阵
    new_data = np.column_stack((y, X[:, feat_selector.support_]))
    new_df = pd.DataFrame(index=None, data=new_data, columns=new_f_names)
    # 输出新的特征集
    new_df.to_csv('{0}-new-mat.csv'.format(args.datapath), index=None)
    print('\nNew Feature Matrix saved to:`{0}-new-mat.csv`'.format(args.datapath))
    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
