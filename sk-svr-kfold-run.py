# -*- coding: utf-8 -*-
"""
Support Vector Regression based on scikit-learn apply it to predict bio datasets.
Date: 2023-06-04
"""
# import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="Support Vector Regression based on scikit-learn apply it to predict bio datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--kernel", type=str,
                        help="Specifies the kernel type to be used in the algorithm. \
                        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or \
                        a callable.",
                        default='sigmoid')
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)

    parser.add_argument('--output', type=str,
                        help='The path of output dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn.svm import SVR
    from sklearn.model_selection import cross_val_predict

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values - 1
    f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # 初始化 Estimator
    regr = SVR(kernel=args.kernel)
    print("\nEstimator parameters:")
    print(regr.get_params())
    print("\n")
    # 交叉验证
    y_pred = cross_val_predict(
        regr, X, y, cv=args.kfolds, n_jobs=-1, verbose=0)

    # 输出统计结果
    df_output = pd.DataFrame(index=None, data=y_pred,
                             columns=['The predicted values'])
    df_output.to_csv(f'{args.output}')
    print(f'The predicted values saved to:`{args.output}`')
    end_time = time.time()  # 程序结束时间
    print(
        f'\n[Finished in: { ((end_time - start_time) / 60):.3f} mins = {(end_time - start_time):.3f} seconds]')
