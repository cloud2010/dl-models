# -*- coding: utf-8 -*-
"""
Show feature importances and explain predictions of tree-based ensembles use ELI5 package.

Date: 2019-08-10
Ref: https://github.com/TeamHG-Memex/eli5
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description='This program is show feature importances and explain predictions of tree-based ensembles use ELI5 package.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--randomseed', type=int,
                        help='The seed of the pseudo random number generator used when shuffling the data for probability estimates.', default=0)
    parser.add_argument('--datapath', type=str, help='The path of dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import eli5

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X_train = df.iloc[:, 1:].values
    y_train = df.iloc[:, 0].values - 1
    f_names = df.columns[1:].values
    t_names = df.iloc[:, 0].unique()
    # 不同 Class 统计 (根据 Target 列)
    print('\nTraining dataset shape: ', X_train.shape, ' Number of features: ', X_train.shape[1])
    num_categories = np.unique(y_train).size
    sum_y = np.asarray(np.unique(y_train.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # 初始化 classifier 并完成数据集训练
    clf = RandomForestClassifier(verbose=1, n_jobs=-1, random_state=args.randomseed,
                                 n_estimators=100).fit(X_train, y_train)
    print('\nClassifier parameters:\n')
    print(clf.get_params())

    # 输出重要特征评分
    df_import = eli5.explain_weights_df(clf, target_names=t_names, feature_names=f_names)
    df_import.to_csv('f_weight_output.csv', index=None)
    print("\nThe importance features have been saved to 'f_weight_output.csv'.")

    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]\n'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
