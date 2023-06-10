# -*- coding: utf-8 -*-
"""
This program implements two ensemble classifiers for predicting biological sequence datasets.
Date: 2023-06-10
"""
# import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This program implements two ensemble classifiers for predicting biological sequence datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-c", "--classifier", type=str,
                        help="Please specify which classifier to use. ExtraTreesClassifier(ext) or AdaBoostClassifier(ada)", default="ext")
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    # parser.add_argument('--output', type=str, help='The path of output dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
    from sklearn.model_selection import cross_val_predict
    from lgb.utils import model_evaluation, bi_model_evaluation

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

    # 初始化 classifier
    if (args.classifier == "ada"):
        clf = AdaBoostClassifier(random_state=args.randomseed)
        print('\nAdaBoostClassifier')
    else:
        clf = ExtraTreesClassifier(random_state=args.randomseed, n_jobs=-1)
        print('\nExtraTreesClassifier')
    print("\nClassifier parameters:")
    print(clf.get_params())
    # 交叉验证
    y_pred = cross_val_predict(
        clf, X, y, cv=args.kfolds, n_jobs=-1, verbose=0)

    # 输出统计结果
    if(num_categories > 2):
        model_evaluation(num_categories, y, y_pred)
    else:
        bi_model_evaluation(y, y_pred)

    # 输出统计结果
    # df_output = pd.DataFrame(index=None, data=y_pred,
    #                          columns=['The predicted values'])
    # df_output.to_csv(f'{args.output}')
    # print(f'The predicted values saved to:`{args.output}`')
    end_time = time.time()  # 程序结束时间
    print(
        f'\n[Finished in: { ((end_time - start_time) / 60):.3f} mins = {(end_time - start_time):.3f} seconds]')
