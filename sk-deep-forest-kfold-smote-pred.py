# -*- coding: utf-8 -*-
"""
A deep forest classifier based on scikit-learn with SMOTE apply it to classify bio datasets (pred).
Date: 2021-03-21
Ref: https://github.com/LAMDA-NJU/Deep-Forest
"""
# import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="A deep forest classifier based on scikit-learn with SMOTE apply it to classify bio datasets (pred).",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-n", "--kneighbors", type=int,
                        help="number of nearest neighbours to used to construct synthetic samples.", default=3)
    parser.add_argument("--train", type=str,
                        help="The path of training dataset.", required=True)
    parser.add_argument("--test", type=str,
                        help="The path of test dataset.", required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    # from sklearn.ensemble import RandomForestClassifier
    from deepforest import CascadeForestClassifier
    from sklearn.model_selection import cross_val_predict
    from imblearn.over_sampling import SMOTE
    from lgb.utils import model_evaluation, bi_model_evaluation

    # 读取数据
    df = pd.read_csv(args.train)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values - 1
    # 读取测试集
    df_test = pd.read_csv(args.test)
    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values - 1
    # f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print("\nTraining Dataset shape: ", X.shape,
          " Number of features: ", X.shape[1])
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
    clf = CascadeForestClassifier(random_state=args.randomseed)
    print("\nClassifier parameters:")
    print(clf.get_params())
    print("\nSMOTE parameters:")
    print(sm.get_params())
    print("\n")

    # 使用SMOTE后数据进行训练
    clf.fit(x_resampled, y_resampled)
    # 预测测试集
    y_pred = clf.predict(X_test)

    # 输出测试集统计结果
    if(num_categories > 2):
        model_evaluation(num_categories, y_test, y_pred)
    else:
        bi_model_evaluation(y_test, y_pred)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
