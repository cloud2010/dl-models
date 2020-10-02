# -*- coding: utf-8 -*-
"""
This a DeepForest implementation for training and predict bio datasets.
Date: 2020-09-30
"""
# import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a DeepForest implementation for training and predict bio datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    import random
    import uuid
    from tqdm import tqdm, trange
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, KFold
    # from imblearn.over_sampling import SMOTE
    from deepForest.utils import model_evaluation, bi_model_evaluation
    from sklearn.metrics import accuracy_score
    from deepForest.deep_forest import MGCForest

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # Apply SMOTE 生成 fake data
    # sm = SMOTE(k_neighbors=args.kneighbors,
    #            random_state=args.randomseed, n_jobs=-1)
    # x_resampled, y_resampled = sm.fit_sample(X, y)
    # # after over sampleing 读取分类信息并返回数量
    # np_resampled_y = np.asarray(np.unique(y_resampled, return_counts=True))
    # df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['Class', 'Sum'])
    # print("\nNumber of samples after over sampleing:\n{0}\n".format(
    #     df_resampled_y))

    # 初始化 classifier
    clf = MGCForest(
        estimators_config={
            'mgs': [{
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 30,
                    'min_samples_split': 21,
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 30,
                    'min_samples_split': 21,
                    'n_jobs': -1,
                }
            }],
            'cascade': [{
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 30,
                    'min_samples_split': 11,
                    'max_features': 1,
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': ExtraTreesClassifier,
                'estimator_params': {
                    'n_estimators': 30,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 30,
                    'min_samples_split': 11,
                    'max_features': 1,
                    'n_jobs': -1,
                }
            }, {
                'estimator_class': RandomForestClassifier,
                'estimator_params': {
                    'n_estimators': 30,
                    'min_samples_split': 11,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                }
            }]
        },
        stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
    )
    # print("\nClassifier parameters:")
    # print(mgc_forest.get_params())
    # print("\nSMOTE parameters:")
    # print(sm.get_params())
    # print("\n")

    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True,
               random_state=args.randomseed)
    # 生成 k-fold 训练集、测试集索引
    cv_index_set = rs.split(y)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in tqdm(cv_index_set):
        print("\nFold:", k_fold_step)
        clf.fit(X[train_index], y[train_index])
        # 测试集验证
        y_pred = clf.predict(X[test_index])
        # 计算测试集 ACC
        accTest = accuracy_score(y[test_index], y_pred)
        print("\nFold:", k_fold_step, "Test Accuracy:",
              "{:.6f}".format(accTest), "Test Size:", test_index.size)
        # 暂存每次选中的测试集和预测结果
        test_cache = np.concatenate((test_cache, y[test_index]))
        pred_cache = np.concatenate((pred_cache, y_pred))
        print("\n=========================================================================")
        # 每个fold训练结束后次数 +1
        k_fold_step += 1

    # 输出统计结果
    if(num_categories > 2):
        model_evaluation(num_categories, y, y_pred)
    else:
        bi_model_evaluation(y, y_pred)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
