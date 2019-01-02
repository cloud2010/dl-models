# -*- coding: utf-8 -*-
"""
This a implementation of the rulefit algorithm for predict bio datasets with kfold cross-validation and SMOTE.
Derived from: Christoph Molnar
Source: https://github.com/christophM/rulefit
Date: 2019-01-03
"""
import os
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a implementation of the rulefit algorithm for predict bio datasets with kfold cross-validation and SMOTE.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--treesize", type=int,
                        help="Number of terminal nodes in generated trees. If exp_rand_tree_size=True, this will be the mean number of terminal nodes.", default=4)
    parser.add_argument("-m", "--maxrules", type=int,
                        help="approximate total number of rules generated for fitting. Note that actual number of rules will usually be lower than this due to duplicates.", default=2000)
    parser.add_argument("--rfmode", type=str,
                        help="'regress' for regression or 'classify' for binary classification.", default='classify')
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=None)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)

    args = parser.parse_args()

    # 导入相关库
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from rulefit import RuleFit
    from rulefit import utils
    from imblearn.over_sampling import SMOTE

    # 读取数据
    df = pd.read_csv(args.datapath)

    # 设定分类信息和特征矩阵
    X_origin = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    # 标准化处理
    scaler = StandardScaler().fit(X_origin)
    X = scaler.transform(X_origin)
    # 读取特征名称
    features = df.columns[1:]
    print("\nDataset shape: ", df.shape, " Number of features: ", features.size)
    # 不同 Class 统计 (根据 Target 列)
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)
    # Apply SMOTE 生成 fake data
    sm = SMOTE(k_neighbors=2)
    x_resampled, y_resampled = sm.fit_sample(X, y)
    # after over sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled.astype(int), return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['Class', 'Sum'])
    print("\nNumber of samples after over sampleing:\n{0}".format(df_resampled_y))

    print("\nTraining Start...")

    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.randomseed)
    # 生成 k-fold 训练集、测试集索引
    resampled_index_set = rs.split(y_resampled)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in resampled_index_set:
        print("\nFold:", k_fold_step)
        # 初始化 estimator 训练集进入模型
        rf = RuleFit(tree_size=args.treesize, rfmode=args.rfmode,
                     max_rules=args.maxrules, random_state=args.randomseed)
        rf.fit(x_resampled[train_index], y_resampled[train_index], features)
        # 测试集验证
        # 验证测试集 (通过 index 去除 fake data)
        real_test_index = test_index[test_index < X.shape[0]]
        batch_test_x = x_resampled[real_test_index]
        batch_test_y = y_resampled[real_test_index]
        batch_test_size = len(real_test_index)
        y_pred = rf.predict(batch_test_x)
        # 计算测试集 ACC
        accTest = accuracy_score(batch_test_y, y_pred)
        print("\nFold:", k_fold_step, "Test Accuracy:",
              "{:.6f}".format(accTest), "Test Size:", batch_test_size)
        # 暂存每次选中的测试集和预测结果
        test_cache = np.concatenate((test_cache, batch_test_y))
        pred_cache = np.concatenate((pred_cache, y_pred))
        print("\n=========================================================================")
        # 每个fold训练结束后次数 +1
        k_fold_step += 1

    # 末尾输出rulefit模型参数
    print("\n=== Model parameters ===")
    print("\nTree generator:{0}, \n\nMax rules:{1}, Tree size:{2}, Random state:{3}, K-fold:{4}".format(
        rf.tree_generator, rf.max_rules, rf.tree_size, rf.random_state, args.kfolds))
    # 输出统计结果
    if(num_categories > 2):
        utils.model_evaluation(num_categories, test_cache, pred_cache)
    else:
        utils.bi_model_evaluation(test_cache, pred_cache)

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
