# -*- coding: utf-8 -*-
"""
This a implementation of the rulefit algorithm for predict bio datasets and kfold cross-validation.
Derived from: Christoph Molnar
Source: https://github.com/christophM/rulefit
Date: 2018-12-11
"""
import os
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a implementation of the rulefit algorithm for predict bio datasets and kfold cross-validation.",
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

    # 读取数据
    df = pd.read_csv(args.datapath)

    # 设定分类信息和特征矩阵
    X_origin = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    # 读取特征名称
    features = df.columns[1:]
    print("\nDataset shape: ", df.shape, " Number of features: ", features.size)
    # 不同 Class 统计 (根据第1列)
    class_info = df.groupby(df.columns[0]).size()
    print('\n', class_info)
    print("\nTraining Start...")

    # 标准化处理
    scaler = StandardScaler().fit(X_origin)
    X = scaler.transform(X_origin)

    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.randomseed)
    # 生成 k-fold 训练集、测试集索引
    cv_index_set = rs.split(y)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in cv_index_set:
        print("\nFold:", k_fold_step)
        # 初始化 estimator 训练集进入模型
        rf = RuleFit(tree_size=args.treesize, rfmode=args.rfmode,
                     max_rules=args.maxrules, random_state=args.randomseed)
        rf.fit(X[train_index], y[train_index], features)
        # 测试集验证
        y_pred = rf.predict(X[test_index])
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

    # 末尾输出rulefit模型参数
    print("\n=== Model parameters ===")
    print("\nTree generator:{0}, \n\nMax rules:{1}, Tree size:{2}, Random state:{3}, K-fold:{4}".format(
        rf.tree_generator, rf.max_rules, rf.tree_size, rf.random_state, args.kfolds))
    # 输出统计结果
    num_categories = class_info.values.size
    if(num_categories > 2):
        utils.model_evaluation(num_categories, test_cache, pred_cache)
    else:
        utils.bi_model_evaluation(test_cache, pred_cache)
    # 输出 rules
    # rules = rf.get_rules()
    # rules = rules[rules.coef != 0].sort_values("support", ascending=False)
    # 输出 rules 至 CSV 文件
    # rules.to_csv('dataset_rules.csv')
    # print("\nThe inspect rules have been saved to 'dataset_rules.csv'")
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
