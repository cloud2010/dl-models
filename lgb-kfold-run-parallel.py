# -*- coding: utf-8 -*-
"""
The implementation is based on LightGBM for predict bio datasets with kfold cross-validation.
LightGBM is A fast, distributed, high performance gradient boosting (GBDT, GBRT, GBM or MART) framework 
based on decision tree algorithms, used for ranking, classification and many other machine learning tasks. 
Project: `https://github.com/Microsoft/LightGBM`
Date: 2019-02-13
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="The implementation is based on LightGBM for predict bio datasets with kfold cross-validation.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--ntrees", type=int,
                        help="Number of boosted trees to fit.", default=100)
    parser.add_argument("-d", "--depth", type=int,
                        help="Maximum tree depth for base learners, -1 means no limit.", default=-1)
    parser.add_argument("-b", "--btype", type=str,
                        help="Boosting type: 'gbdt', traditional Gradient Boosting Decision Tree, 'rf', Random Forest, 'dart', Dropouts meet Multiple Additive Regression Trees, 'goss', Gradient-based One-Side Sampling.", default="gbdt")
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="The seed of the pseudo random number generator used when shuffling the data for probability estimates.", default=0)
    parser.add_argument("--datapath", type=str, help="The path of dataset.", required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
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

    # 初始化 classifier
    clf = lgb.LGBMClassifier(boosting_type=args.btype, n_estimators=args.ntrees,
                             max_depth=args.depth, random_state=args.randomseed)
    print("\nClassifier parameters:")
    print(clf.get_params())
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
        clf.fit(X[train_index], y[train_index])
        # 测试集验证
        y_pred = clf.predict(X[test_index], num_iteration=clf.best_iteration_)
        # 计算测试集 ACC
        accTest = accuracy_score(y[test_index], y_pred)
        print("\nFold:", k_fold_step, "Test Accuracy:",
              "{:.6f}".format(accTest), "Test Size:", test_index.size)
        # eval
        print('\nThe RMSE of test prediction is: {0:.6f}'.format(
            mean_squared_error(y[test_index], y_pred) ** 0.5))
        # feature importances
        # print('\nFeature importances:', list(clf.feature_importances_))
        # 暂存每次选中的测试集和预测结果
        test_cache = np.concatenate((test_cache, y[test_index]))
        pred_cache = np.concatenate((pred_cache, y_pred))
        print("\n=========================================================================")
        # 每个fold训练结束后次数 +1
        k_fold_step += 1

    # 输出统计结果
    if(num_categories > 2):
        model_evaluation(num_categories, test_cache, pred_cache)
    else:
        bi_model_evaluation(test_cache, pred_cache)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
