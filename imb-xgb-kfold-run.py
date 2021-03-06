# -*- coding: utf-8 -*-
"""
A Imbalance-Xgboost classifier apply it to classify bio datasets.
Date: 2020-04-16
Ref: https://github.com/jhwjhw0123/Imbalance-XGBoost
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="A Imbalance-Xgboost classifier apply it to classify bio datasets.",
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
    from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.metrics import accuracy_score, mean_squared_error
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
    clf_focal = imb_xgb(special_objective='focal')
    cv_focal_booster = GridSearchCV(
        clf_focal, {"focal_gamma": [1.0, 1.5, 2.0, 2.5, 3.0]})
    # print("\nClassifier parameters:")
    # print(clf.get_params())
    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True,
               random_state=args.randomseed)
    # 生成 k-fold 训练集、测试集索引
    cv_index_set = rs.split(y)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in cv_index_set:
        print("\nFold:", k_fold_step)
        cv_focal_booster.fit(X[train_index], y[train_index])
        # 获取最佳分类器
        clf_opt = cv_focal_booster.best_estimator_
        # 测试集验证
        y_pred = clf_opt.predict_determine(X[test_index])
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
