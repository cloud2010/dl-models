# -*- coding: utf-8 -*-
"""
Multiple XGBoost integrates the feature sorting algorithm based on scikit-learn for prediction.
Date: 2020-08-14
"""
import os
import sys
import time
from tqdm import tqdm, trange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="Multiple classifier integrates the feature sorting algorithm based on scikit-learn for prediction.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-i", "--idx", type=int,
                        help="Number of features to select. Must be at least 10.", default=150)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("-f", "--feature", type=str,
                        help="The feature selection algorithm.(f_classif, svm_rfe, mutual_info_classif)", default="f_classif")

    args = parser.parse_args()
    # logdir_base = os.getcwd()  # 获取当前目录

    # 导入相关库
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support

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

    # 初始化 classifier 和预测结果字典
    clf_dict = {
        'lgb': lgb.LGBMClassifier(random_state=args.randomseed, n_jobs=-1, boosting_type='goss'),
        'xgb': xgb.XGBClassifier(booster='gblinear', objective='binary:logistic', n_jobs=-1, random_state=args.randomseed),
    }
    k_keys = ['2D_3*3_1D_3', '2D_3*3_1D_4', '2D_3*3_1D_5',
              '2D_5*5_1D_3', '2D_5*5_1D_4', '2D_5*5_1D_5']
    y_pred_dict = dict().fromkeys(k_keys, np.array([]))

    if(args.feature == 'svm_rfe'):
        fs = RFE(estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                              n_jobs=-1,
                                              random_state=args.randomseed),
                 n_features_to_select=100)
        model_fs = fs.fit(X, y)
        # 降序排列特征权重
        fs_idxs = np.argsort(-model_fs.ranking_)
    else:
        # 特征排序，根据函数名，获得相应算法函数对象
        fs = getattr(sys.modules[__name__], args.feature)
        # 特征排序
        model_fs = SelectKBest(fs, k="all").fit(X, y)
        # 降序排列特征权重
        fs_idxs = np.argsort(-model_fs.scores_)

    print('\nStarting cross validating after feature selection...\n')
    # 特征排序后的各分类器在指定的特征维度下进行交叉验证
    for key in tqdm(k_keys):
        bias = k_keys.index(key)  # 获取对应类型的索引
        if bias < 3:
            y_pred_dict[key] = cross_val_predict(clf_dict['xgb'], X[:, fs_idxs[0:args.idx+bias*10]], y,
                                                 cv=KFold(n_splits=args.kfolds,
                                                          shuffle=True,
                                                          random_state=args.randomseed),
                                                 n_jobs=-1)
        else:
            y_pred_dict[key] = cross_val_predict(clf_dict['lgb'], X[:, fs_idxs[0:args.idx+bias*10]], y,
                                                 cv=KFold(n_splits=args.kfolds,
                                                          shuffle=True,
                                                          random_state=args.randomseed),
                                                 n_jobs=-1)
    # 特征排序后的评估指标
    mcc_clf_fs = [matthews_corrcoef(y, y_pred_dict[i])
                  for i in k_keys]
    acc_clf_fs = [accuracy_score(y, y_pred_dict[i]) for i in k_keys]
    recall_pre_f1_clf_fs = [precision_recall_fscore_support(
        y, y_pred_dict[i], average='binary') for i in k_keys]

    # 导出评估指标数据到 CSV
    df_fs = pd.DataFrame(data=list(zip(acc_clf_fs, mcc_clf_fs)), columns=[
                         'acc_fs', 'mcc_fs'], index=None)
    df_f1 = pd.DataFrame(data=recall_pre_f1_clf_fs, columns=[
                         'precision_fs', 'recall_fs', 'f-score_fs', 'support_fs'], index=None)
    df_index = pd.DataFrame(data=k_keys, columns=['comb'], index=None)
    # 合并指标输出
    df_fs = pd.concat([df_index, df_fs, df_f1.iloc[:, 0:3]], axis=1)
    # 提取文件名
    filepath = os.path.basename(args.datapath).split('.')[0]

    # 写入 CSV
    df_fs.to_csv('multi_xgb_{0}_{1}_F{2}.csv'.format(
        args.feature, filepath, args.idx), index=None)
    print("\nThe prediction of all classifiers have been saved to 'multi_{0}_{1}_F{2}.csv'".format(
        args.feature, filepath, args.idx))
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
