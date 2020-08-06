# -*- coding: utf-8 -*-
"""
The classifier integrates new feature select algorithm based on scikit-learn for classification.
Date: 2020-06-29
"""
import os
import sys
import time
from tqdm import tqdm, trange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="The classifier integrates new feature select algorithm based on scikit-learn for classification.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=88)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("-c", "--classifier", type=str,
                        help="The classifier.(lgb, xgb, rf, gdbt, ext, svm)", default="lgb")
    parser.add_argument("-f", "--feature", type=str,
                        help="The feature selection algorithm.(ReliefF, SURF, SURFstar)", default="SURF")

    args = parser.parse_args()
    # logdir_base = os.getcwd()  # 获取当前目录

    # 导入相关库
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support
    from skrebate import ReliefF, TuRF, SURF, SURFstar

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    # y = df.iloc[:, 0].map(lambda x: x if x == 1 else 0)  # 调整正负样本标注(1正样本/0负样本)
    f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # 初始化 classifier 字典
    clf = {
        'lgb': lgb.LGBMClassifier(random_state=args.randomseed, n_jobs=-1, boosting_type='gbdt'),
        'xgb': xgb.XGBClassifier(booster='gblinear', n_jobs=-1, random_state=args.randomseed),
        'gdbt': GradientBoostingClassifier(random_state=args.randomseed),
        'rf': RandomForestClassifier(n_jobs=-1, random_state=args.randomseed),
        'ext': ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=args.randomseed),
        'svm': SVC(kernel='sigmoid', random_state=args.randomseed)
    }

    # 初始化 fs 字典
    fs = {
        'ReliefF': ReliefF(n_features_to_select=100, verbose=False, n_jobs=-1),
        # 'TuRF': TuRF(core_algorithm="ReliefF", n_features_to_select=100, verbose=False, n_jobs=-1),
        'SURF': SURF(n_features_to_select=100, verbose=False, n_jobs=-1),
        'SURFstar': SURFstar(n_features_to_select=100, verbose=False, n_jobs=-1)
    }

    print('\nClassifier parameters:', clf[args.classifier])

    print('\nStarting cross validating without feature selection...\n')

    # 特征排序前的增量特征预测，根据名称调用字典中指定分类器
    y_pred_list = [cross_val_predict(
        clf[args.classifier], X[:, 0:i+1], y, cv=args.kfolds, n_jobs=-1) for i in trange(0, X.shape[1])]
    # print("\nClassifier parameters:")
    # print(clf[args.classifier].get_params())

    # 特征排序前的评估指标
    mcc_clf = [matthews_corrcoef(y, y_pred_list[i])
               for i in trange(0, X.shape[1])]
    acc_clf = [accuracy_score(y, y_pred_list[i])
               for i in trange(0, X.shape[1])]
    # 计算 precision, recall, F-measure 三个指标
    recall_pre_f1_clf = [precision_recall_fscore_support(
        y, y_pred_list[i], average='binary') for i in trange(0, X.shape[1])]

    # 特征排序，根据函数名，获得相应算法函数对象
    # fs_model = getattr(sys.modules[__name__], args.feature)

    # 特征排序
    fs_score = fs[args.feature].fit(X, y)

    # 获取排序好的索引
    fs_idxs = fs_score.top_features_

    # 输出特征排序模型参数
    print('\nFeature selection parameters:', fs[args.feature])

    print('\nStarting cross validating after feature selection...\n')
    # 特征排序后的增量特征预测
    y_pred_list_fs = [cross_val_predict(
        clf[args.classifier], X[:, fs_idxs[0:i+1]], y, cv=args.kfolds, n_jobs=-1) for i in trange(0, X.shape[1])]

    # 特征排序后的评估指标
    mcc_clf_fs = [matthews_corrcoef(y, y_pred_list_fs[i])
                  for i in trange(0, X.shape[1])]
    acc_clf_fs = [accuracy_score(y, y_pred_list_fs[i])
                  for i in trange(0, X.shape[1])]
    recall_pre_f1_clf_fs = [precision_recall_fscore_support(
        y, y_pred_list_fs[i], average='binary') for i in trange(0, X.shape[1])]

    # 测试输出正确性
    print('\nMCC Max before FS:', np.argmax(mcc_clf), np.max(mcc_clf))
    print('\nMCC Max after FS:', np.argmax(mcc_clf_fs), np.max(mcc_clf_fs))

    # 导出评估指标数据到 CSV
    df_fs = pd.DataFrame(data=list(zip(acc_clf, mcc_clf, acc_clf_fs, mcc_clf_fs)), columns=[
                         'acc', 'mcc', 'acc_fs', 'mcc_fs'])
    df_f1 = pd.DataFrame(data=recall_pre_f1_clf, columns=[
        'precision', 'recall', 'f-score', 'support'])
    df_fs_f1 = pd.DataFrame(data=recall_pre_f1_clf_fs, columns=[
        'precision_fs', 'recall_fs', 'f-score_fs', 'support_fs'])
    # 合并指标输出
    df_fs = pd.concat([df_fs, df_f1.iloc[:, 0:3],
                       df_fs_f1.iloc[:, 0:3]], axis=1)
    # 提取文件名
    filepath = os.path.basename(args.datapath).split('.')[0]
    # 方法2
    # filepath = args.datapath.split('/')[-1].split('.')[0]

    # 写入 CSV
    df_fs.to_csv('{0}_{1}_{2}.csv'.format(
        args.classifier, args.feature, filepath), index_label='feature')

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
