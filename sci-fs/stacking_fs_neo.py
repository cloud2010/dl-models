# -*- coding: utf-8 -*-
"""
The Stacking model integrates the feature selection based on scikit-learn for classification.
Date: 2020-07-02
"""
import os
import sys
import time
from tqdm import tqdm, trange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="The Stacking model integrates the feature selection based on scikit-learn for classification.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=88)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("-f", "--feature", type=str,
                        help="The feature selection algorithm.(f_classif, chi2, mutual_info_classif)", default="f_classif")

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support

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

    # 构建基分类器组
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100,
                                      n_jobs=-1, random_state=args.randomseed)),
        ('svm', SVC(kernel='sigmoid', random_state=args.randomseed)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, objective='binary:logistic',
                                  use_label_encoder=False, n_jobs=-1, random_state=args.randomseed)),
        ('lgb', lgb.LGBMClassifier(boosting_type='goss',
                                   n_estimators=100, n_jobs=-1, random_state=args.randomseed)),
        ('nb', GaussianNB()),  # 高斯朴素贝叶斯
        ('knn', KNeighborsClassifier(n_jobs=-1))  # k近邻
    ]

    # 构建元分类器
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            n_jobs=-1, random_state=args.randomseed),
        cv=5)

    print('\nClassifier parameters:', clf)

    # 特征排序，根据函数名，获得相应算法函数对象
    fs = getattr(sys.modules[__name__], args.feature)

    # 特征排序
    model_fs = SelectKBest(fs, k="all").fit(X, y)

    # 降序排列特征权重
    fs_idxs = np.argsort(-model_fs.scores_)

    print('\nStarting cross validating after feature selection...\n')
    # 特征排序后的增量特征预测
    y_pred_list_fs = [cross_val_predict(
        clf, X[:, fs_idxs[0:i+1]], y, cv=args.kfolds, n_jobs=-1) for i in trange(0, X.shape[1])]

    # 特征排序后的评估指标
    mcc_clf_fs = [matthews_corrcoef(y, y_pred_list_fs[i])
                  for i in trange(0, X.shape[1])]
    acc_clf_fs = [accuracy_score(y, y_pred_list_fs[i])
                  for i in trange(0, X.shape[1])]
    recall_pre_f1_clf_fs = [precision_recall_fscore_support(
        y, y_pred_list_fs[i], average='binary') for i in trange(0, X.shape[1])]

    # 测试输出正确性
    print('\nACC Max after FS:', np.argmax(acc_clf_fs), np.max(acc_clf_fs))
    print('\nMCC Max after FS:', np.argmax(mcc_clf_fs), np.max(mcc_clf_fs))

    # 导出评估指标数据到 CSV
    df_fs = pd.DataFrame(data=list(zip(acc_clf_fs, mcc_clf_fs)), columns=[
                         'acc_fs', 'mcc_fs'])
    df_f1 = pd.DataFrame(data=recall_pre_f1_clf_fs, columns=[
                         'precision', 'recall', 'f-score', 'support'])
    # 合并指标输出
    df_fs = pd.concat([df_fs, df_f1.iloc[:, 0:3]], axis=1)

    # 提取文件名
    filepath = os.path.basename(args.datapath).split('.')[0]
    # 方法2
    # filepath = args.datapath.split('/')[-1].split('.')[0]

    # 写入 CSV
    df_fs.to_csv('{0}_{1}_{2}.csv'.format(
        'stacking_neo', args.feature, filepath), index_label='feature')

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
