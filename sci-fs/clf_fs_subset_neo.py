# -*- coding: utf-8 -*-
"""
The classifier integrates the feature sorting algorithm based on scikit-learn for subset prediction.
Date: 2020-08-03
"""
import os
import sys
import time
from tqdm import tqdm, trange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="The classifier integrates the feature sorting algorithm based on scikit-learn for subset prediction.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("-c", "--classifier", type=str,
                        help="The classifier.(lgb, xgb, cgb, rf, nb, knn, svm)", default="knn")
    parser.add_argument("-f", "--feature", type=str,
                        help="The feature selection algorithm.(f_classif, mutual_info_classif)", default="f_classif")

    args = parser.parse_args()
    # logdir_base = os.getcwd()  # 获取当前目录

    # 导入相关库
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cgb
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
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

    # 读取五组特征子集数据
    X_PseAAC = df.filter(regex='PseAAC*', axis=1).iloc[:, 0:].values
    X_Disorder = df.filter(regex='Disorder*', axis=1).iloc[:, 0:].values
    X_PSSM = df.filter(regex='PSSM*', axis=1).iloc[:, 0:].values
    X_CKSAAP = df.filter(regex='CKSAAP*', axis=1).iloc[:, 0:].values
    X_AAIndex = df.filter(regex='AAIndex*', axis=1).iloc[:, 0:].values

    # 初始化 classifier 字典
    clf = {
        'lgb': lgb.LGBMClassifier(random_state=args.randomseed, n_jobs=-1, boosting_type='gbdt'),
        'xgb': xgb.XGBClassifier(n_jobs=-1, random_state=args.randomseed),
        'cgb': cgb.CatBoostClassifier(n_estimators=100, thread_count=-1, logging_level='Silent', random_seed=args.randomseed),
        'rf': RandomForestClassifier(n_jobs=-1, random_state=args.randomseed),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'svm': SVC(kernel='sigmoid', random_state=args.randomseed)
    }

    print('\nClassifier parameters:', clf[args.classifier])

    # 特征排序，根据函数名，获得相应算法函数对象
    fs = getattr(sys.modules[__name__], args.feature)
    # 分组进行特征排序
    model_fs_PseAAC = SelectKBest(fs, k="all").fit(X_PseAAC, y)
    model_fs_Disorder = SelectKBest(fs, k="all").fit(X_Disorder, y)
    model_fs_PSSM = SelectKBest(fs, k="all").fit(X_PSSM, y)
    model_fs_CKSAAP = SelectKBest(fs, k="all").fit(X_CKSAAP, y)
    model_fs_AAIndex = SelectKBest(fs, k="all").fit(X_AAIndex, y)
    # 降序排列特征权重
    fs_idxs_PseAAC = np.argsort(-model_fs_PseAAC.scores_)
    fs_idxs_Disorder = np.argsort(-model_fs_Disorder.scores_)
    fs_idxs_PSSM = np.argsort(-model_fs_PSSM.scores_)
    fs_idxs_CKSAAP = np.argsort(-model_fs_CKSAAP.scores_)
    fs_idxs_AAIndex = np.argsort(-model_fs_AAIndex.scores_)

    print('\nStarting cross validating after feature selection...\n')
    # 特征排序后的五组特征子集IFS预测
    y_pred_list_fs_PseAAC = [cross_val_predict(
        clf[args.classifier], X_PseAAC[:, fs_idxs_PseAAC[0:i+1]], y, cv=args.kfolds, n_jobs=-1) for i in trange(0, X_PseAAC.shape[1])]
    y_pred_list_fs_Disorder = [cross_val_predict(
        clf[args.classifier], X_Disorder[:, fs_idxs_Disorder[0:i+1]], y, cv=args.kfolds, n_jobs=-1) for i in trange(0, X_Disorder.shape[1])]
    y_pred_list_fs_PSSM = [cross_val_predict(
        clf[args.classifier], X_PSSM[:, fs_idxs_PSSM[0:i+1]], y, cv=args.kfolds, n_jobs=-1) for i in trange(0, X_PSSM.shape[1])]
    y_pred_list_fs_CKSAAP = [cross_val_predict(
        clf[args.classifier], X_CKSAAP[:, fs_idxs_CKSAAP[0:i+1]], y, cv=args.kfolds, n_jobs=-1) for i in trange(0, X_CKSAAP.shape[1])]
    y_pred_list_fs_AAIndex = [cross_val_predict(
        clf[args.classifier], X_AAIndex[:, fs_idxs_AAIndex[0:i+1]], y, cv=args.kfolds, n_jobs=-1) for i in trange(0, X_AAIndex.shape[1])]

    # 特征排序后的评估指标
    # PseAAC
    mcc_clf_fs_PseAAC = [matthews_corrcoef(y, y_pred_list_fs_PseAAC[i])
                         for i in trange(0, X_PseAAC.shape[1])]
    acc_clf_fs_PseAAC = [accuracy_score(y, y_pred_list_fs_PseAAC[i])
                         for i in trange(0, X_PseAAC.shape[1])]
    recall_pre_f1_clf_fs_PseAAC = [precision_recall_fscore_support(
        y, y_pred_list_fs_PseAAC[i], average='binary') for i in trange(0, X_PseAAC.shape[1])]
    # Disorder
    mcc_clf_fs_Disorder = [matthews_corrcoef(y, y_pred_list_fs_Disorder[i])
                           for i in trange(0, X_Disorder.shape[1])]
    acc_clf_fs_Disorder = [accuracy_score(y, y_pred_list_fs_Disorder[i])
                           for i in trange(0, X_Disorder.shape[1])]
    recall_pre_f1_clf_fs_Disorder = [precision_recall_fscore_support(
        y, y_pred_list_fs_Disorder[i], average='binary') for i in trange(0, X_Disorder.shape[1])]
    # PSSM
    mcc_clf_fs_PSSM = [matthews_corrcoef(y, y_pred_list_fs_PSSM[i])
                       for i in trange(0, X_PSSM.shape[1])]
    acc_clf_fs_PSSM = [accuracy_score(y, y_pred_list_fs_PSSM[i])
                       for i in trange(0, X_PSSM.shape[1])]
    recall_pre_f1_clf_fs_PSSM = [precision_recall_fscore_support(
        y, y_pred_list_fs_PSSM[i], average='binary') for i in trange(0, X_PSSM.shape[1])]
    # CKSAAP
    mcc_clf_fs_CKSAAP = [matthews_corrcoef(y, y_pred_list_fs_CKSAAP[i])
                         for i in trange(0, X_CKSAAP.shape[1])]
    acc_clf_fs_CKSAAP = [accuracy_score(y, y_pred_list_fs_CKSAAP[i])
                         for i in trange(0, X_CKSAAP.shape[1])]
    recall_pre_f1_clf_fs_CKSAAP = [precision_recall_fscore_support(
        y, y_pred_list_fs_CKSAAP[i], average='binary') for i in trange(0, X_CKSAAP.shape[1])]
    # AAIndex
    mcc_clf_fs_AAIndex = [matthews_corrcoef(y, y_pred_list_fs_AAIndex[i])
                          for i in trange(0, X_AAIndex.shape[1])]
    acc_clf_fs_AAIndex = [accuracy_score(y, y_pred_list_fs_AAIndex[i])
                          for i in trange(0, X_AAIndex.shape[1])]
    recall_pre_f1_clf_fs_AAIndex = [precision_recall_fscore_support(
        y, y_pred_list_fs_AAIndex[i], average='binary') for i in trange(0, X_AAIndex.shape[1])]

    # 测试输出正确性
    print('\nACC Max after FS with PseAAC:', np.argmax(
        acc_clf_fs_PseAAC), np.max(acc_clf_fs_PseAAC))
    print('\nMCC Max after FS with PseAAC:', np.argmax(
        mcc_clf_fs_PseAAC), np.max(mcc_clf_fs_PseAAC))

    print('\nACC Max after FS with Disorder:', np.argmax(
        acc_clf_fs_Disorder), np.max(acc_clf_fs_Disorder))
    print('\nMCC Max after FS with Disorder:', np.argmax(
        mcc_clf_fs_Disorder), np.max(mcc_clf_fs_Disorder))

    print('\nACC Max after FS with PSSM:', np.argmax(
        acc_clf_fs_PSSM), np.max(acc_clf_fs_PSSM))
    print('\nMCC Max after FS with PSSM:', np.argmax(
        mcc_clf_fs_PSSM), np.max(mcc_clf_fs_PSSM))

    print('\nACC Max after FS with CKSAAP:', np.argmax(
        acc_clf_fs_CKSAAP), np.max(acc_clf_fs_CKSAAP))
    print('\nMCC Max after FS with CKSAAP:', np.argmax(
        mcc_clf_fs_CKSAAP), np.max(mcc_clf_fs_CKSAAP))

    print('\nACC Max after FS with AAIndex:', np.argmax(
        acc_clf_fs_AAIndex), np.max(acc_clf_fs_AAIndex))
    print('\nMCC Max after FS with AAIndex:', np.argmax(
        mcc_clf_fs_AAIndex), np.max(mcc_clf_fs_AAIndex))

    # 提取文件名
    filepath = os.path.basename(args.datapath).split('.')[0]

    # 导出评估指标数据到 CSV
    # PseAAC
    df_fs_PseAAC = pd.DataFrame(data=list(zip(acc_clf_fs_PseAAC, mcc_clf_fs_PseAAC)), columns=[
        'acc_fs', 'mcc_fs'])
    df_f1_PseAAC = pd.DataFrame(data=recall_pre_f1_clf_fs_PseAAC, columns=[
        'precision', 'recall', 'f-score', 'support'])
    # 合并指标输出
    df_fs_PseAAC = pd.concat([df_fs_PseAAC, df_f1_PseAAC.iloc[:, 0:3]], axis=1)
    # 写入 CSV
    df_fs_PseAAC.to_csv('PseAAC_{0}_{1}_{2}.csv'.format(
        args.classifier, args.feature, filepath), index_label='feature')

    # Disorder
    df_fs_Disorder = pd.DataFrame(data=list(zip(acc_clf_fs_Disorder, mcc_clf_fs_Disorder)), columns=[
        'acc_fs', 'mcc_fs'])
    df_f1_Disorder = pd.DataFrame(data=recall_pre_f1_clf_fs_Disorder, columns=[
        'precision', 'recall', 'f-score', 'support'])
    # 合并指标输出
    df_fs_Disorder = pd.concat(
        [df_fs_Disorder, df_f1_Disorder.iloc[:, 0:3]], axis=1)
    # 写入 CSV
    df_fs_Disorder.to_csv('Disorder_{0}_{1}_{2}.csv'.format(
        args.classifier, args.feature, filepath), index_label='feature')

    # PSSM
    df_fs_PSSM = pd.DataFrame(data=list(zip(acc_clf_fs_PSSM, mcc_clf_fs_PSSM)), columns=[
        'acc_fs', 'mcc_fs'])
    df_f1_PSSM = pd.DataFrame(data=recall_pre_f1_clf_fs_PSSM, columns=[
        'precision', 'recall', 'f-score', 'support'])
    # 合并指标输出
    df_fs_PSSM = pd.concat([df_fs_PSSM, df_f1_PSSM.iloc[:, 0:3]], axis=1)
    # 写入 CSV
    df_fs_PSSM.to_csv('PSSM_{0}_{1}_{2}.csv'.format(
        args.classifier, args.feature, filepath), index_label='feature')

    # CKSAAP
    df_fs_CKSAAP = pd.DataFrame(data=list(zip(acc_clf_fs_CKSAAP, mcc_clf_fs_CKSAAP)), columns=[
        'acc_fs', 'mcc_fs'])
    df_f1_CKSAAP = pd.DataFrame(data=recall_pre_f1_clf_fs_CKSAAP, columns=[
        'precision', 'recall', 'f-score', 'support'])
    # 合并指标输出
    df_fs_CKSAAP = pd.concat([df_fs_CKSAAP, df_f1_CKSAAP.iloc[:, 0:3]], axis=1)
    # 写入 CSV
    df_fs_CKSAAP.to_csv('CKSAAP_{0}_{1}_{2}.csv'.format(
        args.classifier, args.feature, filepath), index_label='feature')

    # AAIndex
    df_fs_AAIndex = pd.DataFrame(data=list(zip(acc_clf_fs_AAIndex, mcc_clf_fs_AAIndex)), columns=[
        'acc_fs', 'mcc_fs'])
    df_f1_AAIndex = pd.DataFrame(data=recall_pre_f1_clf_fs_AAIndex, columns=[
        'precision', 'recall', 'f-score', 'support'])
    # 合并指标输出
    df_fs_AAIndex = pd.concat(
        [df_fs_AAIndex, df_f1_AAIndex.iloc[:, 0:3]], axis=1)
    # 写入 CSV
    df_fs_AAIndex.to_csv('AAIndex_{0}_{1}_{2}.csv'.format(
        args.classifier, args.feature, filepath), index_label='feature')

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
