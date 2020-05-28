# -*- coding: utf-8 -*-
"""
The classifier integrates the feature sorting algorithm based on scikit-learn for prediction.
Date: 2020-05-28
"""
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="The classifier integrates the feature sorting algorithm based on scikit-learn for prediction.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("-c", "--classifier", type=str,
                        help="The Classifier.(lgb, xgb, rf, gdbt)", default="lgb")
    parser.add_argument("-f", "--feature", type=str,
                        help="The feature selection algorithm.(f_classif, chi2, mutual_info_classif)", default="f_classif")

    args = parser.parse_args()
    # logdir_base = os.getcwd()  # 获取当前目录

    # 导入相关库
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, classification_report

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values - 1
    f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # 初始化 classifier 字典
    clf = {'lgb': lgb.LGBMClassifier(random_state=args.randomseed, n_jobs=-1, boosting_type='goss'),
           'xgb': xgb.XGBClassifier(n_jobs=-1, random_state=args.randomseed),
           'gdbt': GradientBoostingClassifier(random_state=args.randomseed),
           'rf': RandomForestClassifier(n_jobs=-1, random_state=args.randomseed)}

    # 特征排序前的增量特征预测，根据名称调用字典中指定分类器
    y_pred_list = [cross_val_predict(
        clf[args.classifier], X[:, 0:i+1], y, cv=args.kfolds, n_jobs=-1) for i in range(0, X.shape[1])]
    print("\nClassifier parameters:")
    print(clf[args.classifier].get_params())

    # 特征排序前的评估指标
    mcc_clf = [matthews_corrcoef(y, y_pred_list[i])
               for i in range(0, X.shape[1])]
    acc_clf = [accuracy_score(y, y_pred_list[i]) for i in range(0, X.shape[1])]
    f1_clf = [f1_score(y, y_pred_list[i]) for i in range(0, X.shape[1])]

    # 特征排序，根据函数名，获得相应算法函数对象
    fs = getattr(sys.modules[__name__], args.feature)

    # 特征排序
    model_fs = SelectKBest(fs, k="all").fit(X, y)

    # 降序排列特征权重
    fs_idxs = np.argsort(-model_fs.scores_)

    # 特征排序后的增量特征预测
    y_pred_list_fs = [cross_val_predict(
        clf[args.classifier], X[:, fs_idxs[0:i+1]], y, cv=args.kfolds, n_jobs=-1) for i in range(0, X.shape[1])]

    # 特征排序后的评估指标
    mcc_clf_fs = [matthews_corrcoef(y, y_pred_list_fs[i])
                  for i in range(0, X.shape[1])]
    acc_clf_fs = [accuracy_score(y, y_pred_list_fs[i])
                  for i in range(0, X.shape[1])]
    f1_clf_fs = [f1_score(y, y_pred_list_fs[i]) for i in range(0, X.shape[1])]

    # 测试输出正确性
    print('\nMCC Max:', np.argmax(mcc_clf), np.max(mcc_clf))
    print('\nMCC Max:', np.argmax(mcc_clf_fs), np.max(mcc_clf_fs))

    # 导出评估指标数据到 CSV
    header_names = ['{0}_{1}'.format(i, args.classifier) for i in ['acc', 'mcc', 'f1', 'acc_{0}'.format(
        args.feature), 'mcc_{0}'.format(args.feature), 'f1_{0}'.format(args.feature)]]
    df_fs = pd.DataFrame(data=list(zip(acc_clf, mcc_clf, f1_clf, acc_clf_fs, mcc_clf_fs, f1_clf_fs)),
                         columns=header_names)
    df_fs.to_csv('{0}_{1}.csv'.format(args.classifier, args.feature))

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
