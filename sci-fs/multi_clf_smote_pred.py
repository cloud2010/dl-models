# -*- coding: utf-8 -*-
"""
Multiple classifier integrates the feature sorting algorithm with SMOTE based on scikit-learn for prediction and output.
Date: 2020-08-11
"""
import os
import sys
import time
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="Multiple classifier integrates the feature sorting algorithm based on scikit-learn for prediction and output.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--train", type=str,
                        help="The path of training dataset.", required=True)
    parser.add_argument("--test", type=str,
                        help="The path of test dataset.", required=True)
    parser.add_argument("--clf", type=str,
                        help="The classifier. (svm / rf / knn)", default='svm')

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from imblearn.over_sampling import SMOTE
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_recall_fscore_support

    # 读取数据
    df = pd.read_csv(args.train)
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
    # 综合采样
    smote = SMOTE(k_neighbors=2)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("\n", smote)
    # 排序前后正负样本分布
    print("\nAfter SMOTE", sorted(Counter(y_resampled).items()))

    # 初始化 classifier 和预测结果字典
    clf_dict = {
        'rf': RandomForestClassifier(n_jobs=-1, random_state=args.randomseed),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'svm': SVC(kernel='rbf', random_state=args.randomseed)
    }

    y_pred_dict = dict().fromkeys(clf_dict.keys(), np.array([]))

    print('\nStarting training...')
    # 训练
    model = clf_dict[args.clf].fit(X_resampled, y_resampled)
    print('\n', model)
    print('\n', model.get_params())
    # 预测测试集
    print('\nStart predicting...')
    # 读取测试集数据
    df_test = pd.read_csv(args.test)
    # 设定分类信息和特征矩阵
    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values
    y_pred = model.predict(X_test)

    # 进行测试集模型评估
    from utils import model_evaluation, bi_model_evaluation
    if(num_categories > 2):
        model_evaluation(num_categories, y_test, y_pred)
    else:
        bi_model_evaluation(y_test, y_pred)

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
