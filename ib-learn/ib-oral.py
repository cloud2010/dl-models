
# -*- coding: utf-8 -*-
# Authors: Min Liu
#          Min Liu <liumin@shmtu.edu.cn>
# License: MIT
"""
SVM classification with SMOTE + ENN using scikit-learn + imbalanced-learn
"""

import numpy as np
import pandas as pd
# from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
# from collections import Counter
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="SVM classification with SMOTE + ENN using scikit-learn + imbalanced-learn",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-data", type=str, help="The path of CSV dataset.", required=True)
    args = parser.parse_args()
    # 读取数据集
    df = pd.read_csv(args.data, encoding='utf8')

    y = df.iloc[:, 0].values  # 读取第一列 label 信息
    X = df.iloc[:, 1:].values  # 读取特征矩阵数据

    assert(len(X[:, 1]) == np.count_nonzero(y))  # 确保数据样本数一致

    print('\nOriginal dataset summary:')
    print('\nClass 1: ', (y == 1).sum())  # 输出正负样本数
    print('\nClass 2: ', (y == 2).sum())

    # Apply SMOTE + ENN
    sm = SMOTEENN()
    X_resampled, y_resampled = sm.fit_sample(X, y)

    # print((y_resampled == 2).sum())

    # 输出原始样本分布和生成 fake data 后的样本分布
    # print('\nAfter apply SMOTE + ENN: ', Counter(y_resampled).items())
    print('\nAfter apply SMOTE + ENN:')
    print('\nClass 1: ', (y_resampled == 1).sum())  # 输出正负样本数
    print('\nClass 2: ', (y_resampled == 2).sum())

    # Using SVM as a classifier

    from sklearn import svm

    clf = svm.SVC()  # 构建分类器

    print('\nUsing SVM as a classifier and start training resampled dataset...')
    clf.fit(X_resampled, y_resampled)  # 训练 resampled data
    print('\nOptimization Finished.')
    print('\nStart predicting original dataset...')
    y_pred = clf.predict(X)  # 预测原始数据集

    from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report

    # Model evaluation
    print("\n=== Model evaluation ===")
    print(classification_report(y, y_pred, digits=6))
    print("\n=== Accuracy classification score ===")
    print("\nACC = {:.6f}".format(accuracy_score(y, y_pred)))
    print("\n=== Matthews Correlation Coefficient ===")
    print("\nMCC = {:.6f}".format(matthews_corrcoef(y, y_pred)))
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]\n".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
