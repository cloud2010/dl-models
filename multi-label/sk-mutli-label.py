# -*- coding: utf-8 -*-
"""
A implementation of scikit-learn based module for multi-label classification.
Date: 2019-10-05
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="A scikit-learn based module for multi-label classification.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-k", "--kfolds", type=int, help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-c", "--nclass", type=int,
                        help="Number of class. Must be at least 2 aka two-label.", default=2)
    parser.add_argument("--datapath", type=str, help="The path of dataset.", required=True)
    parser.add_argument('--output', type=str, help='The path of output dataset.', required=True)
    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from scipy.io import arff
    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.svm import SVC
    # from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import KFold

    # 读取数据
    train_data, train_meta = arff.loadarff(args.datapath)  # 目前多标签只支持arff格式
    df = pd.DataFrame(data=train_data)
    # 读取特征矩阵
    X_train = df.iloc[:, args.nclass:].values
    # 读取多标签数据
    y_train = df.iloc[:, 0:args.nclass].values.astype(np.int)
    # 不同 Class 统计 (根据 Target 列)
    print("\nDataset shape: ", X_train.shape, " Label shape: ", y_train.shape[1])

    # 初始化 classifier
    clf = BinaryRelevance(classifier=SVC(gamma='scale'), require_dense=[False, True])
    print("\nClassifier parameters:")
    print(clf.get_params())
    # y_pred = cross_val_predict(clf, X_train, y_train, cv=args.kfolds, n_jobs=-1, verbose=1)
    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.randomseed)
    # 生成 k-fold 训练集、测试集索引
    cv_index_set = rs.split(X_train)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集索引和对应预测结果
    test_idx_cache = np.array([], dtype=np.int)
    pred_cache = np.empty((1, args.nclass))
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in cv_index_set:
        clf.fit(X_train[train_index], y_train[train_index])
        # 测试集验证
        y_pred = clf.predict(X_train[test_index])
        # 暂存每次选中的测试集和预测结果
        test_idx_cache = np.concatenate((test_idx_cache, test_index))
        pred_cache = np.concatenate((pred_cache, y_pred.toarray()), axis=0)
        # 每个fold训练结束后次数 +1
        k_fold_step += 1
    # 删除第一行空白元素
    pred_cache = np.delete(pred_cache, 0, axis=0)
    # 输出交叉验证后的预测结果
    df_pred = pd.DataFrame(index=test_idx_cache, data=pred_cache, columns=df.columns.values[0:args.nclass])
    df_pred.to_csv('{0}-pred.csv'.format(args.output), index_label='Sample ID')
    print('\nThe prediction saved to:`{0}-pred.csv`'.format(args.output))
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
