# -*- coding: utf-8 -*-
"""
The implementation is based on ThunderSVM for predict bio datasets with kfold cross-validation and SMOTE.
ThunderSVM exploits GPUs and multi-core CPUs to achieve high efficiency
Derived from: Xtra Computing Group
Project: https://github.com/Xtra-Computing/thundersvm
Ref: https://github.com/scikit-learn-contrib/imbalanced-learn
Date: 2018-12-25
"""
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from imblearn.over_sampling import SMOTE

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="The implementation is based on ThunderSVM for predict bio datasets with kfold cross-validation and SMOTE.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--kernel", type=str,
                        help="Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.", default='rbf')
    parser.add_argument("-g", "--gamma", type=str,
                        help="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.", default='auto')
    parser.add_argument("-c", help="Penalty parameter C of the error term.", type=float, default=1)
    parser.add_argument("-e", "--max_iter", type=int,
                        help="Hard limit on iterations within solver, or -1 for no limit.", default=-1)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="The seed of the pseudo random number generator used when shuffling the data for probability estimates.", default=None)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)

    args = parser.parse_args()

    # 导入相关库
    from sklearn.datasets import load_svmlight_file
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    from thundersvm.thundersvmScikit import SVC
    from thundersvm.utils import model_evaluation, bi_model_evaluation

    # 读取数据
    data = load_svmlight_file(args.datapath)

    # 设定分类信息和特征矩阵
    X, y = data[0], data[1]
    print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    # 不同 Class 统计 (根据 Target 列)
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # Apply SMOTE 生成 fake data
    sm = SMOTE(k_neighbors=2)
    x_resampled, y_resampled = sm.fit_sample(X, y)
    # after over sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled.astype(int), return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['Class', 'Sum'])
    print("\nNumber of samples after over sampleing:\n{0}".format(df_resampled_y))

    # 初始化 classifier
    clf = SVC(kernel=args.kernel, gamma=args.gamma, C=args.c,
              max_iter=args.max_iter, random_state=args.randomseed)
    print("\nClassifier parameters:")
    print(clf.get_params())
    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.randomseed)
    # 生成 k-fold 训练集、测试集索引
    resampled_index_set = rs.split(y_resampled)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    # 暂存每次测试集特征矩阵
    # x_resampled_cache = np.ones([1, X.shape[1]])
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in resampled_index_set:
        print("\nFold:", k_fold_step)
        clf.fit(x_resampled[train_index], y_resampled[train_index])
        # 测试集验证
        # 验证测试集 (通过 index 去除 fake data)
        real_test_index = test_index[test_index < X.shape[0]]
        batch_test_x = x_resampled[real_test_index]
        batch_test_y = y_resampled[real_test_index]
        batch_test_size = len(real_test_index)
        y_pred = clf.predict(batch_test_x)
        # 计算测试集 ACC
        accTest = accuracy_score(batch_test_y, y_pred)
        print("\nFold:", k_fold_step, "Test Accuracy:",
              "{:.6f}".format(accTest), "Test Size:", batch_test_size)
        # 暂存每次选中的测试集和预测结果
        test_cache = np.concatenate((test_cache, batch_test_y))
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
