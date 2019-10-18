# -*- coding: utf-8 -*-
"""
A knn classifier with SMOTE based on scikit-learn apply it to classify bio datasets.
Date: 2019-10-17
"""
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="A knn classifier with SMOTE based on scikit-learn apply it to classify bio datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-n", "--nneighbors", type=int,
                        help="Number of neighbors to use by default for :meth:`kneighbors` queries.", default=5)
    parser.add_argument("-w", "--weights", type=str,
                        help="weight function used in prediction.  Possible values:'uniform' or 'distance'.", default="uniform")
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
    from sklearn import neighbors
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
    from lgb.utils import model_evaluation, bi_model_evaluation
    from imblearn.over_sampling import SMOTE

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

    # Apply SMOTE 生成 fake data
    sm = SMOTE(k_neighbors=2)
    x_resampled, y_resampled = sm.fit_sample(X, y)
    # after over sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled, return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['Class', 'Sum'])
    print("\nNumber of samples after over sampleing:\n{0}\n".format(df_resampled_y))

    # 初始化 classifier
    clf = neighbors.KNeighborsClassifier(
        n_jobs=-1, n_neighbors=args.nneighbors, weights=args.weights)
    print("\nClassifier parameters:")
    print(clf.get_params())
    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.randomseed)
    # 生成 k-fold 训练集、测试集索引
    resampled_index_set = rs.split(y_resampled)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in resampled_index_set:
        print("\nFold:", k_fold_step)
        batch_x = x_resampled[train_index]  # 特征数据用于训练
        batch_y = y_resampled[train_index]  # 标记结果用于验证
        batch_size = train_index.shape[0]
        clf.fit(batch_x, batch_y)
        # 验证测试集 (通过 index 去除 fake data)
        real_test_index = test_index[test_index < num_samples]
        batch_test_x = x_resampled[real_test_index]
        batch_test_y = y_resampled[real_test_index]
        batch_test_size = len(real_test_index)
        # 测试集验证
        y_pred = clf.predict(batch_test_x)
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
