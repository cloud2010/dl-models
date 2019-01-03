# -*- coding: utf-8 -*-
"""
This a implementation of the rulefit algorithm for predict bio datasets with kfold cross-validation.
Derived from: Christoph Molnar
Source: https://github.com/christophM/rulefit
Date: 2019-01-03
"""
import os
import time
import numpy as np
import pandas as pd
from multiprocessing import current_process, Pool
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from rulefit import RuleFit
from rulefit import utils

__author__ = 'Min'


def train_job(train_idx, test_idx, t_size, rf_mode, m_rules, r_seed, X, y, feas):
    """
    每个 fold 中进行训练和验证
    """
    p = current_process()
    print('process counter:', p._identity[0], 'pid:', os.getpid())
    # 初始化 estimator 训练集进入模型
    rf = RuleFit(tree_size=t_size, rfmode=rf_mode,
                 max_rules=m_rules, random_state=r_seed)
    print("\nTree generator:{0}, \n\nMax rules:{1}, Tree size:{2}, Random state:{3}".format(
        rf.tree_generator, rf.max_rules, rf.tree_size, rf.random_state))
    rf.fit(X[train_idx], y[train_idx], feas)
    # 验证测试集
    batch_test_x = X[test_idx]
    batch_test_y = y[test_idx]
    batch_test_size = len(test_idx)
    y_pred = rf.predict(batch_test_x)
    # 计算测试集 ACC
    accTest = accuracy_score(batch_test_y, y_pred)
    print("\nTest Accuracy:", "{:.6f}".format(accTest), "Test Size:", batch_test_size)

    print("\n=========================================================================")
    # 返回测试集和预测结果用于统计
    return batch_test_y, y_pred


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a implementation of the rulefit algorithm for predict bio datasets with kfold cross-validation and SMOTE.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--treesize", type=int,
                        help="Number of terminal nodes in generated trees. If exp_rand_tree_size=True, this will be the mean number of terminal nodes.", default=4)
    parser.add_argument("-m", "--maxrules", type=int,
                        help="approximate total number of rules generated for fitting. Note that actual number of rules will usually be lower than this due to duplicates.", default=2000)
    parser.add_argument("--rfmode", type=str,
                        help="'regress' for regression or 'classify' for binary classification.", default='classify')
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=None)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)

    args = parser.parse_args()

    # 读取数据
    df = pd.read_csv(args.datapath)

    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    # 标准化处理
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    # 读取特征名称
    features = df.columns[1:]
    print("\nDataset shape: ", df.shape, " Number of features: ", features.size)
    # 不同 Class 统计 (根据 Target 列)
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # 交叉验证
    rs = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.randomseed)
    # 生成 k-fold 训练集、测试集索引
    cv_index_set = rs.split(y)
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)
    print("\nTraining Start...")
    # 构建进程池并行训练
    pool = Pool(processes=args.kfolds)
    res = []
    for train_index, test_index in cv_index_set:
        result = pool.apply_async(train_job, args=(train_index, test_index), kwds=dict(t_size=args.treesize, rf_mode=args.rfmode,
                                                                                       m_rules=args.maxrules, r_seed=args.randomseed, X=X, y=y, feas=features))
        res.append(result)
    pool.close()
    pool.join()
    for _ in res:
        """ 
        https://jingsam.github.io/2015/12/31/multiprocessing.html
        获取返回值的过程最好放在进程池回收之后进行，避免阻塞后面的语句
        汇总每次选中的测试集和预测结果
        """
        y_res = _.get()
        # real label
        test_cache = np.concatenate((test_cache, np.array(y_res[0])))
        # predicted label
        pred_cache = np.concatenate((pred_cache, np.array(y_res[1])))

    # 末尾输出rulefit模型参数
    # print("\n=== Model parameters ===")
    # 输出统计结果
    if(num_categories > 2):
        utils.model_evaluation(num_categories, test_cache, pred_cache)
    else:
        utils.bi_model_evaluation(test_cache, pred_cache)

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
