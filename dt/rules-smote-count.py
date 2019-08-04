# -*- coding: utf-8 -*-
"""
In this example, we show how to retrieve:

- the binary tree structure;
- the depth of each node and whether or not it's a leaf;
- the nodes that were reached by a sample using the ``decision_path`` method;
- the leaf that was reached by a sample using the apply method;
- the rules that were used to predict a sample;
- the decision path shared by a group of samples.

Date: 2019-08-04
Ref: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description='This program is show rules count generated from `DecisionTreeClassifier` with SMOTE.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--randomseed', type=int,
                        help='The seed of the pseudo random number generator used when shuffling the data for probability estimates.', default=0)
    parser.add_argument('--datapath', type=str, help='The path of dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from imblearn.over_sampling import SMOTE

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X_train = df.iloc[:, 1:].values
    y_train = df.iloc[:, 0].values - 1
    f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print('\nTraining dataset shape: ', X_train.shape, ' Number of features: ', X_train.shape[1])
    num_categories = np.unique(y_train).size
    sum_y = np.asarray(np.unique(y_train.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)
    # Apply SMOTE 生成 fake data
    sm = SMOTE(k_neighbors=2)
    x_resampled, y_resampled = sm.fit_sample(X_train, y_train)
    # after over sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled.astype(int), return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['Class', 'Sum'])
    print("\nNumber of samples after over sampleing:\n{0}".format(df_resampled_y))

    # 初始化 classifier 并完成数据集训练
    clf = DecisionTreeClassifier(random_state=args.randomseed).fit(x_resampled, y_resampled)
    print('\nClassifier parameters:\n')
    print(clf.get_params())

    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #

    # Using those arrays, we can parse the tree structure:
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = clf.decision_path(X_train)
    # 获取训练集规则路径
    d_paths = node_indicator.todense()
    # 规则路径去重并统计对应路径的出现次数
    d_uniques, d_idxs, d_counts, = np.unique(d_paths, axis=0, return_counts=True, return_index=True)
    # 规则打印
    print('\nThe most precise rules are the following:')
    # i = 0
    # for rule in d_uniques:
    for i, item in enumerate(d_uniques):
        count_max_idx = np.argmax(d_counts)
        rule = d_uniques[count_max_idx]  # 获取通过次数最多的路径
        print("\nRules_{0}, passed counts:{1}".format(i, d_counts.max()))
        for node_id, node_item in enumerate(rule):
            if rule[node_id] == 1 and threshold[node_id] != -2:
                if rule[node_id+1] == 0:
                    threshold_sign = '>'
                else:
                    threshold_sign = '<='
                print("node_{0}: feature_name={2}, feature_id[{1}].value {3} threshold={4}".format(
                    node_id,
                    feature[node_id],
                    f_names[feature[node_id]],
                    threshold_sign,
                    threshold[node_id]))
            if rule[node_id] == 1 and threshold[node_id] == -2:s
                print('Class:', np.argmax(clf.tree_.value[node_id]))
        d_counts = np.delete(d_counts, count_max_idx, 0)  # 剔除刚刚最大值元素
        d_uniques = np.delete(d_uniques, count_max_idx, 0)

    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]\n'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
