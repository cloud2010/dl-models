# -*- coding: utf-8 -*-
"""
In this example, we show how to retrieve:

- the binary tree structure;
- the depth of each node and whether or not it's a leaf;
- the nodes that were reached by a sample using the ``decision_path`` method;
- the leaf that was reached by a sample using the apply method;
- the rules that were used to predict a sample;
- the decision path shared by a group of samples.

Date: 2019-07-14
Ref: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description='This program is show how to predict sample by rules generated from `DecisionTreeClassifier`.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--randomseed', type=int,
                        help='The seed of the pseudo random number generator used when shuffling the data for probability estimates.', default=0)
    parser.add_argument('--datapath', type=str, help='The path of dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    from sklearn.tree import DecisionTreeClassifier

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

    # 初始化 classifier 并完成数据集训练
    clf = DecisionTreeClassifier(random_state=args.randomseed).fit(X_train, y_train)
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
    for i in range(len(d_uniques)):
        count_max_idx = np.argmax(d_counts)
        rule = d_uniques[count_max_idx]  # 获取通过次数最多的路径
        print("\nRules_{0}, passed counts:{1}".format(i, d_counts.max()))
        for node_id in range(len(rule)):
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
            if rule[node_id] == 1 and threshold[node_id] == -2:
                print('Class:', np.argmax(clf.tree_.value[node_id]))
        d_counts = np.delete(d_counts, count_max_idx, 0)  # 剔除刚刚最大值元素
        d_uniques = np.delete(d_uniques, count_max_idx, 0)
        # i = i+1

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = clf.apply(X_train)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
    # sample_id = 10

    # Predicted Array
    # pred_vals = clf.predict(X_train)

    # for sample_id in range(y_train.size):
    #     node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
    #     print('\nRules used to predict sample %s: ' % sample_id)
    #     for node_id in node_index:
    #         if leave_id[sample_id] == node_id:
    #             continue

    #         if (X_train[sample_id, feature[node_id]] <= threshold[node_id]):
    #             threshold_sign = "<="
    #         else:
    #             threshold_sign = ">"

    #         print("decision id node %s : (X_train[%s, %s] (= %s) %s %s)"
    #               % (node_id,
    #                  sample_id,
    #                  feature[node_id],
    #                  X_train[sample_id, feature[node_id]],
    #                  threshold_sign,
    #                  threshold[node_id]))
    #     print('Predicted class: %s' % pred_vals[sample_id])
    #     print('Correct class: %s' % y_train[sample_id])

    # print('\nTrain ACC: %.6f' % accuracy_score(y_train, pred_vals))
    # print('\nTrain MCC: %.6f' % matthews_corrcoef(y_train, pred_vals))

    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]\n'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
