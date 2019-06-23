# -*- coding: utf-8 -*-
"""
In this example, we show how to retrieve:

- the binary tree structure;
- the depth of each node and whether or not it's a leaf;
- the nodes that were reached by a sample using the ``decision_path`` method;
- the leaf that was reached by a sample using the apply method;
- the rules that were used to predict a sample;
- the decision path shared by a group of samples.

Date: 2019-06-17
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
    parser.add_argument('-t', '--testsize', type=float,
                        help='Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.', default=0.25)
    parser.add_argument('--datapath', type=str, help='The path of dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    from sklearn.tree import DecisionTreeClassifier

    # 读取数据
    df = pd.read_csv(args.datapath)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values - 1
    f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print('\nDataset shape: ', X.shape, ' Number of features: ', X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # 原始数据集拆分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.randomseed, test_size=args.testsize)

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

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("\nThe decision tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node => Class=%d." % (node_depth[i] * "\t", i, np.argmax(clf.tree_.value[i])))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()

    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = clf.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = clf.apply(X_test)

    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
    # sample_id = 10

    # Predicted Array
    pred_vals = clf.predict(X_test)

    print('\nTest dataset size:', len(y_test))

    for sample_id in range(y_test.size):
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]
        print('\nRules used to predict sample %s: ' % sample_id)
        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                continue

            if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
                  % (node_id,
                     sample_id,
                     feature[node_id],
                     X_test[sample_id, feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))
        print('Predicted class: %s' % pred_vals[sample_id])
        print('Correct class: %s' % y_test[sample_id])

    print('\nTest ACC: %.6f' % accuracy_score(y_test, pred_vals))
    print('\nTest MCC: %.6f' % matthews_corrcoef(y_test, pred_vals))

    # For a group of samples, we have the following common node.
    # sample_ids = [0, 1, 2, 3]
    # common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids))
    # common_node_id = np.arange(n_nodes)[common_nodes]
    # print("\nThe following samples %s share the node %s in the tree" % (sample_ids, common_node_id))
    # print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))

    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]\n'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
