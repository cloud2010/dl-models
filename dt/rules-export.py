# -*- coding: utf-8 -*-
"""
This program is implementation of rules export as text.
Date: 2019-06-17
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.tree import _tree

__author__ = 'Min'


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    pathto = dict()

    global k
    k = 0

    def recurse(node, depth, parent):
        global k
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            s = "{} <= {} ".format(name, threshold, node)
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent]+' & ' + s

            recurse(tree_.children_left[node], depth + 1, node)
            s = "{} > {}".format(name, threshold)
            if node == 0:
                pathto[node] = s
            else:
                pathto[node] = pathto[parent]+' & ' + s
            recurse(tree_.children_right[node], depth + 1, node)
        else:
            k = k+1
            print('\n')
            print(k, ')', pathto[parent], '==> Class', np.argmax(tree_.value[node]))
    recurse(0, 1, 0)


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description='This program is implementation of rules export as text.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--randomseed', type=int,
                        help='The seed of the pseudo random number generator used when shuffling the data for probability estimates.', default=0)
    # parser.add_argument('-n', '--nestimators', type=int,
    #                     help='The number of base estimators (rules) to use for prediction. More are built before selection.', default=10)
    parser.add_argument('--datapath', type=str, help='The path of dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    # from skrules import SkopeRules

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

    # 初始化 classifier 并完成数据集训练
    clf = DecisionTreeClassifier().fit(X, y)
    # clf = SkopeRules(n_jobs=-1, n_estimators=args.nestimators, feature_names=f_names).fit(X, y)
    print('\nClassifier parameters:\n')
    print(clf.get_params())

    # 输出分类规则
    print('\nThe most precise rules are the following:')
    tree_to_code(clf, f_names.tolist())
    # r = tree.export_text(clf, feature_names=f_names.tolist())
    # print('\nThe most precise rules are the following:\n')
    # for rule in clf.rules_:
    #     print(rule[0])
    # print('\nThe number of generated rules: ', len(clf.rules_))
    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]\n'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
