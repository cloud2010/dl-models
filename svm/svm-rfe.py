# coding: utf-8
"""
This implementation uses SVM-RFE to sort features.
Date: 2019-12-25
Ref: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection
"""
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

__author__ = 'Min'

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description='This program is used to generate datasets after SMOTE operation.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str,
                        help='The path of input dataset.', required=True)
    parser.add_argument('--output', type=str,
                        help='The path of output dataset.', required=True)

    args = parser.parse_args()

    # 读取数据
    df = pd.read_csv(args.input)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    # 不同 Class 统计 (根据 Target 列)
    print('\nInput Dataset shape: ', X.shape,
          ' Number of features: ', X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)
    # 特征名称
    f_name = df.columns[1:].values

    # 构建分类器
    clf_svm = SVC(kernel='linear')
    # 构建 RFE 特征选择器
    selector = RFE(clf_svm, 1, step=1)
    # 训练
    selector.fit(X, y)
    # 获取重要特征的索引位置的排序
    f_idxs = np.argsort(selector.ranking_)
    # 获取对应的特征名称
    # 构建新的列名称向量
    new_f_name = np.insert(f_name[f_idxs[0:]], 0, 'class')
    # pandas 根据特征名称重新排序训练集
    new_df = df[new_f_name]
    # 输出特征排序后结果
    new_df.to_csv('{0}'.format(args.output), index=None)
    print('\nSorted features dataset saved to:`{0}`'.format(args.output))
    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
