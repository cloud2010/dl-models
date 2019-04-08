# -*- coding: utf-8 -*-
"""
This program is used to generate datasets after undersampling operation.
Date: 2019-04-08
"""
# import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description='This program is used to generate datasets after undersampling operation.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, help='The path of input dataset.', required=True)
    parser.add_argument('--output', type=str, help='The path of output dataset.', required=True)

    args = parser.parse_args()

    # 导入相关库
    import numpy as np
    import pandas as pd
    from imblearn.under_sampling import ClusterCentroids

    # 读取数据
    df = pd.read_csv(args.input)
    # 设定分类信息和特征矩阵
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    # f_names = df.columns[1:].values
    # 不同 Class 统计 (根据 Target 列)
    print('\nInput Dataset shape: ', X.shape, ' Number of features: ', X.shape[1])
    num_categories = np.unique(y).size
    sum_y = np.asarray(np.unique(y.astype(int), return_counts=True))
    df_sum_y = pd.DataFrame(sum_y.T, columns=['Class', 'Sum'], index=None)
    print('\n', df_sum_y)

    # Apply Undersampling 生成 fake data
    cc = ClusterCentroids(random_state=0)
    x_resampled, y_resampled = cc.fit_sample(X, y)
    # after under sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled.astype(int), return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['Class', 'Sum'])
    print("\nNumber of samples after under sampling:\n{0}".format(df_resampled_y))
    # 合并新的特征矩阵
    resampled_data = np.column_stack((y_resampled, x_resampled))
    resampled_df = pd.DataFrame(index=None, data=resampled_data, columns=df.columns.values)
    # 输出 resampling 后的特征集
    resampled_df.to_csv('{0}'.format(args.output), index=None)
    print('\nNew datasets after undersampling saved to:`{0}`'.format(args.output))
    end_time = time.time()  # 程序结束时间
    print('\n[Finished in: {0:.6f} mins = {1:.6f} seconds]'.format(
        ((end_time - start_time) / 60), (end_time - start_time)))
