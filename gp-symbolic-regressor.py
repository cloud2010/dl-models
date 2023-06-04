# -*- coding: utf-8 -*-
"""
A Symbolic Regressor Example
Date: 2023-05-02
"""
# import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="A Symbolic Regressor Example",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)

    args = parser.parse_args()
    # logdir_base = os.getcwd()  # 获取当前目录

    # 导入相关库
    import numpy as np
    import pandas as pd
    from gplearn.genetic import SymbolicRegressor
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import mean_squared_error

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
    print('\n============================\n')

    # 初始化 classifier
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=args.randomseed)
    est_gp.fit(X, y)

    # 输出多项式结果
    print(f'\nThe fittest individuals: {est_gp._program}')

    end_time = time.time()  # 程序结束时间
    print(
        f'\n[Finished in: { ((end_time - start_time) / 60):.3f} mins = {(end_time - start_time):.3f} seconds]')
