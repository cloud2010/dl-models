# -*- coding: utf-8 -*-
"""
A PCA implementation of unsupervised dimensionality reduction
Author: liumin@shmtu.edu.cn
Reference: `https://scikit-learn.org/stable/modules/unsupervised_reduction.html`
Date: 2019-09-22
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import argparse
import time


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='A PCA implementation of unsupervised dimensionality reduction')
    parser.add_argument('--datapath', required=True, help="Path of dataset")
    parser.add_argument('-n', type=int, default=10,
                        help="Dimension of the embedded space, int(default: 10), optional")

    # 参数初始化
    args = parser.parse_args()
    # 输出参数信息
    print("\n", args)
    # 读取数据集
    DATASETS = os.path.abspath(args.datapath)
    df = pd.read_csv(DATASETS)
    # 第1列为分类标注名称
    labels = df.values[:, 0].astype(np.int32)
    # 第2列之后为特征矩阵
    X = df.values[:, 1:].astype(np.float)
    print('\nOriginal features matrix shape', X.shape)
    # 非监督降维后的特征矩阵
    clf = PCA(n_components=args.n)
    X_embedded = clf.fit_transform(X)
    print('\nNew features matrix shape after dimensionality reduction', X_embedded.shape)
    # 输出至 CSV 文件
    f_list = ["F" + str(i) for i in range(args.n)]
    df_e = pd.DataFrame(X_embedded, index=labels, columns=f_list)
    df_e.to_csv("embedded_f{0}_output.csv".format(args.n), index_label="Class")
    print("\nThe embedded features matrix is saved to 'embedded_f{0}_output.csv'".format(
        args.n))
    end_time = time.time()
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
