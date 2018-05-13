# -*- coding: utf-8 -*-
"""
A sklearn implementation of T-SNE
Author: liumin@shmtu.edu.cn
Reference: `http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html`
"""
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import os
import argparse
import time


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='T-SNE for citrullination')
    parser.add_argument('--datapath', required=True, help="Path of dataset")
    parser.add_argument('-e', '--embedded', type=int, default=3,
                        help="Dimension of the embedded space, int(default: 3), optional")
    parser.add_argument('-i', '--niter', type=int, default=1000,
                        help="Maximum number of iterations for the optimization. Should be at least 250, int(default: 1000), optional")

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
    features = df.values[:, 1:].astype(np.float)
    print('\nOriginal features matrix shape', features.shape)
    # T-SNE 降维后的特征矩阵
    features_embedded = TSNE(n_components=args.embedded, n_iter=args.niter).fit_transform(features)
    print('\nNew features matrix shape after dimensionality reduction', features_embedded.shape)
    # 输出至 CSV 文件
    f_list = ["F" + str(i) for i in range(args.embedded)]
    df = pd.DataFrame(features_embedded, index=labels, columns=f_list)
    df.to_csv("embedded_f{0}_output.csv".format(args.embedded), index_label="Class")
    print("\nThe embedded features matrix is saved to 'embedded_f{0}_output.csv'".format(
        args.embedded))
    end_time = time.time()
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
