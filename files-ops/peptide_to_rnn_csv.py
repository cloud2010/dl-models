# -*- coding: utf-8 -*-
"""
Convert the singal peptide dataset to CSV format
"""
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd

__author__ = "Min"

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert the bio dataset you input to CSV format",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str,
                        help="The full path of training dataset.", required=True)
    args = parser.parse_args()
    print("\nThe dataset path:", args.datapath)

    # 获取文件夹下所有样本文件名
    files = os.listdir(args.datapath)
    labels = list()
    mat = np.array([], dtype=np.int16)
    # 迭代读取信息
    for filename in files:
        filepath = os.path.join(args.datapath, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                # 读取首行分类信息
                class_info = f.readline().split()
                labels.append(class_info[1])
                # 写入矩阵信息
                sample_mat = np.array([], dtype=np.int16)
                for line in f.readlines():
                    line = line.strip()
                    # 过滤空行
                    if len(line) > 0:
                        # 存储单个样本的矩阵信息
                        sample_mat = np.append(sample_mat, line.split())
        # 存储所有样本矩阵信息
        mat = np.concatenate((mat, sample_mat))

    print("\nLabels:", labels)
    # 将所有样本矩阵进行形状调整便于写入DataFrame
    mat = mat[np.newaxis].reshape((len(labels), -1))
    # 特征编号
    f_list = ["f" + str(i) for i in range(1, (sample_mat.shape[0] + 1))]
    # 写入数据库输出成CSV格式
    df = pd.DataFrame(mat, index=labels, columns=f_list)
    df.insert(0, "Samples", files)
    df.to_csv("peptide_rnn.csv", index_label="Class")
    # print(sample_mat.shape)
    # print(mat.shape)
    # print(files[2])
    # print(mat[2])
