# -*- coding: utf-8 -*-
"""
Convert the singal peptide dataset to CSV format
"""
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.utils import resample
import numpy as np
import pandas as pd

__author__ = "Min"

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="Convert the bio dataset you input to CSV format",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str,
                        help="The full path of training dataset.", required=True)
    parser.add_argument("--ratio", type=int,
                        help="The ratio of negative sample to positive sample.", required=True)
    args = parser.parse_args()
    print("\nThe dataset path:", args.datapath)

    # 步长
    step = 0
    sample_mat = np.array([], dtype=np.int16)
    N_MAT = np.array([], dtype=np.int16)
    P_MAT = np.array([], dtype=np.int16)
    N_FILE = 'signal_peptide_negative_samples_0-1.txt'
    P_FILE = 'signal_peptide_positive_samples_0-1.txt'
    # 先读取所有负样本信息
    samples, labels = list(), list()
    filepath = os.path.join(args.datapath, N_FILE)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) > 0 and step < 20:
                items = line.strip().split()
                if items[0][0:1] == '>':
                    # 读取首行分类信息
                    samples.append(items[0].strip('>'))
                    labels.append(items[1])
                else:
                    # 存储单个样本的矩阵信息
                    sample_mat = np.append(sample_mat, line.split())
                    step = step + 1
            else:
                # 读完一个样本步长归零
                step = 0
                # 总矩阵存储样本矩阵信息
                # print(sample_mat.shape)
                N_MAT = np.concatenate((N_MAT, sample_mat))
                # 清空单样本临时矩阵
                sample_mat = np.array([], dtype=np.int16)

    # 负样本全体矩阵
    N_MAT = N_MAT[np.newaxis].reshape((len(labels), -1))

    # 负样本按比例抽样 1倍正样本 2倍正样本 3倍正样本
    sample_index = resample(np.arange(len(labels)),
                            replace=False, n_samples=(args.ratio * 2683))

    N_MAT = N_MAT[sample_index]
    labels = np.array(labels)[sample_index]
    samples = np.array(samples)[sample_index]

    # 特征编号
    f_list = ["f" + str(i) for i in range(1, 401)]
    # 将所有负样本矩阵写入DataFrame
    n_df = pd.DataFrame(N_MAT, index=labels, columns=f_list)
    n_df.insert(0, "Samples", samples)

    # 再读取所有正样本信息
    samples, labels = list(), list()  # 重置
    filepath = os.path.join(args.datapath, P_FILE)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line) > 0 and step < 20:
                items = line.strip().split()
                if items[0][0:1] == '>':
                    # 读取首行分类信息
                    samples.append(items[0].strip('>'))
                    labels.append(items[1])
                else:
                    # 存储单个样本的矩阵信息
                    sample_mat = np.append(sample_mat, line.split())
                    step = step + 1
            else:
                # 读完一个样本步长归零
                step = 0
                # 总矩阵存储样本矩阵信息
                # print(sample_mat.shape)
                P_MAT = np.concatenate((P_MAT, sample_mat))
                # 清空单样本临时矩阵
                sample_mat = np.array([], dtype=np.int16)

    # 正样本全体矩阵
    P_MAT = P_MAT[np.newaxis].reshape((len(labels), -1))

    # print("\nLabels:", labels)
    # print("\nSamples:", samples)
    print("\nP_Mat:", P_MAT.shape, "N_Mat: ", N_MAT.shape)

    # 写入数据库输出成CSV格式
    p_df = pd.DataFrame(P_MAT, index=labels, columns=f_list)
    p_df.insert(0, "Samples", samples)

    # 合并正负样本矩阵
    df = n_df.append(p_df)

    # 将正负样本写入CSV
    df.to_csv(
        "D://peptide_sample_{0}.csv".format(args.ratio), index_label="Class")

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
