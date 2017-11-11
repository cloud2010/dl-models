"""
We will show how to load training data into numpy array.
"""
import os
import random
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.preprocessing import StandardScaler
__author__ = 'Min'


def get_datasets(dataset_path, rate):
    """读取样本数据集(正负两类)

    Args:
     dataset_path: 数据集路径
     rate: 负样本比例(一倍或两倍)

    Returns:
     返回样本文件列表及对应分类列表
    """
    sample_paths, sample_labels = list(), list()
    # 获取文件夹下所有正负样本文件名
    files_p = os.listdir(os.path.join(dataset_path, "P"))
    # 从全部负样本中选取一定数量做计算
    files_n_all = os.listdir(os.path.join(dataset_path, "N"))
    files_n = random.sample(files_n_all, int(rate) * len(files_p))
    # 先获取所有正样本
    for filename in files_p:
        filepath = os.path.join(dataset_path, "P", filename)
        if os.path.isfile(filepath):
            # 登记相关信息至训练集列表
            sample_paths.append(filepath)
            # 正样本分类为1
            sample_labels.append(1)
    # 再获取抽样的负样本
    for filename in files_n:
        filepath = os.path.join(dataset_path, "N", filename)
        if os.path.isfile(filepath):
            # 登记相关信息至训练集列表
            sample_paths.append(filepath)
            # 负样本分类为0
            sample_labels.append(0)
    # 返回正负样本数据集
    return sample_paths, sample_labels


if __name__ == "__main__":
    parser = ArgumentParser(description="This example will show how to load file data into numpy array.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str,
                        help="The path of training dataset.", required=True)
    args = parser.parse_args()
    print("\nThe dataset path:", args.datapath)

    paths, labels = get_datasets(args.datapath, 1)

    print(len(paths))
    print(len(labels))
