
# coding: utf-8
"""
Convert CSV format to libsvm data format
"""
import numpy as np
import pandas as pd
from sklearn.datasets import dump_svmlight_file
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="Convert CSV format to libsvm data format",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-data", type=str, help="The path of CSV dataset.", required=True)
    parser.add_argument("-output", type=str,
                        help="The path of converted libsvm dataset.", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding='utf8')   # 读取需要转换的数据集
    y = df.iloc[:, 0].values  # 读取第一列class分类信息
    x = df.iloc[0:, 1:].values  # 读取特征矩阵信息(去除第一列之后的数据)

    # 使用sklearn dump_svmlight_file 进行libsvm转换
    # ref:http://scikit-learn.org/stable/modules/generated/sklearn.datasets.dump_svmlight_file.html

    dump_svmlight_file(x, y, args.output)
    
    end_time = time.time()  # 程序结束时间
    print("\nThe converted libsvm dataset have been saved to '{0}'".format(args.output))
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]\n".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
