"""
We will show how to load Membrane data into numpy array.
"""
import os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    parser = ArgumentParser(description="This example will show how to load file data into numpy array.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inputfile", type=str,
                        help="The filename of training dataset.", required=True)
    args = parser.parse_args()
    print("\nThe file path:", args.inputfile)
    result = list()
    with open(args.inputfile, 'r', encoding='utf-8') as f:
        # 读取首行分类信息
        labels = f.readline().split()
        for line in f.readlines():
            line = line.strip()
            # 过滤首行分类信息和文件末尾空行
            if len(line) > 0 and not line.startswith('>'):
                # 逐行存入矩阵信息
                result.append(line.split())
    # python list convert to numpy array
    mat = np.array(result, dtype=np.int16)
    print("\nMembrane name: {0}, Class: {1}".format(
        labels[0].lstrip('>'), labels[1]))
    print("\nMatrix shape:", mat.shape)
    print("\nFirst line:", mat[1])
