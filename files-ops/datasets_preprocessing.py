"""
We will show how to preprocessing Membrane data.
"""
import os
import numpy as np
from shutil import move
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    parser = ArgumentParser(description="Membrane datasets preprocessing.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str,
                        help="The path of Membrane dataset.", required=True)
    args = parser.parse_args()
    print("\nThe datasets path:", args.datapath)
    files = os.listdir(args.datapath)
    for filename in files:
        filepath = os.path.join(args.datapath, filename)  #  Membrane 文件完整路径
        if os.path.isfile(filepath):
            f = open(filepath, 'r', encoding='utf-8')
            # 读取样本分类信息
            labels = f.readline().split()
            f.close()
            # 移动各文件至对应分类目录
            if labels[1] == '1':
                move(filepath, os.path.join(args.datapath, 'C1', filename))
            if labels[1] == '2':
                move(filepath, os.path.join(args.datapath, 'C2', filename))
            if labels[1] == '3':
                move(filepath, os.path.join(args.datapath, 'C3', filename))
            if labels[1] == '4':
                move(filepath, os.path.join(args.datapath, 'C4', filename))
            if labels[1] == '5':
                move(filepath, os.path.join(args.datapath, 'C5', filename))
