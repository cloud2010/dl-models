# -*- coding: utf-8 -*-
"""
This program is used to generate random numbers.
"""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

if __name__ == "__main__":
    parser = ArgumentParser(description="This program is used to generate random numbers.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num", type=int, help="length of the number.", required=True)
    parser.add_argument("-k", type=int, help="times of shuffle.", required=True)
    args = parser.parse_args()

    # 生成对应长度的序列
    x = np.arange(1, args.num+1)
    # k 行 num 列
    x_tmp = np.empty((args.k, args.num))
    # df = pd.DataFrame()
    for i in range(args.k):
        x_tmp[i] = shuffle(x, random_state=i)
    df = pd.DataFrame(data=x_tmp, index=None, columns=None)
    df.to_csv('shuffled_num{0}_k{1}.csv'.format(args.num, args.k), index=False, header=False)
