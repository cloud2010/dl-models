# -*- coding: utf-8 -*-
"""
Get intersection from RWR and diffusr arrays.
"""
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="Get intersection from RWR and diffusr arrays.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", type=str, help="The path of RWR file.", required=True)
    parser.add_argument("-d", type=str, help="The path of diffusr file.", required=True)
    parser.add_argument("-k", type=int, help="Number of elements to calculation. Must be at least 2.",
                        required=True, default=100)
    args = parser.parse_args()

    df_rwr = pd.read_csv(args.r)
    # 根据热度进行降序排列
    df_diffusr = pd.read_csv(args.d).sort_values(by=['heat'], ascending=False)
    # 取前 k 个 diffusr 和 rwr 的元素
    diffusr_k = df_diffusr[df_diffusr['seed'] == 0].iloc[:args.k, 0].values
    rwr_k = df_rwr.columns[:args.k].values
    # 获取交集元素
    result = np.intersect1d(rwr_k, diffusr_k)
    if len(result) > 0:
        pd.DataFrame(data=result).T.to_csv('result.csv', index=False, header=False)
        print("\nThe intersection have been saved to 'result.csv'")
    else:
        print("\nNo intersection element was obtained")
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
