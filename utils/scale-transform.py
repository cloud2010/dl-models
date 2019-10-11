# -*- coding: utf-8 -*-
"""
This program is used to standardize a dataset.
"""
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

if __name__ == "__main__":
    START_TIME = time.time()
    parser = ArgumentParser(description="This program is used to standardize a dataset.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str, help="length of the number.", required=True)
    args = parser.parse_args()

    # 读取数据
    df = pd.read_csv(args.datapath)
    X = df.iloc[:, 1:].values
    print("\nDataset shape: ", X.shape, " Number of features: ", X.shape[1])
    # 标准化操作
    X_new = scale(X)

    # 列合并
    df_s = pd.concat([df.iloc[:, 0], pd.DataFrame(data=X_new)], axis=1)

    df_s.to_csv('standardization_dataset.csv', index=None, header=df.columns)
    print("\nSaved to 'standardization_dataset.csv'")
    END_TIME = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((END_TIME - START_TIME) / 60), (END_TIME - START_TIME)))
