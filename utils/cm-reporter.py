"""Report metrics from a whitespace-delimited true/predicted label file."""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.metrics import model_evaluation

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This program is used to output the evaluation info of the model.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="The path of training log file.", required=True)

    args = parser.parse_args()

    # '\s+' 匹配任意空白字符串
    df = pd.read_csv(args.input, sep=r'\s+', header=None)
    y_true = df.iloc[:, 0].values  # 第一列为真实值
    y_pred = df.iloc[:, 1].values  # 第二列为预测值
    num_categories = np.unique(y_true).size

    # 输出评估结果
    model_evaluation(num_categories, y_true, y_pred)

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(((end_time - start_time) / 60), (end_time - start_time)))
