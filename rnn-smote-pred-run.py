# -*- coding: utf-8 -*-
"""
This a Basic RNN implementation with SMOTE for training and predict bio datasets.
Date: 2019-01-31
"""
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'

if __name__ == "__main__":
    # 设置 GPU 环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid  # 修改: 移到参数解析之前
    parser = ArgumentParser(description="This a Basic RNN implementation with SMOTE for training and predict bio datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=20)
    parser.add_argument("-u", "--nunits", type=int,
                        help="Number of hidden layer units. (-1: use input size)", default=-1)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-f", "--fragment", type=int,
                        help="Specifying the `length` of sequences fragment.", default=1)
    parser.add_argument("--train", type=str,
                        help="The path of training dataset.", required=True)
    parser.add_argument("--test", type=str,
                        help="The path of test dataset.", required=True)
    parser.add_argument("--learningrate", type=float,
                        help="Learning rate.", default=1e-2)
    parser.add_argument("-g", "--gpuid", type=str,
                        help='GPU to use (leave blank for CPU only)', default="")

    args = parser.parse_args()

    start_time = time.time()  # 程序开始时间

    # 输出 RNN 模型相关训练参数
    print("\nRNN HyperParameters:")
    print("\nEpochs:{0}, Learning rate:{1}, Sequences fragment length: {2}, Hidden Units:{3}".format(
        args.epochs, args.learningrate, args.fragment, args.nunits))
    print("\nRandom seed is", args.randomseed)
    # 输出 GPU 相关信息
    print("\nGPU to use:", "No GPU support" if not args.gpuid else f"/gpu:{args.gpuid}")  # 修改: 优化日志输出
    print("\nTraining Start...")

    # 执行 RNN 训练模型并验证
    from rnn.rnn_smote_pred import run
    run(args.train, args.test, args.nunits, args.fragment,
        args.epochs, args.learningrate, args.randomseed)

    end_time = time.time()  # 程序结束时间
    # 打印程序运行时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
