# -*- coding: utf-8 -*-
"""
The implementation is based on FNN for predict bio datasets with kfold cross-validation.
Author: Liu Min
Email: liumin@shmtu.edu.cn
Date: 2019-04-26
"""
# import sys
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a Basic FNN implementation using Pytorch v0.4.0+ and scikit-learn v0.19+.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-u", "--nunits", type=int,
                        help="Number of hidden layer units.", default=1024)
    parser.add_argument("-n", "--nlayers", type=int,
                        help="Number of hidden layer.", default=3)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=100)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-d", "--dropout", type=float,
                        help="Hidden layer dropout rate.", default=5e-2)
    parser.add_argument("-w", "--weight", type=float,
                        help="Weight decay (L2 penalty).", default=0.)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-g", "--gpuid", type=str,
                        help='GPU to use (leave blank for CPU only)', default="")
    parser.add_argument("-l", "--learningrate", type=float,
                        help="Learning rate.", default=1e-2)
    parser.add_argument("--datapath", type=str,
                        help="The file path of training dataset.", required=True)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    print("\nCross-validation info:")
    print("\nK-fold =", args.kfolds, ", Random seed is", args.randomseed)
    print("\nGPU to use:", "No GPU support" if args.gpuid ==
          "" else "/gpu:{0}".format(args.gpuid))
    print("\nTraining Start...")

    # 执行 FNN 训练模型并验证
    from nn.fnn import run
    run(args.datapath, args.nunits, args.nlayers, args.epochs, args.kfolds,
        args.learningrate, args.dropout, args.weight, args.randomseed)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
