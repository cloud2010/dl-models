"""
Implementation of Graph Convolutional Networks in TensorFlow
"""

import sys
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="Implementation of Graph Convolutional Networks in TensorFlow",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--nclass", type=int, help="Number of class. Must be at least 2 aka two-classification.", default=2)
    parser.add_argument("-u", "--nunits", type=int, help="Number of hidden layer units.", default=16)
    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs.", default=20)
    parser.add_argument("-d", "--dropout", type=float, help="The dropout rate, between 0 and 1. E.g. rate=0.1 would drop out 10% of input units.", default=0.1)
    parser.add_argument("-r", "--randomseed", type=int, help="pseudo-random number generator state used for shuffling.", default=None)
    parser.add_argument("-l", "--learningrate", type=float, help="Learning rate.", default=1e-3)
    parser.add_argument("-g", "--gpuid", type=str, help='GPU to use (leave blank for CPU only)', default="")
    parser.add_argument("-k", "--kfolds", type=int, help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("--features", type=str, help="The path of features matrix.", required=True)
    parser.add_argument("--adj", type=str, help="The path of adjacency matrix.", required=True)
    # parser.add_argument("--logdir", type=str, help="The directory for TF logs and summaries.", default="logs")

    # logdir_base = os.getcwd()  # 获取当前目录
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    # 输出CNN模型相关训练参数
    print("\nGCN HyperParameters:")
    print("\nTraining epochs:{0}, Learning rate:{1}, Dropout rate:{2}, Units:{3}, Random seed:{4}".format(args.epochs, args.learningrate, args.dropout, args.nunits, args.randomseed))
    # print("\nCross-validation info:")
    # print("\nK-fold:", args.kfolds)
    # print("\nThe directory for TF logs:", os.path.join(logdir_base, args.logdir))
    print("\nGPU to use:", "No GPU support" if args.gpuid == "" else "/gpu:{0}".format(args.gpuid))
    print("\nTraining Start...")

    # 执行 CNN 训练模型并验证
    # by parsing the arguments already, we can bail out now instead of waiting
    # for TF to load, in case the arguments aren't ok
    from gcn.gcn_bio import run
    run(args.features, args.adj, args.nclass, args.nunits, args.epochs, args.kfolds, args.learningrate, args.dropout, args.randomseed)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(((end_time - start_time) / 60), (end_time - start_time)))
