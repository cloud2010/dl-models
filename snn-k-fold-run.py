# -*- coding: utf-8 -*-
"""
Using SNNs
"""
import sys
import os
import time
# from snn import model_raw
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a SelfNormalizingNetworks implementation using TensorFlow v1.2.1+ and scikit-learn v0.19+.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--nclass", type=int,
                        help="Number of class. Must be at least 2 aka two-classification.", default=2)
    parser.add_argument("-l", "--nlayers", type=int,
                        help="Number of hidden layers.", default=3)
    parser.add_argument("-u", "--nunits", type=int,
                        help="Number of hidden layer units. (-1: use input size)", default=-1)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=10)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-s", "--batchsize", type=int,
                        help="Subsampling size in training. If 0, all samples will be training.", default=0)
    parser.add_argument("-d", "--dropout", type=float,
                        help="Hidden dropout rate (implies input-dropout of 0.2).", default=0.1)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("-g", "--gpuid", type=str,
                        help='GPU to use (leave blank for CPU only)', default="")
    parser.add_argument("--inputfile", type=str,
                        help="The filename of training dataset.", required=True)
    parser.add_argument("--learningrate", type=float,
                        help="Learning rate.", default=1e-5)
    parser.add_argument(
        "--logdir", type=str, help="The directory for TF logs and summaries.", default="logs")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    logdir_base = os.getcwd()  # 获取当前目录

    # 检查是否输入训练集文件
    if args.inputfile == "":
        print("\nPlease input training dataset filename!\n")
        sys.exit(1)

    # 输出SNN模型相关训练参数
    print("\nModel Parameters:")
    print("\nTarget Classes:{0}, Training epochs:{1}, Depth:{2}, Units:{3}, Learning rate:{4}, Dropout rate:{5}, Training Data:{6}".format(
        args.nclass,
        args.epochs,
        args.nlayers,
        args.nunits,
        args.learningrate,
        args.dropout,
        args.inputfile))
    print("\nCross-validation info:")
    print("\nK-fold =", args.kfolds, ", Batch-size =",
          args.batchsize, ", Random seed is", args.randomseed)
    print("\nThe directory for TF logs:",
          os.path.join(logdir_base, args.logdir))
    print("\nGPU to use:", "No GPU support" if args.gpuid ==
          "" else "/gpu:{0}".format(args.gpuid))
    print("\nTraining Start...")

    # 执行 SNN 训练模型并验证
    # by parsing the arguments already, we can bail out now instead of waiting
    # for TF to load, in case the arguments aren't ok
    from snn.model_raw import run
    run(args.inputfile, args.nclass, args.nlayers, args.nunits, args.epochs, args.kfolds,
                      args.batchsize, args.dropout, args.learningrate, args.gpuid, args.logdir, args.randomseed)
    end_time = time.time()  # 程序结束时间
    print("\nRuntime Consumption:", "{0:.6f} mins = {1:.6f} seconds".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
