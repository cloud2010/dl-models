# -*- coding: utf-8 -*-
"""
Using CNN
"""
import sys
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a basic CNN implementation using TensorFlow.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=100)
    parser.add_argument("-k", "--kfolds", type=int,
                         help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-s", "--batchsize", type=int,
                        help="Subsampling size in training.", default=256)
    parser.add_argument("-d", "--dropout", type=float,
                        help="Probability to keep units.", default=8e-1)
    parser.add_argument("-r", "--randomseed", type=int,
                         help="pseudo-random number generator state used for shuffling.", default=None)
    # parser.add_argument("-g", "--gpuid", type=str,
    #                     help='GPU to use (leave blank for CPU only)', default="")
    # parser.add_argument("--inputfile", type=str,
    #                     help="The filename of training dataset.")
    parser.add_argument("--learningrate", type=float,
                        help="Learning rate.", default=1e-3)
    # parser.add_argument("--logdir", type=str, help="The directory for TF logs and summaries.", default="logs")

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    # logdir_base = os.getcwd()  # 获取当前目录

    # 输出CNN模型相关训练参数
    print("\nCNN HyperParameters:")
    print("\nTraining epochs:{0}, Batch size:{1}, Learning rate:{2}, Units keep probability:{3}".format(
        args.epochs, args.batchsize, args.learningrate, args.dropout))
    print("\nCross-validation info:")
    print("\nK-fold:", args.kfolds, ", Random seed is", args.randomseed)
    # print("\nThe directory for TF logs:",
    #       os.path.join(logdir_base, args.logdir))
    # print("\nGPU to use:", "No GPU support" if args.gpuid ==
    #       "" else "/gpu:{0}".format(args.gpuid))
    print("\nTraining Start...")

    # 执行 CNN 训练模型并验证
    # by parsing the arguments already, we can bail out now instead of waiting
    # for TF to load, in case the arguments aren't ok
    from cnn.cnn_basic import model_run
    model_run(args.learningrate, args.epochs, args.batchsize, args.dropout, args.kfolds, args.randomseed)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
