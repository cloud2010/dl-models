# -*- coding: utf-8 -*-
"""
This a Basic ConvNet implementation.
"""
import sys
import os
import time
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = ArgumentParser(description="This a basic CNN implementation using TensorFlow and k-fold cross-validation was performed after training.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--nclass", type=int,
                        help="Number of class. Must be at least 2 aka two-classification.", default=2)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=20)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-s", "--batchsize", type=int,
                        help="Subsampling size in training. If 0, all samples will be training.", default=0)
    parser.add_argument("-d", "--dropout", type=float,
                        help="The dropout rate, between 0 and 1. E.g. rate=0.1 would drop out 10%% of input units.", default=0.1)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=None)
    parser.add_argument("--conv1h", type=int,
                        help="Specifying the height of the 1st 2D convolution window.", default=5)
    parser.add_argument("--conv1w", type=int,
                        help="Specifying the width of the 1st 2D convolution window.", default=5)
    parser.add_argument("--conv2h", type=int,
                        help="Specifying the height of the 2nd 2D convolution window.", default=3)
    parser.add_argument("--conv2w", type=int,
                        help="Specifying the width of the 2nd 2D convolution window.", default=3)
    parser.add_argument("--conv2f", type=int,
                        help="The number of filters in the 2nd 2D convolution.", default=64)
    parser.add_argument("--datah", type=int,
                        help="Specifying the height of the sample.", default=20)
    parser.add_argument("--dataw", type=int,
                        help="Specifying the width of the sample.", default=20)
    # parser.add_argument("-g", "--gpuid", type=str,
    #                     help='GPU to use (leave blank for CPU only)', default="")
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("--learningrate", type=float,
                        help="Learning rate.", default=1e-3)
    # parser.add_argument("--logdir", type=str, help="The directory for TF logs and summaries.", default="logs")

    return parser.parse_args()

def print_hyperparameters(args):
    logging.info("CNN HyperParameters:")
    logging.info("N-classes: %d, Training epochs: %d, Learning rate: %f, Dropout rate: %f",
                 args.nclass, args.epochs, args.learningrate, args.dropout)
    logging.info("The kernel size of the 1st 2D convolution window: [%d, %d], The number of filters: 32",
                 args.conv1h, args.conv1w)
    logging.info("The kernel size of the 2nd 2D convolution window: [%d, %d], The number of filters: %d",
                 args.conv2h, args.conv2w, args.conv2f)
    logging.info("Cross-validation info:")
    logging.info("K-fold: %d, Random seed is %s", args.kfolds, args.randomseed)

if __name__ == "__main__":
    try:
        start_time = time.time()
        args = parse_arguments()
        print_hyperparameters(args)

        from cnn.utils import get_timestamp
        logging.info("%s Start training...", get_timestamp())

        # 执行 CNN 训练模型并验证
        from cnn.cnn_bio import run_model
        run_model(args.datapath, args.learningrate, args.epochs, args.dropout, args.conv1h, args.conv1w, args.conv2h,
                  args.conv2w, args.conv2f, args.datah, args.dataw, args.nclass, args.kfolds, args.randomseed)

        end_time = time.time()  # 程序结束时间
        logging.info("[Finished in: %.6f mins = %.6f seconds]",
                     ((end_time - start_time) / 60), (end_time - start_time))
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
