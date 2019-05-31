# -*- coding: utf-8 -*-
"""
This a Bidirectional Recurrent Neural Network implementation.
Date: 2019-05-31
"""
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'

if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This a Bidirectional Recurrent Neural Network implementation.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-e", "--epochs", help="Number of training epochs.", type=int, default=20)
    parser.add_argument("-k", "--kfolds", type=int, help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-r", "--randomseed", type=int, help="pseudo-random number generator state used for shuffling.",
                        default=0)
    parser.add_argument("-u", "--nunits", type=int, help="Number of hidden layer units.", default=256)
    parser.add_argument("-n", "--nlayers", type=int, help="Number of hidden layers.", default=2)
    parser.add_argument("-f", "--fragment", type=int, help="Specifying the `length` of sequences fragment.", default=1)
    parser.add_argument("--datapath", type=str, help="The path of dataset.", required=True)
    parser.add_argument("--learningrate", type=float, help="Learning rate.", default=3e-3)
    parser.add_argument("-g", "--gpuid", type=str, help='GPU to use (leave blank for CPU only)', default="")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

    # 输出 Bi-RNN 模型相关训练参数
    print("\nBi-RNN HyperParameters:")
    print("\nTraining epochs: {0}, Learning rate: {1}, Sequences fragment length: {2}, Hidden Units: {3}, Hidden Layers: {4}".format(
        args.epochs, args.learningrate, args.fragment, args.nunits, args.nlayers))
    print("\nCross-validation info:")
    print("\nK-fold:", args.kfolds, ", Random seed is", args.randomseed)

    print("\nGPU to use:", "No GPU support" if args.gpuid == "" else "/gpu:{0}".format(args.gpuid))
    print("\nTraining Start...")

    # 执行 RNN 训练模型并验证
    # by parsing the arguments already, we can bail out now instead of waiting
    # for TF to load, in case the arguments aren't ok
    from rnn.birnn_torch import run

    run(args.datapath, args.nunits, args.fragment, args.nlayers,
        args.learningrate, args.epochs, args.kfolds, args.randomseed)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
