# -*- coding: utf-8 -*-
"""
Implement Random Forest algorithm apply it to classify bio datasets with TensorFlow and SMOTE.
Derived from: Aymeric Damien
Source: https://github.com/aymericdamien/TensorFlow-Examples
Ref: https://github.com/scikit-learn-contrib/imbalanced-learn
Date: 2018-12-27
"""
import os
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="Implement Random Forest algorithm apply it to classify bio datasets with TensorFlow and SMOTE.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--ntrees", type=int,
                        help="The number of trees in the forest.", default=10)
    parser.add_argument("-m", "--mnodes", type=int,
                        help="Grow trees with `max_leaf_nodes` in best-first fashion. Best nodes are defined as relative reduction in impurity.", default=1000)
    parser.add_argument("-e", "--epochs", type=int,
                        help="Number of training epochs.", default=100)
    parser.add_argument("-k", "--kfolds", type=int,
                        help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-n", "--kneighbors", type=int,
                        help="Number of nearest neighbours to used to construct synthetic samples.", default=2)
    parser.add_argument("-r", "--randomseed", type=int,
                        help="pseudo-random number generator state used for shuffling.", default=0)
    parser.add_argument("--datapath", type=str,
                        help="The path of dataset.", required=True)
    parser.add_argument("-c", "--cpu", type=int,
                        help='The number of cores per socket in the machine', default="4")

    args = parser.parse_args()
    # Ignore all GPUs, tf random forest does not benefit from it.
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    # logdir_base = os.getcwd()  # 获取当前目录

    # 输出 Random Forest 模型相关训练参数
    print("\nModel parameters:")
    print("\nNumber of trees:{0}, Max leaf nodes:{1}, Epochs:{2}, K-fold:{3}, Random seed:{4}".format(
        args.ntrees, args.mnodes, args.epochs, args.kfolds, args.randomseed))
    # print("\nThe directory for TF logs:", os.path.join(logdir_base, args.logdir))
    print("\nTraining Start...")

    # 执行 Random Forest 训练模型并验证
    # by parsing the arguments already, we can bail out now instead of waiting
    # for TF to load, in case the arguments aren't ok
    from rf.random_forest_smote import run
    run(args.datapath, args.ntrees, args.mnodes, args.randomseed,
        args.epochs, args.kfolds, args.kneighbors, args.cpu)
    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
