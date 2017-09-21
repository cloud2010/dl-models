# -*- coding: utf-8 -*-
"""
Using SNNs
"""
import sys
import time
import getopt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


def snn_usage():
    """SNN程序使用说明
    """
    print("\nThis a SelfNormalizingNetworks implementation using TensorFlow v1.2.1 and scikit-learn v0.19.")
    print(
        "\nUsage:python %s [-c|-l|-u||-e|-k|-s|-d|-r] [--help|--inputFile] args...." % sys.argv[0])
    print("\nExample:python %s -c 2 -l 2 -u 256 -e 100 -k 10 -s 50 -d 0.5 -r 6 --inputFile=train.csv" %
          sys.argv[0])
    print("\nIntroduction:")
    print("\n-c: Number of class. Must be at least 2 aka two-classification.")
    print("\n-l: Number of hidden layers. Default=2")
    print("\n-u: Number of hidden layer units. Default=256")
    print("\n-e: Training epochs in each fold. Default=10")
    print("\n-k: Number of folds. Must be at least 2. Default=10")
    print("\n-s: Subsampling size in training. If 0, all samples will be training. Default=0")
    print("\n-d: Dropout rate. Default=0.5")
    print("\n-r: Random seed, pseudo-random number generator state used for shuffling")
    print("\n--inputFile: The filename of training dataset\n")


def run():
    """ 运行SNN模型  """
    parser = ArgumentParser(description="This a SelfNormalizingNetworks implementation using TensorFlow v1.2.1+ and scikit-learn v0.19+.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--nclass", type=int, help="Number of class. Must be at least 2 aka two-classification.", default=2)
    parser.add_argument("-l", "--nlayers", type=int, help="Number of hidden layers.", default=3)
    parser.add_argument("-u", "--nunits", type=int, help="Number of hidden layer units. (-1: use input size)", default=-1)
    parser.add_argument("-e", "--epochs", type=int, help="Number of training epochs.", default=10)
    parser.add_argument("-k", "--kfolds", type=int, help="Number of folds. Must be at least 2.", default=10)
    parser.add_argument("-s", "--batchsize", type=int, help="Subsampling size in training. If 0, all samples will be training.", default=0)
    parser.add_argument("-d", "--dropout", type=float, help="Hidden dropout rate (implies input-dropout of 0.2).", default=0.1)
    parser.add_argument("-r", "--randomseed", type=int, help="pseudo-random number generator state used for shuffling.", default=0)
    try:
        # 默认参数构建
        start_time = time.time()  # 程序开始时间
        n_hiddens = 2  # 默认2层隐藏层
        n_hidden_units = 256  # 默认256个隐藏层单元
        n_classes = 2  # 默认2分类问题
        n_epochs = 10  # 默认每个fold进行10次训练
        n_fold = 10  # 默认 10-fold
        n_samples = 0  # 默认训练集不进行 subsampling
        dropout_rate = 0.5  # 默认 dropout_rate = 0.5
        random_seed = None  # 默认随机种子为 0
        train_file = ""

        opts, args = getopt.getopt(sys.argv[1:], "hc:l:u:e:k:s:d:r:", [
            "help", "inputFile="])

        # 无输入参数显示帮助信息
        if len(opts) == 0:
            snn_usage()
            sys.exit()
        # 相关运行参数检查
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                snn_usage()
                sys.exit(1)
            if opt in ("-c"):
                n_classes = arg
            if opt in ("-l"):
                n_hiddens = arg
            if opt in ("-u"):
                n_hidden_units = arg
            if opt in ("-e"):
                n_epochs = arg
            if opt in ("-k"):
                n_fold = arg
            if opt in ("-s"):
                n_samples = arg
            if opt in ("-d"):
                dropout_rate = arg
            if opt in ("-r"):
                random_seed = arg
            if opt in ("--inputFile"):
                train_file = arg
            # print("%s  ==> %s" % (opt, arg))
        # 检查是否输入训练集文件
        if train_file == "":
            print("\nPlease input training dataset filename!\n")
            sys.exit(1)

        # 输出SNN模型相关训练参数
        print("\nSNN Parameters info:")
        print("\nTarget Classes:{0}, Hidden Layers:{1}, Hidden Layer Units:{2}, Training Data:{3}".format(
            n_classes, n_hiddens, n_hidden_units, train_file))
        print("\nCross-validation info:")
        print("\nK-fold =", n_fold, ", Training epochs per fold:", n_epochs, ", Subsampling size in training:",
              n_samples, ", Dropout rate is:", dropout_rate, ", Random seed is", random_seed)
        print("\nTraining Start...")

        # 执行主程序
        main(inputFile=train_file, output_classes=int(n_classes), h_nums=int(n_hiddens), h_units=int(n_hidden_units), epochs=int(
            n_epochs), folds=int(n_fold), sample_size=int(n_samples), d_rate=float(dropout_rate), random_s=int(random_seed))

        end_time = time.time()  # 程序结束时间
        print("\nRuntime Consumption:", "{0:.6f} mins = {1:.6f} seconds".format(
            ((end_time - start_time) / 60), (end_time - start_time)))
    except getopt.GetoptError:
        print("\nMaybe you input some invalid parameters!")
        print("\nTry `python %s --help` for more information." % sys.argv[0])
        sys.exit(1)
