# -*- coding: utf-8 -*-

' a test module '

__author__ = 'Min'

import sys

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
