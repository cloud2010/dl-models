# -*- coding: utf-8 -*-
"""
A SelfNormalizingNetworks implementation for Two-classification Problem using TensorFlow and sklearn-kit library.
Author: liumin@shmtu.edu.cn
Date: 2017-08-23
Tested under: Python3.5 / Python3.6 and TensorFlow 1.1 / Tensorflow 1.2
Derived from: Guenter Klambauer, 2017
Source: https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
Cross-validation: k-fold using sklearn.model_selection.KFold
Source: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
Encode categorical integer features using a one-hot aka one-of-K scheme.
Source: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
import numbers
import math
import sys
import getopt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder  # One-hot matrix transform
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
# from sklearn.metrics import matthews_corrcoef  # MCC Metric
# 避免输出TensorFlow未编译CPU指令集信息
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def selu(x):
    """
    Definition of scaled exponential linear units (SELUs)
    SELU激活函数定义
    """
    with ops.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def dropout_selu(x, rate, alpha=-1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Definition of dropout variant for SNNs"""
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1), got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(
            keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(
            x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(
            noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1 - binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - fixedPointMean, 2) + fixedPointVar)))

        b = fixedPointMean - a * \
            (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(
                                    x, rate, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))


def multilayer_perceptron(x, weights, biases, n_layers, rate, is_training):
    """
    层模型构建，建立n个隐藏层
    ----
    参数
    ----
    x: 输入神经元矩阵
    weights: 权重矩阵
    biases: 偏置矩阵
    n_layers: 隐藏层数
    rate: dropout 概率
    is_training: 是否在训练
    输出
    ---
    out_layer: 线性模型输出(W*X_plus_b)
    """
    # 首先建立 1st hidden layer
    layers = {
        1: tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    }
    # Hidden layer with SELU activation
    layers[1] = selu(layers[1])
    # add dropout layer 防止 over-fitting
    layers[1] = dropout_selu(layers[1], rate, training=is_training)
    
    # 从 2nd 开始循环建立中间 hidden layer
    for i in range(2, n_layers + 1):
        layers[i] = tf.add(tf.matmul(layers[i - 1], weights['h' + str(i)]), biases['b' + str(i)])
        layers[i] = selu(layers[i])
        layers[i] = dropout_selu(layers[i], rate, training=is_training)

    # 输出层建立，Output layer with linear activation
    out_layer = tf.matmul(layers[n_layers], weights['out']) + biases['out']
    return out_layer


def main(inputFile, output_classes=2, h_nums=2, h_units=256, epochs=10, folds=10, random_s=None):
    """
    SNN主程序
    参数
    ----
    inputFile: 训练集文件路径
    n_classes: 分类数，即输出层单元数，默认2分类问题
    h_nums: 隐藏层数，默认2层
    h_units: 隐藏层单元数，默认256个
    epochs: 每个fold的训练次数，默认10次
    folds: k-fold折数
    random_s: 随机种子
    """
    try:
        # 导入CSV数据
        TRAIN_CSV = os.path.join(os.path.dirname(__file__), inputFile)
        # 去掉CSV文件标题行
        train_set = np.genfromtxt(TRAIN_CSV, delimiter=',', skip_header=1)
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)

    # 设定参数
    start_time = time.clock()  # 程序开始时间
    learning_rate = 0.01
    training_epochs = epochs  # 每个fold的训练次数
    display_step = 20  # 结果输出步长
    dropoutRate = tf.placeholder(tf.float32)  # dropout 概率
    is_training = tf.placeholder(tf.bool)  # 是否在训练

    # 设定 K-fold 分割器
    rs = KFold(n_splits=folds, shuffle=True, random_state=random_s)

    # 设定神经网络参数，输入层神经元个数为 n_features 列数
    n_input = train_set.shape[1] - 1  # features input 即为特征数，此处为特征矩阵的列数，941个
    n_classes = 2  # 分类数，本例为2类

    # Tensorboard 记录位置
    logs_path = 'tmp'

    # 分类矩阵为第一列数据
    n_target = train_set[:, 0]

    # 特征矩阵为去第一列之后数据
    n_features = train_set[:, 1:]

    # 总样本数索引矩阵
    total_index = np.arange(0, train_set.shape[0])

    # 转换原始分类矩阵为 One-hot Vector
    # reshape(-1, 1) 代表将 1行多列 转为 n行1列
    enc = OneHotEncoder(sparse=True, dtype=np.int)
    one_hot_mat = enc.fit(n_target.reshape(-1, 1))
    print("\nClass Info:{0}\n".format(one_hot_mat.active_features_))
    new_target = one_hot_mat.transform(n_target.reshape(-1, 1)).toarray()
    # 不同 Class 统计
    for i in one_hot_mat.active_features_:
        print("Sum of Class {0} : {1}".format(i, np.sum(n_target == i)))

    # 输出样本数
    print("Sum of Samples :", n_features.shape[0])
    
    with tf.variable_scope('Inputs') as scope:
        # tf Graph input 建立TensorFlow占位符，用于存储输入输出矩阵
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])

    """
    建立权重和偏置变量，默认分配非0随机值
    stddev: The standard deviation of the normal distribution.
    先建立 1st hidden layer 和 output layer 的权重变量，再迭代建立中间隐藏层的
    1st hidden layer weights 输入单元数为特征数， output layer weights 输出单元数为分类数
    中间 hidden layer weights 输入输出单元数一致
    """
    with tf.name_scope('Weights') as scope:
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, h_units], stddev=np.sqrt(1 / n_input))),
            'out': tf.Variable(tf.random_normal([h_units, output_classes], stddev=np.sqrt(1 / h_units)))
        }

        # 循环建立中间 hidden layer weights
        for i in range(2, h_nums + 1):
            weights["h" + str(i)] = tf.Variable(tf.random_normal([h_units, h_units], stddev=np.sqrt(1 / h_units)))
    
    with tf.name_scope('Biases') as scope:
        # 建立 biases 变量
        biases = {
            'out': tf.Variable(tf.random_normal([output_classes], stddev=0))
        }
        for i in range(1, h_nums + 1):
            biases["b" + str(i)] = tf.Variable(tf.random_normal([h_units], stddev=0))

    with tf.name_scope('FNN_with_SELU') as scope:
        # 应用模型（FNN配合SELU激活函数）
        pred = multilayer_perceptron(
            x, weights, biases, h_nums, rate=dropoutRate, is_training=is_training)

    # 定义成本函数
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # 交叉熵

    # 定义优化器（梯度下降）
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(cost)

    # 定义判别函数
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 初始化所有变量
    init = tf.global_variables_initializer()

    # 添加模型保存器
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # 创建TensorFlow统计图表（权重、偏置、loss、acc)
    for i in range(1, h_nums + 1):
        tf.summary.histogram("weights" + str(i), weights['h' + str(i)])
        tf.summary.histogram("biases" + str(i), biases['b' + str(i)])

    # Create a summary to monitor cost tensor
    tf.summary.scalar("Loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("Accuracy", accuracy)
    # 将所有统计信息汇总
    merged_summary_op = tf.summary.merge_all()

    # 启动会话
    with tf.Session() as sess:
        sess.run(init)

        # 记录训练工程数据至log文件，便于TensorBoard可视化
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # 所有正确样本计数
        acc_val = 0.
        # class 1 计数，标记为Positive样本
        tp_plus_fn = 0.
        # True Positive Counter
        tp = 0.
        # class 2 计数，标记为Negative样本
        tn_plus_fp = 0.
        # True Negative Counter
        tn = 0.

        # 生成 k-fold 训练集、测试集索引
        cv_index_set = rs.split(new_target)
        training_step = 1  # 训练次数
        k_fold_step = 1  # 折数
        
        # 迭代训练 k-fold 交叉验证
        for train_index, test_index in cv_index_set:
            print("\nFold:", k_fold_step)
            # print("\nTrain-index:\n", train_index, "\nTest-index:\n", test_index)
            # 开始每个 fold 的训练
            for epoch in range(training_epochs):
                batch_x = n_features[train_index]  # 特征数据用于训练
                batch_y = new_target[train_index]  # 标记结果用于验证
                # 运行优化器进行训练
                _, costTrain, accTrain, summary = sess.run([optimizer, cost, accuracy, merged_summary_op], feed_dict={x: batch_x,
                                                                                                                      y: batch_y,
                                                                                                                      dropoutRate: 0.05,
                                                                                                                      is_training: True})
                # 输出训练结果
                print("\nTraining Epoch:", '%06d' % training_step, "Train Accuracy:", "{:.6f}".format(accTrain),
                      "Train Loss:", "{:.6f}".format(costTrain), "Train Size:", train_index.shape[0])
                
                # 记录日志
                summary_writer.add_summary(summary, training_step)
                # 训练次数累加
                training_step += 1

            # 输入测试数据
            batch_test_x = n_features[test_index]
            batch_test_y = new_target[test_index]

            # 代入TensorFlow计算图验证测试集
            accTest, costTest, predVal, corrPred = sess.run([accuracy, cost, pred, correct_prediction], feed_dict={x: batch_test_x,
                                                                                                                   y: batch_test_y,
                                                                                                                   dropoutRate: 0.0,
                                                                                                                   is_training: False})
            
            print("\nCorrect Prediction:\n", corrPred)
            print("Test Accuracy:", "{:.6f}".format(accTest), "Test Loss:", "{:.6f}".format(costTest), "Test Size:", test_index.shape[0])
            """
            汇总每次测试结果，用于计算ACC / MCC / SP / SN
            首先将预测集和测试集进行argmax转换
            """
            pred_val_array = np.argmax(predVal, axis=1)
            test_val_array = np.argmax(batch_test_y, axis=1)
            # 循环数组统计
            for i in range(pred_val_array.size):
                if(test_val_array[i] == 1):  # TP FN 统计
                    # class 1 计数 positive
                    tp_plus_fn += 1
                    if(pred_val_array[i] == 1):
                        tp += 1  # class 1 预测正确TP加1
                if(test_val_array[i] == 0):  # TN FP 统计
                    # class 0 计数 negative
                    tn_plus_fp += 1
                    if(pred_val_array[i] == 0):
                        tn += 1  # class 1 预测正确TP加1

            # 打印测试结果
            # print("\nBefore Argmax Predicted-Value:", predVal, "Actual-Value:", batch_test_y)
            # print("\nAfter Argmax Predicted-Value:", np.argmax(predVal, axis=1), "\nActual-Value:", np.argmax(batch_test_y, axis=1))
            
            # 每个fold训练结束后次数 +1
            k_fold_step += 1

        # 训练结束计算Precision、Recall、ACC、MCC等统计指标
        fn = tp_plus_fn - tp
        fp = tn_plus_fp - tn
        mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * tp_plus_fn * tn_plus_fp * (tn + fn))
        print("\n===============================================================")
        print("\nOptimization Finished!\tTotal training times:", training_step - 1)
        print("\nModel evaluation:")
        print("\nP = TP + FN =", tp_plus_fn, ", N = TN + FP =", tn_plus_fp)
        print("\nTP =", tp, ", TN =", tn)
        print("\nTPR = TP / (TP+FN) =", "{:.6%}".format(tp / tp_plus_fn))
        print("\nTNR = TN / (TN+FP) =", "{:.6%}".format(tn / tn_plus_fp))
        print("\nPPV = TP / (TP+FP) =", "{:.6%}".format(tp / (tp + fp)))
        print("\nNPV = TN / (TN+FN) =", "{:.6%}".format(tn / (tn + fn)))
        print("\nACC = (TP + TN) / (P + N) =", "{:.6%}".format((tp + tn) / (tp_plus_fn + tn_plus_fp)))
        print("\nMCC =", "{:.6f}".format(mcc))
        print("\nRead more about these estimator score method, see below:")
        print("\n\thttps://en.wikipedia.org/wiki/Matthews_correlation_coefficient")
        # 模型各变量持久化
        save_path = saver.save(sess, logs_path + "//model.ckpt")
        print("\nModel saved in file: %s" % save_path)
        end_time = time.clock()  # 程序结束时间
        print("\nRuntime Consumption:", "{:.6f} seconds".format((end_time - start_time)))


def snn_usage():
    """SNN程序使用说明
    """
    print("\nThis a SelfNormalizingNetworks implementation using TensorFlow v1.2.1 and scikit-learn v0.19.")
    # print("\nUsage:%s [-c|-l|-u|-k|-r] [--help|--inputFile] args...." % sys.argv[0])
    print("\nUsage:python %s [-l|-u||-e|-k|-r] [--help|--inputFile] args...." % sys.argv[0])
    print("\nExample:python %s -l 2 -u 256 -e 100 -k 10 -r 6 --inputFile=train.csv" % sys.argv[0])
    print("\nIntroduction:")
    # print("\n-c: Number of class, e.g 2 or n")
    print("\n-l: Number of hidden layers. Default=2")
    print("\n-u: Number of hidden layer units. Default=256")
    print("\n-e: Training epochs in each fold. Default=10")
    print("\n-k: Number of folds. Must be at least 2. Default=10")
    print("\n-r: Random seed, pseudo-random number generator state used for shuffling")
    print("\n--inputFile: The filename of training dataset\n")

if __name__ == "__main__":
    try:
        # 默认参数构建
        n_hiddens = 2  # 默认2层隐藏层
        n_hidden_units = 256  # 默认256个隐藏层单元
        n_classes = 2  # 默认2分类问题
        n_epochs = 10  # 默认每个fold进行10次训练
        n_fold = 10  # 默认 10-fold
        random_seed = None  # 默认随机种子为 0
        train_file = ""

        opts, args = getopt.getopt(sys.argv[1:], "hc:l:u:e:k:r:", ["help", "inputFile="])

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
        print("\nTarget Classes:{0}, Hidden Layers:{1}, Hidden Layer Units:{2}, Training Data:{3}".format(n_classes, n_hiddens, n_hidden_units, train_file))
        print("\nCross-validation info:")
        print("\nK-fold =", n_fold, ", Training epochs per fold:", n_epochs, ", Random seed is", random_seed)
        print("\nTraining Start...")
        # 执行主程序
        main(inputFile=train_file, output_classes=int(n_classes), h_nums=int(n_hiddens), h_units=int(
            n_hidden_units), epochs=int(n_epochs), folds=int(n_fold), random_s=int(random_seed))
    except getopt.GetoptError:
        print("\nMaybe you input some invalid parameters!")
        print("\nTry `python %s --help` for more information." % sys.argv[0])
        sys.exit(1)
