# -*- coding: utf-8 -*-
"""
A SelfNormalizingNetworks implementation example using TensorFlow library.
Author: liumin@shmtu.edu.cn
Date: 2017-08-16
Tested under: Python3.5 / Python3.6 and TensorFlow 1.1 / Tensorflow 1.2
Derived from: Guenter Klambauer, 2017
Source: https://github.com/bioinf-jku/SNNs/blob/master/SelfNormalizingNetworks_MLP_MNIST.ipynb
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


def one_hot_matrix(ori_target, nums_classes=2):
    """
    原始单列分类矩阵转换为 One-hot Vector
    例如：将 Class 1 or 2 or 3 转换为  [0, 0, 1] or [0, 1, 0] or [1, 0, 0]
    参数
    ----
    ori_target: 原始分类矩阵
    nums_classes: 分类数，默认为2分类
    返回
    ----
    新分类矩阵
    """
    new_target = np.zeros(shape=[1, nums_classes])
    for i in ori_target:
        if i == 1:  # Label 1
            new_target = np.vstack((new_target, np.array([0, 1])))
        elif i == 2:  # Label 2
            new_target = np.vstack((new_target, np.array([1, 0])))
    # 删除第一行空值
    new_target = np.delete(new_target, 0, 0)
    return new_target


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


def main(inputFile, output_classes=2, h_nums=2, h_units=256):
    """
    SNN主程序
    参数
    ----
    inputFile: 训练集文件路径
    n_classes: 分类数，即输出层单元数，默认2分类问题
    h_nums: 隐藏层数，默认2层
    h_units: 隐藏层单元数，默认256个
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
    training_epochs = train_set.shape[0]  # 训练次数
    batch_size = training_epochs - 1  # 每批次随机选择的样本数（留一法）
    display_step = 20  # 结果输出步长
    dropoutRate = tf.placeholder(tf.float32)  # dropout 概率
    is_training = tf.placeholder(tf.bool)  # 是否在训练

    # 设定神经网络参数，输入层神经元个数为 n_features 列数
    n_input = train_set.shape[1] - 1  # features input 即为特征数，此处为特征矩阵的列数，941个
    n_classes = 2  # 分类数，本例为2类

    # Tensorboard 记录位置
    logs_path = 'tmp'

    # 分类矩阵为第一列数据
    n_target = train_set[:, 0]

    # 特征矩阵为去第一列之后数据
    n_features = train_set[:, 1:]

    # 总样本数索引矩阵（留一法随机抽样使用）
    total_index = np.arange(0, train_set.shape[0])

    # 转换原始分类矩阵为 One-hot Vector
    new_target = one_hot_matrix(n_target, n_classes)

    # 输出分类和特征矩阵的大小
    print("\nTarget Shape:", new_target.shape, ", Features Shape:",
          n_features.shape, ", Sum of Samples:", n_features.shape[0])
    
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
            'out': tf.Variable(tf.random_normal([h_units, n_classes], stddev=np.sqrt(1 / h_units)))
        }

        # 循环建立中间 hidden layer weights
        for i in range(2, h_nums + 1):
            weights["h" + str(i)] = tf.Variable(tf.random_normal([h_units, h_units], stddev=np.sqrt(1 / h_units)))
    
    with tf.name_scope('Biases') as scope:
        # 建立 biases 变量
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes], stddev=0))
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
    tf.summary.scalar("loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # 将所有统计信息汇总
    merged_summary_op = tf.summary.merge_all()

    # 启动会话
    with tf.Session() as sess:
        sess.run(init)

        # 记录训练工程数据至log文件，便于TensorBoard可视化
        summary_writer = tf.summary.FileWriter(
            logs_path, graph=tf.get_default_graph())

        # 存储每次验证结果
        test_val = 0.
        # class 1 计数，标记为Positive样本
        tp_plus_fn = 0.
        # True Positive Counter
        tp = 0.
        # class 2 计数，标记为Negative样本
        tn_plus_fp = 0.
        # True Negative Counter
        tn = 0.
        # 迭代训练
        for epoch in range(training_epochs):
            # 平均成本
            avg_cost = 0.
            total_batch = int(n_target.shape[0] / batch_size)
            """
            留一法进行训练，每次留1组测试数据，剩余数据全部参与训练
            """
            # 从索引0开始逐一选取测试数据
            test_index = np.array([epoch])
            # 差集为训练数据
            sample_index = np.setdiff1d(total_index, test_index)
            for i in range(total_batch):
                # 生成抽样训练集(特征和分类结果)
                batch_x = n_features[sample_index]
                batch_y = new_target[sample_index]
                # 运行优化器进行训练
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              dropoutRate: 0.05,
                                                              is_training: True})
                # 计算平均训练成本
                avg_cost += c / total_batch
                # 留一法选择出测试集
                batch_test_x = n_features[test_index]
                batch_test_y = new_target[test_index]

                # 代入 tf graph 测试集结果输出
                accTest, costTest, predVal = sess.run([accuracy, cost, pred], feed_dict={x: batch_test_x,
                                                                                         y: batch_test_y,
                                                                                         dropoutRate: 0.0,
                                                                                         is_training: False})
                """
                汇总每次测试结果，用于计算ACC / MCC / SP / SN
                """
                # TP / FN 统计
                if np.argmax(batch_test_y) == 1:
                    # class 1 计数
                    tp_plus_fn += 1
                    if accTest == 1.:
                        # class 1 预测正确TP加1
                        tp += 1
                # TN / FP 统计
                if np.argmax(batch_test_y) == 0:
                    # class 2 计数
                    tn_plus_fp += 1
                    if accTest == 1.:
                        # class 2 预测正确TN加1
                        tn += 1
                # 计数
                test_val += accTest
                # 打印测试结果
                print("\nTest index:", test_index[0])
                print("Before Argmax Predicted-Value:",
                      predVal, "Actual-Value:", batch_test_y)
                print("After Argmax Test-Accuracy:", accTest,
                      "Test-Loss:", costTest, "Test-size: 1")

            """每隔display_step步长打印训练统计信息"""
            if epoch % display_step == 0:
                print("\nEpoch:", '%06d' % (epoch + 1), "Avg-Cost=",
                      "{:.9f}".format(avg_cost), "Train-Size:", batch_size)
                accTrain, costTrain, summary = sess.run([accuracy, cost, merged_summary_op], feed_dict={x: batch_x,
                                                                                                        y: batch_y,
                                                                                                        dropoutRate: 0.0,
                                                                                                        is_training: False})
                # 记录日志
                summary_writer.add_summary(summary, epoch)
                # 控制台输出结果
                print("Train-Accuracy:", accTrain, "Train-Loss:", costTrain)

        # 训练结束
        print("\nTest-times:", training_epochs, "Accuracy:",
              "{:.4%}".format(test_val / training_epochs))
        print("\nTP =", tp, " TN =", tn)
        print("\nTP + FN =", tp_plus_fn, " TN + FP =", tn_plus_fp)
        print("\nTP / (TP+FN) =", "{:.4%}".format(tp / tp_plus_fn))
        print("\nTN / (TN+FP) =", "{:.4%}".format(tn / tn_plus_fp))
        # 计算MCC
        fn = tp_plus_fn - tp
        fp = tn_plus_fp - tn
        mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * tp_plus_fn * tn_plus_fp * (tn + fn))
        print("\nMCC:", "{:.4f}".format(mcc))
        print("\nOptimization Finished! ")
        # 模型各变量持久化
        save_path = saver.save(sess, logs_path + "//model.ckpt")
        print("\nModel saved in file: %s" % save_path)
        end_time = time.clock()  # 程序结束时间
        print("\nRun-Time Consumption:",
              "{:.4f} seconds".format((end_time - start_time)))


def snn_usage():
    """SNN程序使用说明
    """
    print("\nThis a SelfNormalizingNetworks implementation using TensorFlow v1.2.1 framework.")
    print("\nUsage:%s [-c|-l|-u] [--help|--inputFile] args...." % sys.argv[0])
    print("\nExample:python %s -c 2 -l 2 -u 256 --inputFile=train.csv" % sys.argv[0])
    print("\nIntroduction:")
    print("\n-c: The number of class, e.g 2 or n")
    print("\n-l: The number of hidden layers")
    print("\n-u: The number of hidden layer units")
    print("\n--inputFile: The filename of training dataset\n")

if __name__ == "__main__":
    try:
        # 默认参数构建
        n_hiddens = 2  # 默认2层隐藏层
        n_hidden_units = 256  # 默认256个隐藏层单元
        n_classes = 2  # 默认2分类问题
        train_file = ""

        opts, args = getopt.getopt(sys.argv[1:], "hc:l:u:", ["help", "inputFile="])

        # 无输入参数显示帮助信息
        if len(opts) == 0:
            snn_usage()
            sys.exit()
        # print(opts)    
        # 相关运行参数检查
        for opt, arg in opts:
            if opt in ("--help"):
                snn_usage()
                sys.exit(1)
            if opt in ("-c"):
                n_classes = arg
            if opt in ("-l"):
                n_hiddens = arg
            if opt in ("-u"):
                n_hidden_units = arg
            if opt in ("--inputFile"):
                train_file = arg
            # print("%s  ==> %s" % (opt, arg))
        # 检查是否输入训练集文件
        if train_file == "":
            print("\nPlease input training dataset filename!\n")
            sys.exit(1)
        # 输出SNN模型相关训练参数
        print("\nSNN Parameters info:")
        print("\nN-Classes:{0}, Hidden Layers:{1}, Hidden Layer Units:{2}, Training Data:{3}".format(n_classes, n_hiddens, n_hidden_units, train_file))
        print("\nTraining Start...")
        # 执行主程序
        main(inputFile=train_file, output_classes=int(n_classes),
             h_nums=int(n_hiddens), h_units=int(n_hidden_units))
    except getopt.GetoptError:
        snn_usage()
        sys.exit(1)
