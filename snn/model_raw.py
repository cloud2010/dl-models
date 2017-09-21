# -*- coding: utf-8 -*-
"""
A SelfNormalizingNetworks implementation for Multi-classification Problem using TensorFlow and sklearn-kit library.
Author: liumin@shmtu.edu.cn
Date: 2017-08-23
Tested under: Python3.5 / Python3.6 and TensorFlow 1.1 / 1.2 / 1.3
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
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder  # One-hot matrix transform
from sklearn.metrics import accuracy_score  # 计算 ACC
from sklearn.metrics import confusion_matrix  # 输出混淆矩阵
from sklearn.metrics import classification_report  # 输出 recall f1等指标
from sklearn.utils import resample  # 添加 subsampling 工具类
from .selu import selu, dropout_selu
from .utils import matthews_corrcoef, get_timestamp

# from sklearn.metrics import matthews_corrcoef  # MCC Metric
# 避免输出TensorFlow未编译CPU指令集信息

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
        layers[i] = tf.add(
            tf.matmul(layers[i - 1], weights['h' + str(i)]), biases['b' + str(i)])
        layers[i] = selu(layers[i])
        layers[i] = dropout_selu(layers[i], rate, training=is_training)

    # 输出层建立，Output layer with linear activation
    out_layer = tf.matmul(layers[n_layers], weights['out']) + biases['out']
    return out_layer


def run(inputFile, n_class, h_nums, h_units, epochs, folds, batch_size, d_rate, l_rate, gpu_id, log_dir, random_s=None):
    """
    SNN主程序
    参数
    ----
    inputFile: 训练集文件路径
    n_class: 分类数，即输出层单元数，默认2分类问题
    h_nums: 隐藏层数
    h_units: 隐藏层单元数
    epochs: 每个fold的训练次数
    folds: k-fold折数
    batch_size: 训练时子抽样大小
    d_rate: Dropout rate
    l_rate: Learning rate
    gpu_id: 是否应用GPU加速
    log_dir: 日志记录位置
    random_s: 随机种子
    """
    try:
        # 导入CSV数据
        # TRAIN_CSV = os.path.join(os.path.dirname(__file__), inputFile)
        TRAIN_CSV = inputFile
        # 去掉CSV文件标题行
        train_set = np.genfromtxt(TRAIN_CSV, delimiter=',', skip_header=1)
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)

    # 设定 TF 占位符
    dropout_rate = tf.placeholder(tf.float32)  # dropout 概率
    is_training = tf.placeholder(tf.bool)  # 是否在训练

    # 设定 K-fold 分割器
    rs = KFold(n_splits=folds, shuffle=True, random_state=random_s)

    # 设定神经网络参数，输入层神经元个数为 n_features 列数
    n_input = train_set.shape[1] - 1  # features input 即为特征数

    # 如未设定隐藏层单元数则数目为输入特征数
    if h_units == -1:
        h_units = n_input

    # Tensorboard 记录位置
    logs_path = os.path.join(os.getcwd(), log_dir)

    # 分类矩阵为第一列数据
    n_target = train_set[:, 0]

    # 特征矩阵为去第一列之后数据
    n_features = train_set[:, 1:]

    # 总样本数索引矩阵
    # total_index = np.arange(0, train_set.shape[0])

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
    print("Number of Samples :", n_features.shape[0])

    with tf.variable_scope('Inputs') as scope:
        # tf Graph input 建立TensorFlow占位符，用于存储输入输出矩阵
        x = tf.placeholder(tf.float32, [None, n_input])
        y = tf.placeholder(tf.float32, [None, n_class])

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
            'out': tf.Variable(tf.random_normal([h_units, n_class], stddev=np.sqrt(1 / h_units)))
        }

        # 循环建立中间 hidden layer weights
        for i in range(2, h_nums + 1):
            weights["h" + str(i)] = tf.Variable(tf.random_normal([h_units,
                                                                  h_units], stddev=np.sqrt(1 / h_units)))

    with tf.name_scope('Biases') as scope:
        # 建立 biases 变量
        biases = {
            'out': tf.Variable(tf.random_normal([n_class], stddev=0))
        }
        for i in range(1, h_nums + 1):
            biases["b" +
                   str(i)] = tf.Variable(tf.random_normal([h_units], stddev=0))

    with tf.name_scope('FNN_with_SELU') as scope:
        # 应用模型(FNN配合SELU激活函数)
        pred = multilayer_perceptron(
            x, weights, biases, h_nums, rate=dropout_rate, is_training=is_training)

    # 定义成本函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=pred, labels=y))  # 交叉熵

    # 定义优化器(梯度下降)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=l_rate).minimize(cost)

    # 定义判别函数
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 初始化所有变量
    init = tf.global_variables_initializer()

    # 添加模型保存器
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # 创建TensorFlow统计图表(权重、偏置、loss、acc)
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
        summary_writer = tf.summary.FileWriter(
            logs_path, graph=tf.get_default_graph())

        # 生成 k-fold 训练集、测试集索引
        cv_index_set = rs.split(new_target)
        training_step = 1  # 初始化训练次数
        k_fold_step = 1  # 初始化折数

        # 暂存每次选中的测试集和对应预测结果
        test_cache = pred_cache = np.array([], dtype=np.int)
        # 迭代训练 k-fold 交叉验证
        for train_index, test_index in cv_index_set:
            print("\nFold:", k_fold_step)
            # print("\nTrain-index:\n", train_index, "\nTest-index:\n", test_index)
            # 开始每个 fold 的训练
            for epoch in range(epochs):
                """
                subsampling 防止过拟合，每次抽取(不放回)训练集一定量数据
                如果 batch_size = 0 则不进行 subsampling
                """
                if batch_size > 0:
                    sample_index = resample(
                        train_index, replace=False, n_samples=batch_size)
                else:
                    sample_index = train_index
                batch_x = n_features[sample_index]  # 特征数据用于训练
                batch_y = new_target[sample_index]  # 标记结果用于验证
                # 运行优化器进行训练
                _, costTrain, accTrain, summary = sess.run([optimizer, cost, accuracy, merged_summary_op], feed_dict={x: batch_x,
                                                                                                                      y: batch_y,
                                                                                                                      dropout_rate: d_rate,
                                                                                                                      is_training: True})
                # 输出训练结果
                print("\nTraining Epoch:", '%06d' % (epoch + 1), "Train Accuracy:", "{:.6f}".format(accTrain),
                      "Train Loss:", "{:.6f}".format(costTrain), "Train Size:", sample_index.shape[0])

                # 记录日志
                summary_writer.add_summary(summary, training_step)
                # 训练次数累加
                training_step += 1

            # 输入测试数据
            batch_test_x = n_features[test_index]
            batch_test_y = new_target[test_index]

            # 代入TensorFlow计算图验证测试集
            accTest, costTest, predVal = sess.run([accuracy, cost, pred], feed_dict={x: batch_test_x,
                                                                                     y: batch_test_y,
                                                                                     dropout_rate: 0.0,
                                                                                     is_training: False})
            # One-hot 矩阵转换为原始分类矩阵
            argmax_test = np.argmax(batch_test_y, axis=1)
            argmax_pred = np.argmax(predVal, axis=1)
            print("\nTest dataset Index:\n", test_index)
            print("\nActual Values:\n", argmax_test)
            print("\nPredicted Values:\n", argmax_pred)
            print("\nFold:", k_fold_step, "Test Accuracy:", "{:.6f}".format(
                accTest), "Test Loss:", "{:.6f}".format(costTest), "Test Size:", test_index.shape[0])
            # 暂存每次选中的测试集和预测结果
            test_cache = np.concatenate((test_cache, argmax_test))
            pred_cache = np.concatenate((pred_cache, argmax_pred))

            # 每个fold训练结束后次数 +1
            k_fold_step += 1
            # 完成一个fold训练，权重偏置矩阵重新初始化
            sess.run(init)
            print("\nVariable `Weights` and `Biases` reinitialize.")
            print(
                "\n=========================================================================")

        # 模型各变量持久化
        save_path = saver.save(sess, os.path.join(
            logs_path, "model-{0}.ckpt".format(get_timestamp())))

    # 训练结束计算Precision、Recall、ACC、MCC等统计指标
    class_names = []
    pred_names = []
    for i in range(n_class):
        class_names.append('Class ' + str(i + 1))
        pred_names.append('Pred C' + str(i + 1))

    # 混淆矩阵生成
    cm = confusion_matrix(test_cache, pred_cache)
    df = pd.DataFrame(data=cm, index=class_names, columns=pred_names)

    # 混淆矩阵添加一列代表各类求和
    df['Sum'] = df.sum(axis=1).values

    print("\nOptimization Finished!")
    print("\nNumber of features:",
          n_features.shape[1], ", Number of Samples :", n_features.shape[0])
    print("\nModel saved in file: %s" % save_path)
    # print("\nTest dataset actual values:")
    # print(test_cache)
    # print("\nPredicted values as One-hot matrix:")
    # print(pred_x)
    print("\n=== Model evaluation ===")
    print("\n=== Accuracy classification score ===")
    print("\nACC = {:.6f}".format(accuracy_score(test_cache, pred_cache)))
    print("\n=== Matthews Correlation Coefficient ===")
    print("\nMCC = {:.6f}".format(matthews_corrcoef(cm)))
    print("\n=== Confusion Matrix ===\n")
    print(df)
    print("\n=== Detailed Accuracy By Class ===\n")
    print(classification_report(test_cache, pred_cache,
                                target_names=class_names, digits=6))
    print("\nRead more about these estimator score method, see below:")
    print("\n\thttp://scikit-learn.org/stable/modules/model_evaluation.html")
    print("\thttps://en.wikipedia.org/wiki/Matthews_correlation_coefficient")
