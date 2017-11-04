# -*- coding: utf-8 -*-
""" Build an Basic ConvNet for training membrane dataset in TensorFlow.

convolutional layer1 + max pooling;
convolutional layer2 + max pooling;
fully connected layer1 + dropout;
fully connected layer2 to prediction.

Derived from: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Membrane Parameters
N_CLASSES = 5  # CHANGE HERE, total number of classes
DATA_HEIGHT = 5000  # CHANGE HERE, the membrane's height
DATA_WIDTH = 20  # CHANGE HERE, the membrane's width
CHANNELS = 1  # change to 1 if grayscale


def read_membrane_datasets(dataset_path):
    """读取 Membrane 数据集

    Args:
     dataset_path: 数据集路径

    Returns:
     返回 Membrane 文件列表及对应分类列表
    """
    membrane_paths, membrane_labels = list(), list()
    # 获取文件夹下所有数据集文件名
    files = os.listdir(dataset_path)
    for filename in files:
        filepath = os.path.join(dataset_path, filename)
        if os.path.isfile(filepath):
            f = open(filepath, 'r', encoding='utf-8')
            # 读取样本首行分类信息
            m_class_info = f.readline().split()
            f.close()  # Membrane 文件完整路径
            # 登记相关信息至训练集列表
            membrane_paths.append(filepath)
            membrane_labels.append(m_class_info[1])
    return membrane_paths, membrane_labels


def read_membrane_data(file_path):
    """读取 Membrane 样本矩阵信息

    Args:
     file_path: 样本文件名称

    Returns:
     numpy 格式矩阵
    """
    result = list()
    with open(file_path, 'r', encoding='utf-8') as f:
        # 读取首行分类信息
        for line in f.readlines():
            line = line.strip()
            # 过滤首行分类信息和文件末尾空行
            if len(line) > 0 and not line.startswith('>'):
                # 逐行存入矩阵信息
                result.append(line.split())
    # 转换至 numpy array 格式
    mat = np.array(result, dtype=np.int16)

    # 矩阵填充 padding with 0
    new_mat = np.lib.pad(
        mat, ((0, DATA_HEIGHT - mat.shape[0]), (0, 0)), 'constant', constant_values=0)
    return new_mat


def conv_net(x, n_classes, dropout, reuse, is_training):
    """ 基本卷积神经网络构建

    Args:
     x: 2d 输入矩阵
     n_classes: 输出分类数
     dropout: Dropout 概率
     reuse: 是否应用同样的权重矩阵
     is_training: 网络是否在训练状态

    Returns:
     全连接层输出

    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # data input is a 1-D vector of (DATA_HEIGHT*DATA_WIDTH) features
        # Reshape to format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        # -1 代表每次可以自定义样本输入数量
        # x = tf.reshape(x, shape=[-1, DATA_HEIGHT, DATA_WIDTH, 1])

        # Convolution Layer 1 with 32 filters and a kernel size of 5
        # 卷积层1：卷积核大小为 5x5，卷积核数量为 32， 激活函数使用 RELU
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # 采用 2x2 维度的最大化池化操作，步长为2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer 2 with 64 filters and a kernel size of 3
        # 卷积层2：卷积核大小为 3x3，卷积核数量为 64， 激活函数使用 RELU
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # 采用 2x2 维度的最大化池化操作，步长为2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        # 全链接层具有1024神经元
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        # 对全链接层的数据加入dropout操作，防止过拟合
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        # 输出层 (Softmax层)，对dropout层的输出Tensor，执行分类操作
        out = tf.nn.softmax(out) if not is_training else out

    return out


def run_model(d_path, l_rate, n_steps, b_size, k_prob, folds, seed=None):
    """运行 CNN 模型

    Args:
      d_path: 数据集路径
      l_rate: 学习率
      n_steps: 训练次数
      b_size: 每次训练取样大小
      k_prob: 训练过程单元保留比例 (Dropout rate)
      folds: 交叉验证次数
      seed: 随机数种子
    """
    # tf Graph input
    # 占位符
    X = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.int16)
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # 读取数据
    traning_set, training_labels = read_membrane_datasets(d_path)
    sample = read_membrane_data(traning_set[0])
    tf_sample = tf.convert_to_tensor(sample, dtype=tf.int16)
    tf_sample_file = tf.convert_to_tensor(traning_set[0], dtype=tf.string)
    tf_sample_label = tf.convert_to_tensor(int(training_labels[0]), dtype=tf.int16)

    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that share the same weights.
    # Create a graph for training
    logits_train = conv_net(X, N_CLASSES, keep_prob,
                            reuse=False, is_training=True)
    # Create another graph for testing that reuse the same weights
    logits_test = conv_net(X, N_CLASSES, keep_prob,
                           reuse=True, is_training=False)

    # Define loss and optimizer (with train logits, for dropout to take effect)
    # sparse_softmax_cross_entropy_with_logits() do not use the one hot version of labels
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(y, tf.int16))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Saver object
    # saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        # print(tf_sample)
        # print(sess.run([tf_sample, tf_sample_label, tf_sample_file]))

        # Save your model
        # saver.save(sess, 'membrane_tf_model')
if __name__ == "__main__":
    run_model("D:\\datasets\\membrane", l_rate=0.01, n_steps=10, b_size=100, k_prob=0.8, folds=10)
