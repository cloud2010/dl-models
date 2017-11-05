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
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.logging.set_verbosity(tf.logging.INFO)

# Membrane Parameters
N_CLASSES = 5  # CHANGE HERE, total number of classes
DATA_HEIGHT = 2000  # CHANGE HERE, the data height
DATA_WIDTH = 20  # CHANGE HERE, the data width
CHANNELS = 1  # change to 1 if grayscale


def get_datasets(dataset_path):
    """读取样本数据集

    Args:
     dataset_path: 数据集路径

    Returns:
     返回样本文件列表及对应分类列表
    """
    sample_paths, sample_labels = list(), list()
    # 获取文件夹下所有数据集文件名
    files = os.listdir(dataset_path)
    for filename in files:
        filepath = os.path.join(dataset_path, filename)
        if os.path.isfile(filepath):
            f = open(filepath, 'r', encoding='utf-8')
            # 读取样本首行分类信息
            m_class_info = f.readline().split()
            f.close()  # sample 文件完整路径
            # 登记相关信息至训练集列表
            sample_paths.append(filepath)
            sample_labels.append(int(m_class_info[1]) - 1)
    return sample_paths, sample_labels


def read_data(file_path):
    """读取样本矩阵信息

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
    mat = np.array(result, dtype=np.float)
    if mat.shape[0] > DATA_HEIGHT:
        # 超出 Max Height 则矩阵做截断
        new_mat = mat[0:DATA_HEIGHT, ]
    else:
        # 矩阵填充 padding with 0
        new_mat = np.lib.pad(mat, ((
            0, DATA_HEIGHT - mat.shape[0]), (0, 0)), 'constant', constant_values=0)
    # 标准化处理
    scale = StandardScaler()
    # 按列 Standardize features by removing the mean and scaling to unit variance
    scale.fit(new_mat)
    # 2-D array 转换为 3-D array [Height, Width, Channel]
    new_mat = scale.transform(new_mat)[:, :, np.newaxis]
    return new_mat.astype(np.float32)


def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    """ 基本卷积神经网络构建

    Args:
     x: 4-D 输入矩阵 [Batch Size, Height, Width, Channel] 在给定的 4-D input 与 filter下计算2D卷积
     n_classes: 输出分类数
     dropout: Dropout 概率
     reuse: 是否应用同样的权重矩阵
     is_training: 网络是否在训练状态

    Returns:
     全连接层输出

    """
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        """
        data input is a 1-D vector of (DATA_HEIGHT*DATA_WIDTH) features
        Reshape to format [Height x Width x Channel]
        Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        特征矩阵调整为 4-D 输入，-1 代表每次可以自定义样本输入数量
        在给定的 4-D input与 filter下计算2D卷积
        """
        # TF Estimator 的输入为字典结构，从而可以更加通用
        x = x_dict['membranes']

        # Convolution Layer 1 with 32 filters and a kernel size of 5
        # 卷积层1：卷积核大小为 5x5，卷积核数量为 32， 激活函数使用 RELU
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # 采用 2x2 维度的最大化池化操作，步长为2
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer 2 with 64 filters and a kernel size of 3
        # 卷积层2：卷积核大小为 3x3，卷积核数量为 64， 激活函数使用 RELU
        conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # 采用 2x2 维度的最大化池化操作，步长为2
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(pool2)

        # Fully connected layer (in contrib folder for now)
        # 全链接层具有1024神经元
        fc1 = tf.layers.dense(fc1, units=1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        # 对全链接层的数据加入dropout操作，防止过拟合
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
    return out


def model_fn(features, labels, mode, params):
    """ Define the model function (following TF Estimator Template)
    使用TensorFlow Estimator 构建神经网络模型

    Args:
    features: 输入矩阵，卷积为 4-D array
    labels: 分类矩阵 [Batch Size, class]
    mode: 预测模式(训练 / 评估 / 预测)
    dropout: 神经单元丢弃比例

    Returns:
    `EstimatorSpec` fully defines the model to be run by an `Estimator`.
    """
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(
        features, N_CLASSES, params['dropout_rate'], reuse=False, is_training=True)
    logits_test = conv_net(
        features, N_CLASSES, params['dropout_rate'], reuse=True, is_training=False)

    # Predictions
    # 一个转换为具体分类，一个输出为概率
    pred_classes = tf.argmax(logits_test, axis=1)
    # pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


def run_model(d_path, l_rate, n_steps, b_size, d_rate, folds, seed=None):
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
    # 构建 Hyperparameters 字典
    model_params = {"dropout_rate": d_rate, "learning_rate": l_rate}

    # 构建 Estimator
    model = tf.estimator.Estimator(model_fn, params=model_params, model_dir="D:\\")

    # 读取数据
    traning_set, training_labels = get_datasets(d_path)

    # 选取一部分索引进行测试
    sample_list = read_data(traning_set[b_size]).reshape(
        1, DATA_HEIGHT, DATA_WIDTH, 1)
    label_list = training_labels[b_size]
    for i in range(b_size):
        sample_list = np.vstack((sample_list, read_data(
            traning_set[i]).reshape(1, DATA_HEIGHT, DATA_WIDTH, 1)))
        label_list = np.hstack((label_list, training_labels[i]))
    # print(sample_list[0])
    print(sample_list.shape)
    print(label_list.shape)

    # 定义训练样本数据输入方法和相关参数配置
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'membranes': sample_list}, y=label_list, num_epochs=n_steps, shuffle=True)

    # 训练模型
    model.train(input_fn=train_input_fn)

    # 模型评估
    # 定义测试样本数据输入方法和相关参数配置
    # test_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={'membranes': mnist.test.images}, y=mnist.test.labels,
    #     batch_size=batch_size, shuffle=False)

    # Use the Estimator 'evaluate' method
    # e = model.evaluate(input_fn)

    # print("Traning Accuracy:", e['accuracy'])
    # print("Traning Loss:", e['loss'])

    # Save your model
    # saver.save(sess, 'membrane_tf_model')
if __name__ == "__main__":
    run_model("D:\\datasets\\membrane", l_rate=0.01,
              n_steps=5, b_size=100, d_rate=0.2, folds=10)
