""" Raw Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

convolutional layer1 + max pooling;
convolutional layer2 + max pooling;
fully connected layer1 + dropout;
fully connected layer2 to prediction.

Cited from: https://github.com/aymericdamien/TensorFlow-Examples/
Author: Aymeric Damien
"""

from __future__ import division, print_function, absolute_import
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
# 计算 ACC 混淆矩阵 输出 recall f1等指标
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample  # 添加 subsampling 工具类

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create some wrappers for simplicity
# 卷积层实现函数


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# 池化层实现函数


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
# 创建卷积神经网络


def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    # -1 代表每次可以自定义样本输入数量
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    # 下采样，压缩卷积层
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    # 输出层
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def model_run(l_rate, n_steps, b_size, k_prob, folds, seed=None):
    """运行CNN模型
    ----
    参数
    ----
    lr: 学习率
    n_steps: 训练次数
    b_size: 每次训练取样大小
    k_prob: 训练过程单元保留比例 (Dropout rate)
    k_fold: 交叉验证次数
    seed: 随机数种子
    """
    # Import MNIST data
    image_data_path = os.path.join(os.getcwd(), "datasets")
    from tensorflow.examples.tutorials.mnist import input_data
    # 验证集设为 0 便于实现自定义的 k-fold 交叉验证
    mnist = input_data.read_data_sets(
        image_data_path, one_hot=True, validation_size=0)

    # Training Parameters
    # 超参数（学习率、训练次数、显示步长等）
    learning_rate = l_rate
    num_steps = n_steps
    batch_size = b_size
    display_step = 10

    # Network Parameters
    # 数据集相关参数（手写数字数据集，图像大小28*28，累计输入单元784个，输出单元10个）
    num_input = 784  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)
    dropout = k_prob  # Dropout, probability to keep units

    # 设定 K-fold 分割器
    rs = KFold(n_splits=folds, shuffle=True, random_state=seed)
    # 生成 k-fold 训练集、验证集索引
    cv_index_set = rs.split(mnist.train.labels)
    k_fold_step = 1  # 初始化折数
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)

    # tf Graph input
    # 占位符
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, num_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Construct model
    # 构建模型
    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    # 定义损失函数和优化器
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    # 模型评估
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # k-fold cross-validation
    for train_index, test_index in cv_index_set:
        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            for step in range(1, num_steps + 1):
                # 每次选取一定数量样本进行计算
                # batch_x, batch_y = mnist.train.next_batch(batch_size)
                sample_index = resample(
                    train_index, replace=False, n_samples=batch_size)
                batch_x = mnist.train.images[sample_index]
                batch_y = mnist.train.labels[sample_index]
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={
                         X: batch_x, Y: batch_y, keep_prob: dropout})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    # 计算ACC时保留所有单元数
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={
                                         X: batch_x, Y: batch_y, keep_prob: 1.0})
                    print("\nTraining Step " + str(step) + ", Training Accuracy = " +
                          "{:.6f}".format(acc) + ", Minibatch Loss = " +
                          "{:.6f}".format(loss))

            print("\nTraining Finished!")

            # 测试集评估模型
            batch_test_x = mnist.train.images[test_index]
            batch_test_y = mnist.train.labels[test_index]
            # print(batch_test_y)

            lossTest, accTest, predVal = sess.run([loss_op, accuracy, prediction], feed_dict={
                                                  X: batch_test_x, Y: batch_test_y, keep_prob: 1.0})

            print("\nFold:", k_fold_step, ", Test Accuracy =", "{:.6f}".format(
                accTest), ", Test Loss =", "{:.6f}".format(lossTest), ", Test Size:", test_index.shape[0])
            # print(predVal)

            # One-hot 矩阵转换为原始分类矩阵
            argmax_test = np.argmax(batch_test_y, axis=1)
            argmax_pred = np.argmax(predVal, axis=1)
            # 暂存每次选中的测试集和预测结果
            test_cache = np.concatenate((test_cache, argmax_test))
            pred_cache = np.concatenate((pred_cache, argmax_pred))

        print("\n=================================================================================")

        # 每个fold训练结束后次数 +1
        k_fold_step += 1

    # 训练结束计算Precision、Recall、ACC、MCC等统计指标
    class_names = []
    pred_names = []
    for i in range(num_classes):
        class_names.append('Class ' + str(i + 1))
        pred_names.append('Pred C' + str(i + 1))

    # 混淆矩阵生成
    cm = confusion_matrix(test_cache, pred_cache)
    df = pd.DataFrame(data=cm, index=class_names, columns=pred_names)

    # 混淆矩阵添加一列代表各类求和
    df['Sum'] = df.sum(axis=1).values

    print("\n=== Model evaluation ===")
    print("\n=== Accuracy classification score ===")
    print("\nACC = {:.6f}".format(accuracy_score(test_cache, pred_cache)))
    print("\n=== Matthews Correlation Coefficient ===")
    from .utils import matthews_corrcoef
    print("\nMCC = {:.6f}".format(matthews_corrcoef(cm)))
    print("\n=== Confusion Matrix ===\n")
    print(df.to_string())
    print("\n=== Detailed Accuracy By Class ===\n")
    print(classification_report(test_cache, pred_cache,
                                target_names=class_names, digits=6))
