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

import tensorflow as tf
import os

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


def model_run(l_rate, n_steps, b_size, k_prob):
    """运行CNN模型
    ----
    参数
    ----
    lr: 学习率
    n_steps: 训练次数
    b_size: 每次训练取样大小
    k_prob: 训练过程单元保留比例 (Dropout rate)
    """
    # Import MNIST data
    image_data_path = os.path.join(os.getcwd(), "datasets")
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(image_data_path, one_hot=True)

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

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, num_steps + 1):
            # 每次选取一定数量样本进行计算
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                # 计算ACC时保留所有单元数
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.4f}".format(acc))

        print("\nOptimization Finished!")

        # Calculate accuracy for 256 MNIST test images
        # 用测试集进行验证，保留所有单元数
        print("\n256 test images accuracy:",
              sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                            Y: mnist.test.labels[:256],
                                            keep_prob: 1.0}))
