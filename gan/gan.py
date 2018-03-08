""" Generative Adversarial Networks (GAN).

Using generative adversarial networks (GAN) to generate digit images from a
noise distribution.

References:
    - Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza,
    B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information
    processing systems, 2672-2680.
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256

Links:
    - [GAN Paper](https://arxiv.org/pdf/1406.2661.pdf).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
# 避免输出TensorFlow未编译CPU指令集信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 导入CSV数据
TRAIN_CSV = os.path.join(os.path.dirname(
    __file__), "../datasets/train_nitration_941_standard.csv")
# 去掉CSV文件标题行
train_set = np.genfromtxt(TRAIN_CSV, delimiter=',', skip_header=1)
# 分类矩阵为第一列数据
n_target = train_set[:, 0]
# 特征矩阵为去第一列之后数据
n_features = train_set[:, 1:]
# 样本数
nums_samples = n_target.shape[0]
# 特征数
nums_features = n_features.shape[1]
# 获取正负样本数据集
pos_samples = n_features[np.where(n_target == 1)]
nums_pos = np.sum(n_target == 1)
neg_samples = n_features[np.where(n_target == 2)]
nums_neg = np.sum(n_target == 2)

print("Sum of samples:", nums_samples)
print("Sum of features:", nums_features)

# 不同 Class 统计
for i in [1, 2]:
    print("Sum of Class {0}: {1}".format(i, np.sum(n_target == i)))

# Training Params
num_steps = 10000
batch_size = 128
learning_rate = 0.0002

# image_dim = 784  # 28*28 pixels
# Network Params
features_dim = nums_features
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100  # Noise data points

# A custom initialization (see Xavier Glorot init)


def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


# Store layers weight & bias
weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, features_dim])),
    'disc_hidden1': tf.Variable(glorot_init([features_dim, disc_hidden_dim])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}
biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([features_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1])),
}


# Generator
def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Discriminator
def discriminator(x):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)
    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


# Build Networks
# Network Inputs
gen_input = tf.placeholder(
    tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(
    tf.float32, shape=[None, features_dim], name='disc_input')

# Build Generator Network
gen_sample = generator(gen_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

# Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = [weights['gen_hidden1'], weights['gen_out'],
            biases['gen_hidden1'], biases['gen_out']]
# Discriminator Network Variables
disc_vars = [weights['disc_hidden1'], weights['disc_out'],
             biases['disc_hidden1'], biases['disc_out']]

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

start_time = time.time()
print("\nUse %d real positive samples as training input" % (nums_pos))
print("\nStart training...\n")
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(1, num_steps+1):
        # Prepare Negative Data
        # Get the next batch of Negative data (only features are needed, not labels)
        batch_x = pos_samples
        # Generate noise to feed to the generator
        # 噪声数据样本数和训练集样本数一致
        z = np.random.uniform(-1., 1., size=[nums_pos, noise_dim])

        # Train
        feed_dict = {disc_input: batch_x, gen_input: z}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i: Time: %4.4fs Generator Loss: %f, Discriminator Loss: %f' % (
                i, time.time() - start_time, gl, dl))

    # After training generate fake data from noise, using the generator network.
    # Noise input.
    z = np.random.uniform(-1., 1., size=[100, noise_dim])
    # gen_sample is trained generator network
    g = sess.run([gen_sample], feed_dict={gen_input: z})
    g_data = np.array(g[0])
    # output fake negative data
    print("\nFake data shape:", g_data.shape)
    print("\n----- Export generated positive data to CSV -----")
    # print(g)
    # print(len(g))
    # export generated data to CSV file
    # 特征编号及labels
    f_list = ["f" + str(i) for i in range(1, nums_features+1)]
    labels = np.full(100, 1)
    # 将所有生成的负样本矩阵写入DataFrame
    df = pd.DataFrame(g_data, index=labels, columns=f_list)
    # 将数据库写入CSV
    df.to_csv("g_pos_data.csv", index_label="Class")
