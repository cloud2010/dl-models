"""
A FNN with RELU implementation example using TensorFlow library.
Author: Min
Date: 2017-08-10
Source: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

# 避免输出TensorFlow未编译CPU指令集信息
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设定参数
learning_rate = 0.01
training_epochs = 10  # 训练次数
batch_size = 211  # 每批次随机选择的样本数（留一法）
display_step = 1  # 训练步长
# dropoutRate = tf.placeholder(tf.float32)  # dropout 概率
# is_training = tf.placeholder(tf.bool)  # 是否在训练

# 导入CSV数据
TRAIN_NITRATION = os.path.join(os.path.dirname(
    __file__), "d:\\datasets\\train_nitration_941_standard.csv")
train_nitration_set = np.genfromtxt(
    TRAIN_NITRATION, delimiter=',', skip_header=1)

# Tensorboard 记录位置
logs_path = 'd://tmp'

# 分类矩阵为第一列数据
n_target = train_nitration_set[:, 0]

# 特征矩阵 212*941
n_features = train_nitration_set[:, 1:]

# 总样本数索引矩阵（留一法随机抽样使用）
total_index = np.arange(0, train_nitration_set.shape[0])

# 输出两个矩阵的Shape
print("Target Shape: ", n_target.shape)
print("Features Shape: ", n_features.shape)

# 建立新target矩阵 212 * 2
new_target = np.array([0, 0])
# 将 Label 1 or 2 转换为  [0, 1] or [1, 0]
for i in n_target:
    if i == 1:  # Label 1
        new_target = np.vstack((new_target, np.array([0, 1])))
    elif i == 2:  # Label 2
        new_target = np.vstack((new_target, np.array([1, 0])))

# 删除新target矩阵第一行空值
new_target = np.delete(new_target, 0, 0)

# 设定神经网络参数，建立2个隐藏层，输入层神经元个数为 n_features 列数
n_hidden_1 = 256  # 1st hidden layer number of features
n_hidden_2 = 256  # 2nd hidden layer number of features
n_input = 941  # features input
n_classes = 2  # Label (1 or 2)

# tf Graph input 建立输入输出矩阵
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


# 创建FNN模型
def multilayer_perceptron(x, weights, biases):
    """
    层模型构建，建立2个隐藏层
    ----
    参数
    ----
    x: 输入神经元矩阵
    weights: 权重矩阵
    biases: 偏置矩阵
    输出
    ---
    out_layer: 线性模型输出
    """
    # 1st Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # 2nd Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# 建立权重和偏置变量，默认分配随机值
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# 应用模型（FNN配合RELU激活函数）
pred = multilayer_perceptron(x, weights, biases)

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

# 启动会话
with tf.Session() as sess:
    sess.run(init)
    
    # 存储每次验证结果
    test_val = 0.
    # 迭代训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_target.shape[0] / batch_size)
        # 每次随机选取 batch_size 个数据，累计212个实例，每次选取100个
        for i in range(total_batch):
            # 随机不放回抽样矩阵索引
            sample_index = np.random.choice(
                total_index, batch_size, replace=False)
            # 生成抽样训练集(特征和分类结果)
            batch_x = n_features[sample_index]
            batch_y = new_target[sample_index]
            # 运行优化器进行训练
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # 计算平均成本
            avg_cost += c / total_batch
        # 打印每步训练结果
        if epoch % display_step == 0:
            print("\nEpoch:", '%04d' % (epoch + 1), "Avg-Cost=",
                  "{:.9f}".format(avg_cost), "Train-Size:", batch_size)
            accTrain, costTrain = sess.run([accuracy, cost], feed_dict={x: batch_x,
                                                                        y: batch_y})
            print("Train-Accuracy:", accTrain, "Train-Loss:", costTrain)

            # 留一法选择出测试集
            test_index = np.setdiff1d(total_index, sample_index)  # 测试集唯一索引
            batch_test_x = n_features[test_index]
            batch_test_y = new_target[test_index]

            # 代入 tf graph 测试集结果输出
            accTest, costTest = sess.run([accuracy, cost], feed_dict={x: batch_test_x,
                                                                      y: batch_test_y})
            # 累加每次验证结果
            test_val += accTest
            print("Validation-Accuracy:", accTest,
                  "Val-Loss:", costTest, "Test-size: 1")

    # 训练结束
    print("\nTest-times:", training_epochs, "Total-Accuracy:",
          "{:.3%}".format(test_val / training_epochs))
    print("Optimization Finished! ")
