"""
Forked from: https://github.com/aymericdamien/TensorFlow-Examples
A Multilayer Perceptron(MLP) implementation example using TensorFlow library.
Under Python 3.5 and TensorFlow 1.2
"""
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import os
mnist = input_data.read_data_sets("D://MNIST_data/", one_hot=True)
# 避免输出TensorFlow未编译CPU指令集信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设定参数
learning_rate = 0.001
training_epochs = 100  # 训练次数
batch_size = 100  # 每批次选择的样本数
display_step = 10  # 训练步长

# 神经网络参数，建立2个隐藏层，每层256个神经元
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input 建立输入输出矩阵
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# 创建模型，定义层，每层激活函数使用RELU
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
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
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

# 应用模型
pred = multilayer_perceptron(x, weights, biases)

# 定义成本函数和优化器
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # 交叉熵

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam优化器

# 初始化各个变量
init = tf.global_variables_initializer()

# 启动会话
with tf.Session() as sess:
    sess.run(init)

    # 迭代训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # 每次随机选取100个数据
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # 计算平均成本
            avg_cost += c / total_batch
        # 打印每步训练结果
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # 定义判别方法
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy: ", accuracy.eval(
        {x: mnist.test.images, y: mnist.test.labels}))
