"""
TensorFlow MINIST Sample
Source: https://www.tensorflow.org/get_started/mnist/beginners
"""
import tensorflow as tf
# 避免输出TensorFlow未编译CPU指令集信息
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D://MNIST_data/", one_hot=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2维输入 784个像素点 28*28=784
# Here 'None' means that a dimension can be of any length
x = tf.placeholder(tf.float32, [None, 784])

# 设定权重和偏置变量，默认初始值为0
# 权重784行，10列（0-9累计10种类型数字）
# 偏置10行，代表10类数字的偏置量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 建立神经网络激活函数（Softmax）
# fun(x) = softmax(W*x+b)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练模型
# 建立成本函数，目的通过训练降低 loss，此处选择"交叉熵"
# 第一步，构建占位符（2维矩阵）,0-9累计10种数字的可能概率
y_ = tf.placeholder(tf.float32, [None, 10])
# 交叉熵定义：y 是我们预测的概率分布, y_ 是实际的分布
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 定义优化器，如何降低 loss
# 默认选择梯度下降，learningRate = 0.05
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# 建立会话，此处选择 InteractiveSession
# 也可以是 sess = tf.Session()
sess = tf.InteractiveSession()
# 初始化变量
tf.global_variables_initializer().run()
# 训练1000次，每次随机选取100组数字
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
"""
使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）
在这里更确切的说是随机梯度下降训练，每一次训练我们可以使用不同的数据子集，
这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性
"""

# 模型评估
"""
tf.argmax 是一个非常有用的函数，
它能给出某个 tensor 对象在某一维上的其数据最大值所在的索引值。
由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，
比如 tf.argmax(y,1) 返回的是模型对于任一输入x预测到的标签值，
而 tf.argmax(y_,1) 代表正确的标签，
我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
"""
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 准确度定义，即把布尔值转换成浮点数，然后取平均值求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 最后用测试集 test data 验证模型准确率
print("The MINIST Predict Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))