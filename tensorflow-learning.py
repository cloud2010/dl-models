"""
TensorFlow basic operation learning
"""
import tensorflow as tf
# 避免输出TensorFlow未编译CPU指令集信息
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 常量
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)

# 打印节点
print(node1, node2)

# 创建会话sess正式运行，输出节点信息
sess = tf.Session()
print(sess.run([node1, node2]))

# 创建节点运算符
node3 = tf.add(node1, node2)
print(node3)
print('node3: ', sess.run(node3))

# 创建占位符，不提前赋值
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# 会话运行时进行赋值
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# 运算符亦可当做输入参数
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# 创建变量 Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# 变量在进入sess运行时务必先初始化，并且执行sess.run()之后进行赋值
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# 整合
y = tf.placeholder(tf.float32)
# squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(tf.square(linear_model - y))
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# tf.assign() 方法实现对变量赋值
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
