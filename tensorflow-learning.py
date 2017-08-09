import tensorflow as tf
# 避免输出TensorFlow未编译CPU指令集信息
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 常量f
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0, dtype=tf.float32)

# 打印
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
