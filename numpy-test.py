"""
Numpy learning
"""
import numpy as np
# import tensorflow as tf
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入CSV数据
TRAIN_NITRATION = os.path.join(os.path.dirname(
    __file__), "d:\\train_nitration_941_standard.csv")
train_nitration_set = np.genfromtxt(
    TRAIN_NITRATION, delimiter=',', skip_header=1)

# 分类矩阵为第一列数据
n_target = train_nitration_set[:, 0]

# 特征矩阵 212*941
n_features = train_nitration_set[:, 1:]

# 建立新target矩阵 212 * 2
new_target = np.array([0, 0])
# print(n_target)
# print(np.where(n_target == 2))
for i in n_target:
    if i == 1:  # Label 1
        new_target = np.vstack((new_target, np.array([0, 1])))
    elif i == 2:  # Label 2
        new_target = np.vstack((new_target, np.array([1, 0])))

# 删除第一行空值
new_target = np.delete(new_target, 0, 0)
# print(new_target)
# print("New target shape:", new_target.shape)
# 输出两个矩阵的Shape
# print("Target Shape: ", n_target.shape)
# print("Features Shape: ", n_features.shape)
# print("Train Set Shape:", train_nitration_set.shape)

"""
留一法测试
"""
# 总索引数
total_index = np.arange(0, 212)
# 不放回随机抽样
sample_index = np.random.choice(total_index, size=211, replace=False)

# print(sample_index)
# print(total_index)

# 打印差集
# print(np.setdiff1d(total_index, sample_index))

"""
留一法测试2
"""
test_index = np.array([3])
print("\nTest-index:", test_index)
# 打印去除测试集索引的差集
print("\nTrain-index:\n", np.setdiff1d(total_index, test_index))

"""
矩阵元素统计
"""
print("\nSum of Label2: ", np.sum(n_target == 2),
      "\nSum of Label1: ", np.sum(n_target == 1))

# 输出抽样矩阵
# print(n_target[sample_index])
# 输出转置的target矩阵
# n_target_sample = n_target[sample_index]
# n_target_t = np.transpose([n_target_sample])
# print(n_target[sample_index].reshape(len(n_target[sample_index]), -1))
# print(n_target_t)

"""
矩阵判别运算
"""
test_a = np.array([0, 1])
test_b = np.array([1, 0])
print("\nTest np.equal func:\n")
print(np.equal(test_a, np.array([0, 1])))
print(np.equal(test_a, np.array([0, 0])))
print(np.equal(test_a, np.array([1, 0])))

print("\nTest_a argmax:", np.argmax(test_a))
print("\nTest_b argmax:", np.argmax(test_b))


def random_sample(train_set, size=10, isReplace=True):
    """
    训练集随机放回抽样
    ---
    参数
    ---
    train_set: 训练集或数据集矩阵
    size: 抽样数，默认10个
    isReplace: 是否放回抽样
    返回
    ---
    target: 训练集抽样的分类结果矩阵
    sample_input: 训练集抽样的输入矩阵
    """
    # 计算机训练集实例数（训练集的行数）
    instance_num = np.arange(1, train_set.shape[0])
    # 随机生成抽样行
    sample_row = np.random.choice(instance_num, size, isReplace)
    # 输出抽样行矩阵切片（切片列数为训练集的列数）
    sample_set = np.empty(shape=[train_set.shape[1], ])
    # 循环抽取对应行构建出抽样矩阵
    for i in sample_row:
        sample_set = np.vstack((sample_set, train_set[i, :]))
    # 剔除第一行初始值
    sample_set = np.delete(sample_set, 0, 0)
    # 返回抽样的分类结果和输入数据（通常是特征值），默认第一列为target（分类结果）
    target = sample_set[:, 0]
    features = sample_set[:, 1:]
    return (target, features)

# 测试抽样结果
# batch_x, batch_y = random_sample(train_nitration_set, 10, True)
# print("batch_x:", batch_x)

# 开始测试 TensorFlow 赋值
# node1 = tf.constant(batch_x, dtype=tf.float32)
# print("node1:", node1)

# sess = tf.Session()
# print("node1:", sess.run(node1))
