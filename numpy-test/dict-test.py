"""
python dict learning
"""
import numpy as np

# 隐藏层数
n_hiddens = 6

i = 0

# 定义权重矩阵
weights = {"out": "output layer"}

print(weights)

# 动态添加字典
for i in range(1, int(n_hiddens) + 1):
    weights["h" + str(i)] = "hidden layer{0}".format(i)

print(weights)

biases = {0: 0}
for i in range(1, n_hiddens + 1):
    biases[i] = i**2

print(biases)

# np merge
test0 = np.array([], dtype=np.int)
test1 = np.array([1, 2, 3, 5])
test2 = np.array([1, 2, 4, 6])
test3 = np.concatenate((test0, test1, test2))
print(test0)
print(test1)
print(test2)
print("\nAfter concatenate:", test3)
