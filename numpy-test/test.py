"""
python matplot learning
"""
import numpy as np
# from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# iris = datasets.load_iris()
# 打印第6行数据
# print(iris.data[5, :])
# 打印第2列数据
# print(iris.data[:, 1])

# np.linspace() 函数学习
# train_size = np.linspace(0.05, 1, 20)
# print(train_size)


def test_func(x, err=0.5):
    y = 10 - 1. / (x + 0.1)
    if err > 0:
        y = np.random.normal(y, err)
    return y


def make_data(N=40, error=1.0, random_seed=1):
    """
    make some test data
    """
    # randomly sample the data
    np.random.seed(1)
    X = np.random.random(N)[:, np.newaxis]
    y = test_func(X.ravel(), error)
    return X, y


X, y = make_data(120, error=1.0)

plt.scatter(X, y)
plt.show()
