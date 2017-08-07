"""
python learning
"""
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
# 打印第6行数据
print(iris.data[5, :])
# 打印第2列数据
print(iris.data[:, 1])
