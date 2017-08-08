"""
交叉验证可视化学习
"""

from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

# Activate Seaborn Style
sns.set()


def plot_with_err(x, data, **kwargs):
    """
    概述
    ---
    描绘曲线包含错误值阴影
    参数
    ---
    x: x轴输入
    data: y轴输入
    **kwargs: 其它参数，例如label=绘图标题
    """
    # 得分结果转换(均值和标准差)
    mu, std = -np.mean(data, axis=1), np.std(data, axis=1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                     facecolor=lines[0].get_color(), alpha=0.2)


# 学习曲线 函数学习，train_sizes 分割为10份，交叉验证10折，分类器SVM
train_sizes, val_train, val_test = learning_curve(
    SVC(gamma=0.01), X, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10))

# 训练得分转换
# train_scores = -np.mean(val_train, axis=1)

# 测试集验证结果得分转换
# test_scores = -np.mean(val_test, axis=1)

# 学习曲线可视化
plot_with_err(train_sizes, val_train, label='training scores')
plot_with_err(train_sizes, val_test, label='validation scores')
plt.xlabel('Training Set Size')
plt.ylabel('neg mean squared error')
plt.legend()
plt.show()

print(train_sizes)
print(val_train)
print(val_test)
