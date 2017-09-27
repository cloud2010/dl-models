# -*- coding: utf-8 -*-
"""
IFS-Curve plt
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA1 = "logs/mcc-c2-l3-e300-lr1e-2.csv"
DATA2 = "logs/mcc-sum-c2-l3-u512-e400-lr1e-3-d01.csv"
# 去读取数据
datasets_1 = np.genfromtxt(DATA1, delimiter=',')
datasets_2 = np.genfromtxt(DATA2, delimiter=',')
# 纵坐标 MCC
mcc_1 = datasets_1[:, 1]
mcc_2 = datasets_2[:, 1]

# 横坐标特征数
x = np.arange(1, mcc_1.size + 1)

# print(x)
# print(mcc)

# 激活 seaborn 绘图样式
sns.set()
plt.plot(x, mcc_1, linewidth="0.5", label="LR=1e-2")
plt.plot(189, mcc_1[188], 'co')
plt.plot(x, mcc_2, linewidth="0.5", label="LR=1e-3")
plt.ylim(0, 0.8)
plt.annotate("Max value:(f{0:d}, {1:.4f})".format(189, mcc_1[188]), xy=(
    189, mcc_1[188]), xytext=(189, 0.7), fontsize=14, arrowprops=dict(facecolor='black'))
plt.xlabel("Number of features")
plt.ylabel("MCC values")
plt.title("The IFS-curve yielded by Selu")
plt.legend(loc="upper right")
plt.show()
