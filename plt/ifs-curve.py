# -*- coding: utf-8 -*-
"""
IFS-Curve plt
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MCC_LR1E_2 = "logs/mcc-sum-c2-l3-u512-e300-lr1e-2-d01.csv"
MCC_LR1E_3 = "logs/mcc-sum-c2-l3-u512-e400-lr1e-3-d01.csv"
# 去读取数据
mcc_datasets_lr1e_2 = np.genfromtxt(MCC_LR1E_2, delimiter=',')
mcc_datasets_lr1e_3 = np.genfromtxt(MCC_LR1E_3, delimiter=',')
# 纵坐标 MCC
mcc_lr1e_2 = mcc_datasets_lr1e_2[:, 1]
mcc_lr1e_3 = mcc_datasets_lr1e_3[:, 1]

# 最大值及对应索引
max_lr1e_2 = mcc_lr1e_2.max()
max_index_lr1e_2 = mcc_lr1e_2.argmax(axis=0)
max_lr1e_3 = mcc_lr1e_3.max()
max_index_lr1e_3 = mcc_lr1e_3.argmax(axis=0)

# 横坐标特征数
x = np.arange(1, mcc_lr1e_2.size + 1)

# 激活 seaborn 绘图样式
sns.set()
# 第 1 条 IFS 曲线
plt.plot(x, mcc_lr1e_2, linewidth="0.5", color="C1", label="LR=1e-2")
# 第 2 条 IFS 曲线
plt.plot(x, mcc_lr1e_3, linewidth="0.5", color="C2", label="LR=1e-3")
# 最大点
plt.plot(max_index_lr1e_2, max_lr1e_2, color="C1", marker="*", markersize="14")
plt.plot(max_index_lr1e_3, max_lr1e_3, color="C2", marker="*", markersize="14")
# 纵坐标标尺
plt.ylim(0.1, 0.8)
# 文本标注
plt.annotate("Max:(f{0:d}, {1:.4f})".format(
    max_index_lr1e_2, max_lr1e_2), color="black", xy=(max_index_lr1e_2, max_lr1e_2))
plt.annotate("Max:(f{0:d}, {1:.4f})".format(
    max_index_lr1e_3, max_lr1e_3), color="black", xy=(max_index_lr1e_3, max_lr1e_3))
# 横坐标文本
plt.xlabel("Number of features")
# 纵坐标文本
plt.ylabel("MCC values")
# 图标题
plt.title("The IFS-curve yielded by Selu")
# 图标题位置
plt.legend(loc="upper right")
# 图显示
plt.show()
