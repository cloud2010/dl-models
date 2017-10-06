# -*- coding: utf-8 -*-
"""
rate_SNN_10-fold_citrullination_standard IFS-Curve plt
"""
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# 数据集文件名
MCC_SNN = "D:\\logs\\rate_SNN_10-fold_citrullination_standard.csv"
# 去读取数据
mcc_datasets = np.genfromtxt(MCC_SNN, delimiter=',')
# 纵坐标 MCC
mcc = mcc_datasets[:, 1]
# 最大值及对应索引
max_mcc = mcc.max()
max_mcc_index = mcc.argmax(axis=0) + 1
# 横坐标特征数
x = np.arange(1, mcc.size + 1)

# 激活 seaborn 绘图样式
# sns.set()
# 分割两个子图，共享 Y 轴，Two subplots, unpack the axes array immediately
# f, (plt, ax2) = plt.subplots(1, 2, sharey=True)

# IFS 曲线
plt.plot(x, mcc, linewidth="0.5", color="Blue", label="SNN, Depth=4, Unit=300")

# 最大点
plt.plot(max_mcc_index, max_mcc, color="Blue", marker="*", markersize="14")

# 文本标注
plt.annotate("Max:(F{0:d}, {1:.4f})".format(
    max_mcc_index, max_mcc), color="black", xy=(max_mcc_index + 10, max_mcc), fontsize="14")

# 横坐标文本
plt.xlabel("Number of features", fontsize="14")

# 纵坐标文本
plt.ylabel("MCC values", fontsize="14")

# 图标题
plt.title("The IFS-curve yielded by SNN (Hidden layers=4, Units=300)")

# 图标题位置
plt.legend(loc="upper right")

# 纵坐标标尺
plt.ylim(0.1, 0.8)

# 显示网格
plt.grid(True)

# 图显示
plt.show()
