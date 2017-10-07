# -*- coding: utf-8 -*-
"""
IFS-Curve plt
"""
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

MCC_LR1E_1_D5 = "logs/mcc-sum-c2-l5-u512-e500-lr1e-1-d01.csv"
MCC_LR1E_2_D5 = "logs/mcc-sum-c2-l5-u512-e800-lr1e-2-d01.csv"
MCC_LR1E_3_D5 = "logs/mcc-sum-c2-l5-u512-e600-lr1e-3-d01.csv"
MCC_LR1E_1_D3 = "logs/mcc-sum-c2-l3-u512-e500-lr1e-1-d01.csv"
MCC_LR1E_2_D3 = "logs/mcc-sum-c2-l3-u512-e300-lr1e-2-d01.csv"
MCC_LR1E_3_D3 = "logs/mcc-sum-c2-l3-u512-e400-lr1e-3-d01.csv"
# 去读取数据
mcc_datasets_lr1e_1_d5 = np.genfromtxt(MCC_LR1E_1_D5, delimiter=',')
mcc_datasets_lr1e_2_d5 = np.genfromtxt(MCC_LR1E_2_D5, delimiter=',')
mcc_datasets_lr1e_3_d5 = np.genfromtxt(MCC_LR1E_3_D5, delimiter=',')
mcc_datasets_lr1e_1_d3 = np.genfromtxt(MCC_LR1E_1_D3, delimiter=',')
mcc_datasets_lr1e_2_d3 = np.genfromtxt(MCC_LR1E_2_D3, delimiter=',')
mcc_datasets_lr1e_3_d3 = np.genfromtxt(MCC_LR1E_3_D3, delimiter=',')
# 纵坐标 MCC
mcc_lr1e_1_d5 = mcc_datasets_lr1e_1_d5[:, 1]
mcc_lr1e_2_d5 = mcc_datasets_lr1e_2_d5[:, 1]
mcc_lr1e_3_d5 = mcc_datasets_lr1e_3_d5[:, 1]
mcc_lr1e_1_d3 = mcc_datasets_lr1e_1_d3[:, 1]
mcc_lr1e_2_d3 = mcc_datasets_lr1e_2_d3[:, 1]
mcc_lr1e_3_d3 = mcc_datasets_lr1e_3_d3[:, 1]

# 最大值及对应索引
max_lr1e_1_d5 = mcc_lr1e_1_d5.max()
max_index_lr1e_1_d5 = mcc_lr1e_1_d5.argmax(axis=0) + 1
max_lr1e_2_d5 = mcc_lr1e_2_d5.max()
max_index_lr1e_2_d5 = mcc_lr1e_2_d5.argmax(axis=0) + 1
max_lr1e_3_d5 = mcc_lr1e_3_d5.max()
max_index_lr1e_3_d5 = mcc_lr1e_3_d5.argmax(axis=0) + 1
max_lr1e_1_d3 = mcc_lr1e_1_d3.max()
max_index_lr1e_1_d3 = mcc_lr1e_1_d3.argmax(axis=0) + 1
max_lr1e_2_d3 = mcc_lr1e_2_d3.max()
max_index_lr1e_2_d3 = mcc_lr1e_2_d3.argmax(axis=0) + 1
max_lr1e_3_d3 = mcc_lr1e_3_d3.max()
max_index_lr1e_3_d3 = mcc_lr1e_3_d3.argmax(axis=0) + 1

# 横坐标特征数
x = np.arange(1, 942)

# 激活 seaborn 绘图样式
# sns.set()
# 分割两个子图，共享 Y 轴，Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# IFS 曲线
ax1.plot(x, mcc_lr1e_1_d5, linewidth="0.5", color="C0", label="LR=1e-1, Depth=5")
ax1.plot(x, mcc_lr1e_2_d5, linewidth="0.5", color="C1", label="LR=1e-2, Depth=5")
ax1.plot(x, mcc_lr1e_3_d5, linewidth="0.5", color="C2", label="LR=1e-3, Depth=5")
# 最大点
ax1.plot(max_index_lr1e_1_d5, max_lr1e_1_d5, color="C0", marker="*", markersize="14")
ax1.plot(max_index_lr1e_2_d5, max_lr1e_2_d5, color="C1", marker="*", markersize="14")
ax1.plot(max_index_lr1e_3_d5, max_lr1e_3_d5, color="C2", marker="*", markersize="14")
# 文本标注
ax1.annotate("Max:(F{0:d}, {1:.4f})".format(
    max_index_lr1e_1_d5, max_lr1e_1_d5), color="black", xy=(max_index_lr1e_1_d5, max_lr1e_1_d5))
ax1.annotate("Max:(F{0:d}, {1:.4f})".format(
    max_index_lr1e_2_d5, max_lr1e_2_d5), color="black", xy=(max_index_lr1e_2_d5, max_lr1e_2_d5))
ax1.annotate("Max:(F{0:d}, {1:.4f})".format(
    max_index_lr1e_3_d5, max_lr1e_3_d5), color="black", xy=(max_index_lr1e_3_d5, max_lr1e_3_d5))
# IFS 曲线
ax2.plot(x, mcc_lr1e_1_d3, linewidth="0.5", color="C3", label="LR=1e-1, Depth=3")
ax2.plot(x, mcc_lr1e_2_d3, linewidth="0.5", color="C4", label="LR=1e-2, Depth=3")
ax2.plot(x, mcc_lr1e_3_d3, linewidth="0.5", color="C5", label="LR=1e-3, Depth=3")
# 最大点
ax2.plot(max_index_lr1e_1_d3, max_lr1e_1_d3, color="C3", marker="*", markersize="14")
ax2.plot(max_index_lr1e_2_d3, max_lr1e_2_d3, color="C4", marker="*", markersize="14")
ax2.plot(max_index_lr1e_3_d3, max_lr1e_3_d3, color="C5", marker="*", markersize="14")
# 文本标注
ax2.annotate("Max:(F{0:d}, {1:.4f})".format(
    max_index_lr1e_1_d3, max_lr1e_1_d3), color="black", xy=(max_index_lr1e_1_d3, max_lr1e_1_d3))
ax2.annotate("Max:(F{0:d}, {1:.4f})".format(
    max_index_lr1e_2_d3, max_lr1e_2_d3), color="black", xy=(max_index_lr1e_2_d3, max_lr1e_2_d3))
ax2.annotate("Max:(F{0:d}, {1:.4f})".format(
    max_index_lr1e_3_d3, max_lr1e_3_d3), color="black", xy=(max_index_lr1e_3_d3, max_lr1e_3_d3))

# 横坐标文本
ax1.set_xlabel("Number of features")
ax2.set_xlabel("Number of features")
# 纵坐标文本
ax1.set_ylabel("MCC values")
# 图标题
ax1.set_title("The IFS-curve yielded by SNN (Layers=5)")
ax2.set_title("The IFS-curve yielded by SNN (Layers=3)")
# 图标题位置
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")
# 纵坐标标尺
plt.ylim(0.1, 0.7)
# 图显示
ax1.grid(True)
ax2.grid(True)
plt.show()
