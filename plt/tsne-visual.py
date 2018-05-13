
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

fig = plt.figure()
ax = plt.axes(projection='3d')

df = pd.read_csv('../tsne/embedded_f3_output.csv')


# 第1列为分类标注名称
t_labels = df.values[:, 0].astype(np.int32)

X = df.values[:, 1].astype(np.float)
Y = df.values[:, 2].astype(np.float)
Z = df.values[:, 3].astype(np.float)

# 颜色标注
C = []
for i in t_labels:
    if i == 1:
        C.append('c')
    if i == 2:
        C.append('r')

ax.scatter3D(X, Y, Z, c=C, cmap='viridis', s=50, label=t_labels)

ax.set_xlabel('F0')
ax.set_ylabel('F1')
ax.set_zlabel('F2')

ax.set_xlim(X.min()/2, X.max()/2)
ax.set_ylim(Y.min()/2, Y.max()/2)
ax.set_zlim(Z.min()/2, Z.max()/4)

plt.show()
