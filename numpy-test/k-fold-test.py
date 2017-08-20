"""
Learning sklearn.model_selection.ShuffleSplit

ShuffleSplit is thus a good alternative to KFold cross validation
that allows a finer control on the number of iterations and the
proportion of samples on each side of the train / test split.
"""
import numpy as np
from sklearn.model_selection import ShuffleSplit
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入CSV数据
TRAIN_CSV = os.path.join(os.path.dirname(__file__), "d:\\datasets\\train_nitration_941_standard.csv")
train_set = np.genfromtxt(TRAIN_CSV, delimiter=',', skip_header=1)

# 分类矩阵为第一列数据
n_target = train_set[:, 0]

# 特征矩阵
n_features = train_set[:, 1:]

# 建立 K-fold 分割器
rs = ShuffleSplit(n_splits=10, test_size=0.1, train_size=0.9, random_state=10)

# 输出分割信息
print("\nK-ford info:", rs)

# 分割训练集、测试集，默认 10-fold
kfold = rs.get_n_splits(n_target)
cv_set = rs.split(n_target)

print("\nk-fold={%d}" % kfold)

for train_index, test_index in cv_set:
    print("\nTrain-index:\n", train_index, "\nTest-index:\n", test_index)