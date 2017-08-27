"""
Learning sklearn.model_selection.ShuffleSplit and KFold

Ref: http://scikit-learn.org/stable/modules/cross_validation.html
"""
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.utils import resample
import os
import logging  # 导入日志模块

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='d://train.log',
                    filemode='w')

# 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误
# 并将其添加到当前的日志处理对象
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入CSV数据
TRAIN_CSV = os.path.join(os.path.dirname(__file__), "d:\\datasets\\train_nitration_941_standard.csv")
train_set = np.genfromtxt(TRAIN_CSV, delimiter=',', skip_header=1)

# 分类矩阵为第一列数据
n_target = train_set[:, 0]

# 特征矩阵
n_features = train_set[:, 1:]

"""
Shuffle 分割器，类似k-fold，默认随机抽取
但为保证测试集训练集大小一致，所以存在多个测试样本重复
"""
# rs = ShuffleSplit(n_splits=10, test_size=0.1, train_size=0.9, random_state=10)

"""
K-fold 介绍
shuffle 参数为 True 非顺序抽取，即先洗牌后切割，反之顺序抽取 Test-dataset
shuffle = True 才可以激活随机种子 random_state，通常我们是这种做法
通过设定 random_state 确保每次随机排列一致
"""
rs = KFold(n_splits=10, shuffle=True, random_state=10)

# 输出分割信息
print("\nK-ford info:", rs)

# 分割训练集、测试集，默认 10-fold
kfold = rs.get_n_splits(n_target)
cv_set = rs.split(n_target)

print("\nk-fold={%d}" % kfold)

for train_index, test_index in cv_set:
    print("\nTrain-index:\n", train_index, "\nTest-index:\n", test_index)
    batch_x = n_target[train_index]
    # Sub-sampling
    batch_sample_index = resample(train_index, replace=False, n_samples=50)
    print("\nSub-Sample-index:", batch_sample_index)
    print("\nTest dataset size:", test_index.shape)
