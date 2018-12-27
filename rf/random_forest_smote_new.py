"""
TensorFlow RandomForest Estimator and SMOTE.
Project: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensor_forest
Ref: https://github.com/scikit-learn-contrib/imbalanced-learn
Date: 2018-12-27
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
from tensorflow.contrib.learn import SKCompat
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

__author__ = 'Min'

DISPLAY_STEP = 50


def run(inputFile, n_trees, m_nodes, random_s, epochs, folds, kneighbors, cores):
    """
    Random Forest 主程序
    参数
    ----
    inputFile: 训练集文件路径
    n_tress: 树大小
    m_nodes: 最大节点数
    random_s: 随机种子
    epochs: 每个fold的训练次数
    folds: k-fold折数
    kneighbors: k邻近点数
    cores: CPU核心数
    """
    # The number of cores per socket in the machine
    NUM_PARALLEL_EXEC_UNITS = cores
    try:
        # 导入训练数据
        df = pd.read_csv(inputFile, encoding='utf8')
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)

    # 分类矩阵为第一列数据(分类值减去1)
    n_target = df.iloc[:, 0].values.astype(np.int32)-1
    # 特征矩阵为去第一列之后数据
    n_features = df.iloc[:, 1:].values.astype(np.float32)
    # 读取特征名称
    features = df.columns[1:]
    print("\nDataset shape: ", df.shape, " Number of features: ", features.size)
    # 不同 Class 样本数统计 (根据第1列)
    np_target_y = np.asarray(np.unique(n_target, return_counts=True))
    df_y = pd.DataFrame(np_target_y.T, columns=['Class', 'Sum'])
    print("\nNumber of samples:\n{0}".format(df_y))
    # 总样本数及分类数
    num_samples = n_target.size
    n_class = df_y['Class'].size

    # Apply SMOTE 生成 fake data
    sm = SMOTE(k_neighbors=kneighbors)
    x_resampled, y_resampled = sm.fit_sample(n_features, n_target)
    # after over sampleing 读取分类信息并返回数量
    np_resampled_y = np.asarray(np.unique(y_resampled, return_counts=True))
    df_resampled_y = pd.DataFrame(np_resampled_y.T, columns=['Class', 'Sum'])
    print("\nNumber of samples after over sampleing:\n{0}\n".format(df_resampled_y))

    # 模型参数
    num_steps = epochs  # Total steps to train
    num_features = features.size
    num_trees = n_trees
    max_nodes = m_nodes

    # 设定 K-fold 分割器
    rs = KFold(n_splits=folds, shuffle=True, random_state=random_s)

    # Random Forest Parameters
    params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(num_classes=n_class,
                                                                         num_features=num_features,
                                                                         num_trees=num_trees,
                                                                         max_nodes=max_nodes)

    # Build the Random Forest
    classifier = SKCompat(
        tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params))

    # 生成 k-fold 索引
    resampled_index_set = rs.split(y_resampled)
    # 初始化折数
    k_fold_step = 1
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)

    # Change the parallelism threads and OpenMP* make use of all the cores available in the machine.
    # https://software.intel.com/en-us/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                            allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
    os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in resampled_index_set:

        print("\nFold:", k_fold_step)

        # Prepare Data
        batch_x = x_resampled[train_index]  # 特征数据用于训练
        batch_y = y_resampled[train_index]  # 标记结果用于验证
        batch_size = train_index.shape[0]
        # Training
        classifier.fit(x=batch_x, y=batch_y)
        # 输入测试数据
        # 验证测试集 (通过 index 去除 fake data)
        real_test_index = test_index[test_index < num_samples]
        batch_test_x = x_resampled[real_test_index]
        batch_test_y = y_resampled[real_test_index]
        batch_test_size = len(real_test_index)

        # 验证
        predVal = classifier.predict(x=batch_test_x)
        accTest = accuracy_score(batch_test_y, predVal['classes'])
        # print("\n", predVal['classes'])
        print("\nFold:", k_fold_step, "Test Accuracy:",
              "{:.6f}".format(accTest), "Test Size:", batch_test_size)
        # 暂存每次选中的测试集和预测结果
        test_cache = np.concatenate((test_cache, batch_test_y))
        pred_cache = np.concatenate((pred_cache, predVal['classes']))
        print("\n=========================================================================")
        # 每个fold训练结束后次数 +1

        k_fold_step += 1

    # 模型评估结果输出
    from .utils import model_evaluation
    model_evaluation(n_class, test_cache, pred_cache)
