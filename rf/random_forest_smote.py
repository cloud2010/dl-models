""" Random Forest.

Implement Random Forest algorithm with TensorFlow and SMOTE.

Derived: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
Ref: https://github.com/scikit-learn-contrib/imbalanced-learn
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import KFold
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from imblearn.over_sampling import SMOTE

__author__ = 'Min'

DISPLAY_STEP = 50
NUM_PARALLEL_EXEC_UNITS = 4


def run(inputFile, n_trees, m_nodes, random_s, epochs, folds, kneighbors):
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
    """
    try:
        # 导入训练数据
        df = pd.read_csv(inputFile, encoding='utf8')
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)

    # 分类矩阵为第一列数据(分类值减去1)
    n_target = df.iloc[:, 0].values-1
    # 特征矩阵为去第一列之后数据
    n_features = df.iloc[:, 1:].values
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

    # Input and Target data
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # For random forest, labels must be integers (the class id)
    Y = tf.placeholder(tf.int32, shape=[None])

    # Random Forest Parameters
    hparams = tensor_forest.ForestHParams(num_classes=n_class,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes).fill()

    # Build the Random Forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    # print(forest_graph.params.__dict__)
    # Get training graph and loss
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    # Measure the accuracy
    infer_op, _, _ = forest_graph.inference_graph(X)
    pred_val = tf.argmax(infer_op, 1)
    correct_prediction = tf.equal(pred_val, tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    # init = tf.global_variables_initializer()

    # Initialize the variables (i.e. assign their default value) and forest resources
    init_vars = tf.group(tf.global_variables_initializer(),
                         resources.initialize_resources(resources.shared_resources()))

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
        # Start TensorFlow session
        # sess = tf.train.MonitoredSession()
        sess = tf.Session(config=config)
        # Set random seed
        tf.set_random_seed(random_s)
        # Run the initializer
        sess.run(init_vars)
        print("\nFold:", k_fold_step)
        # Training
        for i in range(1, num_steps + 1):
            # Prepare Data
            batch_x = x_resampled[train_index]  # 特征数据用于训练
            batch_y = y_resampled[train_index]  # 标记结果用于验证
            batch_size = train_index.shape[0]
            _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            # 输出训练结果
            if i % DISPLAY_STEP == 0 or i == 1:
                acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
                print("\nTraining Epoch:", '%06d' % i, "Train Accuracy:", "{:.6f}".format(acc),
                      "Train Loss:", "{:.6f}".format(l), "Train Size:", batch_size)

        # 输入测试数据
        # 验证测试集 (通过 index 去除 fake data)
        real_test_index = test_index[test_index < num_samples]
        batch_test_x = x_resampled[real_test_index]
        batch_test_y = y_resampled[real_test_index]
        batch_test_size = len(real_test_index)

        # 代入TensorFlow计算图验证测试集
        accTest, costTest, predVal = sess.run([accuracy_op, loss_op, pred_val], feed_dict={
            X: batch_test_x, Y: batch_test_y})
        # print("\n", predVal)
        print("\nFold:", k_fold_step, "Test Accuracy:", "{:.6f}".format(
            accTest), "Test Loss:", "{:.6f}".format(costTest), "Test Size:", batch_test_size)
        # 暂存每次选中的测试集和预测结果
        test_cache = np.concatenate((test_cache, batch_test_y))
        pred_cache = np.concatenate((pred_cache, predVal))
        print("\n=========================================================================")
        # 每个fold训练结束后次数 +1

        k_fold_step += 1

    # 模型评估结果输出
    from .utils import model_evaluation
    model_evaluation(n_class, test_cache, pred_cache)
