""" Random Forest.

Implement Random Forest algorithm with TensorFlow.

Derived: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import KFold
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

__author__ = 'Min'

DISPLAY_STEP = 50


def run(inputFile, n_trees, m_nodes, random_s, epochs, folds):
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
    class_info = df.groupby(df.columns[0]).size()
    print('\n', class_info)
    n_class = class_info.values.size
    print("\nNumber of categories:{0}\n".format(n_class))
    # 总样本数
    num_samples = n_target.size

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
                                          num_trees=n_trees,
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
    resampled_index_set = rs.split(n_target)
    # 初始化折数
    k_fold_step = 1
    # 暂存每次选中的测试集和对应预测结果
    test_cache = pred_cache = np.array([], dtype=np.int)

    # 迭代训练 k-fold 交叉验证
    for train_index, test_index in resampled_index_set:
        # Start TensorFlow session
        sess = tf.train.MonitoredSession()
        # Set random seed
        tf.set_random_seed(random_s)
        # Run the initializer
        sess.run(init_vars)
        print("\nFold:", k_fold_step)
        # Training
        for i in range(1, num_steps + 1):
            # Prepare Data
            batch_x = n_features[train_index]
            batch_y = n_target[train_index]
            batch_size = train_index.shape[0]
            _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            # 输出训练结果
            if i % DISPLAY_STEP == 0 or i == 1:
                acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
                print("\nTraining Epoch:", '%06d' % i, "Train Accuracy:", "{:.6f}".format(acc),
                      "Train Loss:", "{:.6f}".format(l), "Train Size:", batch_size)

        # 输入测试数据
        batch_test_x = n_features[test_index]
        batch_test_y = n_target[test_index]
        batch_test_size = test_index.shape[0]
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
