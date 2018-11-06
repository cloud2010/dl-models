""" Graph Convolutional Networks.

Implement Graph Convolutional Networks with TensorFlow,
and apply it to classify membrane pseudo datasets.

Author: Min
Date: 2018-11-6
Ref: https://github.com/tkipf/gcn
"""

import time
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

def evaluate(features, support, labels, mask, placeholders):
    """  
    模型评估函数
    """
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


def run(i_features, i_adj, n_class, h_units, epochs, folds, l_rate, d_rate, seed=123):
    """
    GCN主程序
    参数
    ----
    i_features: 特征矩阵(训练集)
    i_adj: 邻接矩阵(训练集)
    n_class: 分类数，即输出层单元数，默认2分类问题
    h_units: 隐藏层单元数
    epochs: 每个fold的训练次数
    folds: 交叉验证次数
    l_rate: Learning rate
    d_rate: Dropout rate
    seed: 随机种子
    """
    try:
        # 加载 CSV 数据
        df_features = pd.read_csv(i_features, encoding='utf8')
        df_adj = pd.read_csv(i_adj, encoding='utf8')
    except (OSError) as e:
        print("\n\t", e)
        print("\nPlease make sure you input correct filename of training dataset!")
        sys.exit(1)

    # 设定随机种子
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # GCN 默认参数设置
    weight_decay = 5e-4 # Weight for L2 loss on embedding matrix.
    early_stopping = 10 # Tolerance for early stopping (# of epochs).
    max_degree = 3 # Maximum Chebyshev polynomial degree.

    # 数据装载及预处理
    keys = df_adj.loc[:, 0].unique() # select first column and get unique values
    edge_dict = {key: df_adj[df_adj[0] == key].loc[:, 1].values.tolist() for key in keys} # 根据第一列的 key 来过滤出所有第二列的值并转换为 dict
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(edge_dict)) # 构建邻接矩阵
    
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, n_class)), # one-hot vector shape
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=False)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: d_rate})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
            "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
