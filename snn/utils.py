# -*- coding: utf-8 -*-
"""
The :mod:`utils` module includes various utilities.
"""
import math
import numpy as np

__author__ = "Min"


def matthews_corrcoef(c_matrix):
    """
    多分类问题MCC计算
    MCC = cov(X, Y) / sqrt(cov(X, X)*cov(Y, Y))
    Ref: http://scikit-learn.org/stable/modules/model_evaluation.html
         https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    t_k=\sum_{i}^{K} C_{ik} the number of times class K truly occurred
    p_k=\sum_{i}^{K} C_{ki} the number of times class K was predicted
    c=\sum_{k}^{K} C_{kk} the total number of samples correctly predicted
    s=\sum_{i}^{K} \sum_{j}^{K} C_{ij} the total number of samples
    参数
    ----
    c_matrix: 混淆矩阵 array, shape = [n_classes, n_classes]
    返回
    ----
    mcc: Matthews correlation coefficient, float
    """
    # 先获取分类数
    cm_classes = c_matrix.shape[0]
    # 初始化变量
    t_k = np.zeros(cm_classes)
    p_k = np.zeros(cm_classes)
    c = 0
    s = c_matrix.sum()
    for i in range(cm_classes):
        # 计算相关变量值
        c += c_matrix[i, i]
        t_k[i] = c_matrix[i, :].sum()
        p_k[i] = c_matrix[:, i].sum()

    sum_tk_dot_pk = np.array([t_k[i] * p_k[i]
                              for i in range(cm_classes)]).sum()
    sum_power_tk = np.array([t_k[i]**2 for i in range(cm_classes)]).sum()
    sum_power_pk = np.array([p_k[i]**2 for i in range(cm_classes)]).sum()
    # 计算 MCC
    mcc = (c * s - sum_tk_dot_pk) / \
        math.sqrt((s**2 - sum_power_pk) * (s**2 - sum_power_tk))

    # 返回 MCC
    return mcc
    # return mcc, t_k, p_k
