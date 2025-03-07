# -*- coding: utf-8 -*-
"""
The :mod:`utils` module includes various utilities.
"""

import datetime
import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

__author__ = "Min"


def matthews_corrcoef(c_matrix):
    """多分类问题MCC计算
    MCC = cov(X, Y) / sqrt(cov(X, X)*cov(Y, Y))
    Ref: http://scikit-learn.org/stable/modules/model_evaluation.html
         https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    
    参数
    ----
    c_matrix: 混淆矩阵 array, shape = [n_classes, n_classes]
    
    返回
    ----
    mcc: Matthews correlation coefficient, float
    """
    if not isinstance(c_matrix, np.ndarray) or len(c_matrix.shape) != 2 or c_matrix.shape[0] != c_matrix.shape[1]:
        raise ValueError("输入必须是一个方阵形式的混淆矩阵")
    
    if c_matrix.size == 0:
        return 0.0  # 空矩阵返回0
    
    cm_classes = c_matrix.shape[0]
    
    t_k = np.sum(c_matrix, axis=1)  # 实际类别出现次数
    p_k = np.sum(c_matrix, axis=0)  # 预测类别出现次数
    c = np.trace(c_matrix)  # 正确预测的总样本数
    s = np.sum(c_matrix)  # 总样本数
    
    sum_tk_dot_pk = (t_k * p_k).sum()
    sum_power_tk = (t_k ** 2).sum()
    sum_power_pk = (p_k ** 2).sum()
    
    denominator = math.sqrt((s**2 - sum_power_pk) * (s**2 - sum_power_tk))
    if denominator == 0:
        return 0.0  # 分母为零时返回0
    
    mcc = (c * s - sum_tk_dot_pk) / denominator
    return mcc


def generate_confusion_matrix(y_true, y_pred, class_names, pred_names):
    """生成混淆矩阵并添加求和列"""
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(data=cm, index=class_names, columns=pred_names)
    df['Sum'] = df.sum(axis=1).values
    return df, cm


def print_model_evaluation(accuracy, mcc, confusion_df, detailed_report):
    """打印模型评估结果"""
    print("\n=== Model evaluation ===")
    print("\n=== Accuracy classification score ===")
    print(f"\nACC = {accuracy:.6f}")
    print("\n=== Matthews Correlation Coefficient ===")
    print(f"\nMCC = {mcc:.6f}")
    print("\n=== Confusion Matrix ===\n")
    print(confusion_df.to_string())
    print("\n=== Detailed Accuracy By Class ===\n")
    print(detailed_report)


def model_evaluation(num_classes, y_true, y_pred):
    """每个fold测试结束后计算Precision、Recall、ACC、MCC等统计指标"""
    class_names = [f'Class {i + 1}' for i in range(num_classes)]
    pred_names = [f'Pred C{i + 1}' for i in range(num_classes)]

    confusion_df, cm = generate_confusion_matrix(y_true, y_pred, class_names, pred_names)
    
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(cm)
    detailed_report = classification_report(y_true, y_pred, target_names=class_names, digits=6)

    print_model_evaluation(accuracy, mcc, confusion_df, detailed_report)


def bi_model_evaluation(y_true, y_pred):
    """二分类问题每个fold测试结束后计算Precision、Recall、ACC、MCC等统计指标"""
    class_names = ["Positive", "Negative"]
    pred_names = ["Pred Positive", "Pred Negative"]

    confusion_df, cm = generate_confusion_matrix(y_true, y_pred, class_names, pred_names)
    
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(cm)
    detailed_report = classification_report(y_true, y_pred, target_names=class_names, digits=6)

    print_model_evaluation(accuracy, mcc, confusion_df, detailed_report)


def get_timestamp(fmt='%Y/%m/%d %H:%M:%S'):
    '''Returns a string that contains the current date and time.

    Suggested formats:
        short_format=%y%m%d-%H%M
        long format=%Y%m%d-%H%M%S  (default)
    '''
    now = datetime.datetime.now()
    return now.strftime(fmt)