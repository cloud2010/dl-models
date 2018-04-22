import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv
import math
# 计算 ACC 混淆矩阵 输出 recall f1等指标
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

__author__ = "Min"

def matthews_corrcoef(c_matrix):
    """多分类问题MCC计算
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


def model_evaluation(num_classes, y_true, y_pred):
    """每个fold测试结束后计算Precision、Recall、ACC、MCC等统计指标

    Args:
    num_classes : 分类数
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    """
    class_names = []
    pred_names = []
    for i in range(num_classes):
        class_names.append('Class ' + str(i + 1))
        pred_names.append('Pred C' + str(i + 1))

    # 混淆矩阵生成
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(data=cm, index=class_names, columns=pred_names)

    # 混淆矩阵添加一列代表各类求和
    df['Sum'] = df.sum(axis=1).values

    print("\n=== Model evaluation ===")
    print("\n=== Accuracy classification score ===")
    print("\nACC = {:.6f}".format(accuracy_score(y_true, y_pred)))
    print("\n=== Matthews Correlation Coefficient ===")
    print("\nMCC = {:.6f}".format(matthews_corrcoef(cm)))
    print("\n=== Confusion Matrix ===\n")
    print(df.to_string())
    print("\n=== Detailed Accuracy By Class ===\n")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, digits=6))


def bi_model_evaluation(y_true, y_pred):
    """二分类问题每个fold测试结束后计算Precision、Recall、ACC、MCC等统计指标

    Args:
    num_classes : 分类数
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    """
    class_names = ["Positive", "Negative"]
    pred_names = ["Pred Positive", "Pred Negative"]

    # 混淆矩阵生成
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(data=cm, index=class_names, columns=pred_names)

    # 混淆矩阵添加一列代表各类求和
    df['Sum'] = df.sum(axis=1).values

    print("\n=== Model evaluation ===")
    print("\n=== Accuracy classification score ===")
    print("\nACC = {:.6f}".format(accuracy_score(y_true, y_pred)))
    print("\n=== Matthews Correlation Coefficient ===")
    print("\nMCC = {:.6f}".format(matthews_corrcoef(cm)))
    print("\n=== Confusion Matrix ===\n")
    print(df.to_string())
    print("\n=== Detailed Accuracy By Class ===\n")
    print(classification_report(y_true, y_pred,
                                target_names=class_names, digits=6))


def get_timestamp(fmt='%Y/%m/%d %H:%M:%S'):
    '''Returns a string that contains the current date and time.

    Suggested formats:
        short_format=%y%m%d-%H%M
        long format=%Y%m%d-%H%M%S  (default)
    '''
    import datetime
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, fmt)


def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))

    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    epoch_axis = 0
    for i, key in enumerate(keys):
        if key == 'epoch':
            epoch_axis = i
            values[:, epoch_axis] += 1
            break
    for i, key in enumerate(keys):
        if key.find('loss') >= 0:  # loss
            print(values[:, i])
            plt.plot(values[:, epoch_axis], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, epoch_axis], values[:, i], label=key)
    plt.legend()
    plt.grid()
    plt.title('Accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


if __name__ == "__main__":
    plot_log('result/log.csv')
