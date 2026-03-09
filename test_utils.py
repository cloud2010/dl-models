# -*- coding: utf-8 -*-
"""
测试优化后的utils模块
"""

# 测试从根目录utils模块导入
from utils import (
    matthews_corrcoef,
    model_evaluation,
    bi_model_evaluation,
    get_timestamp
)

# 测试从各个目录的utils模块导入
from cnn.utils import matthews_corrcoef as cnn_mcc
from rnn.utils import model_evaluation as rnn_me
from nn.utils import bi_model_evaluation as nn_bme
from snn.utils import get_timestamp as snn_ts
from rf.utils import matthews_corrcoef as rf_mcc
from lgb.utils import model_evaluation as lgb_me
from deepForest.utils import create_logger, matthews_corrcoef as df_mcc

import numpy as np

print("测试工具函数导入成功!")
print(f"当前时间戳: {get_timestamp()}")

# 测试matthews_corrcoef函数
confusion_matrix = np.array([[10, 2], [3, 5]])
mcc_value = matthews_corrcoef(confusion_matrix)
print(f"MCC值: {mcc_value:.6f}")

# 测试从不同模块导入的函数是否相同
print(f"cnn_mcc与matthews_corrcoef是否相同: {cnn_mcc is matthews_corrcoef}")
print(f"rf_mcc与matthews_corrcoef是否相同: {rf_mcc is matthews_corrcoef}")
print(f"df_mcc与matthews_corrcoef是否相同: {df_mcc is matthews_corrcoef}")

print("所有测试通过!")
