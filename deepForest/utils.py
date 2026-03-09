import logging
import uuid

# 从根目录的utils模块导入工具函数
from utils import (
    matthews_corrcoef,
    model_evaluation,
    bi_model_evaluation,
    get_timestamp
)

__author__ = "Min"


def create_logger(instance, verbose):
    logger = logging.getLogger(str(uuid.uuid4()))
    fmt = logging.Formatter('{} - %(message)s'.format(instance))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger

