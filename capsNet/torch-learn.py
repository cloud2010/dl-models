import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

FILE = "peptide_caps_net.csv"  # 数据集名称
DATASETS = Path(__file__).parent.parent.joinpath("datasets", FILE)  # 构建路径
print(Path(__file__).parent.parent)
print(DATASETS)
