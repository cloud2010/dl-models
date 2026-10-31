"""Compatibility exports for the shared evaluation implementation."""

from pathlib import Path
import sys

# Support historical direct execution from a model subdirectory.
_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from experiments.metrics import (
    matthews_corrcoef, model_evaluation, bi_model_evaluation, get_timestamp,
)

import pandas as pd
from sklearn.metrics import confusion_matrix


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
