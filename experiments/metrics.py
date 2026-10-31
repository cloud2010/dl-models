"""One implementation of the historical ACC/MCC and classification reports."""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels


def matthews_corrcoef(c_matrix):
    """Compute multiclass MCC from a confusion matrix; degenerate cases give 0."""
    matrix = np.asarray(c_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError("Expected a nonempty square confusion matrix.")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Confusion counts must be finite and nonnegative.")
    rows, cols, total = matrix.sum(axis=1), matrix.sum(axis=0), matrix.sum()
    denominator = np.sqrt(max(total**2 - rows @ rows, 0) * max(total**2 - cols @ cols, 0))
    return float((matrix.trace() * total - rows @ cols) / denominator) if denominator else 0.0


def generate_confusion_matrix(y_true, y_pred, class_names=None, pred_names=None):
    """Keep the historical XGBoost helper API, with explicit label ordering."""
    labels = unique_labels(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    names = class_names if class_names is not None else [f"Class {v}" for v in labels]
    columns = pred_names if pred_names is not None else [f"Pred {v}" for v in labels]
    frame = pd.DataFrame(matrix, index=names, columns=columns)
    frame["Sum"] = matrix.sum(axis=1)
    return frame


def model_evaluation(num_classes, y_true, y_pred):
    """Print metrics without assuming labels are consecutive integers."""
    labels = unique_labels(y_true, y_pred)
    expected = int(num_classes) if np.isscalar(num_classes) else len(num_classes)
    if expected != len(labels):
        raise ValueError("num_classes does not match observed labels.")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    scores = {"accuracy": float(accuracy_score(y_true, y_pred)), "mcc": matthews_corrcoef(matrix)}
    print("\n=== Model evaluation ===")
    print(f"ACC = {scores['accuracy']:.6f}\nMCC = {scores['mcc']:.6f}")
    print("\n=== Confusion Matrix (rows: true, columns: predicted) ===")
    print(generate_confusion_matrix(y_true, y_pred).to_string())
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=[str(v) for v in labels], digits=6, zero_division=0))
    return scores


def bi_model_evaluation(y_true, y_pred):
    """Binary reporting uses actual labels rather than guessed positive/negative names."""
    return model_evaluation(len(unique_labels(y_true, y_pred)), y_true, y_pred)


def get_timestamp(fmt="%Y/%m/%d %H:%M:%S"):
    return datetime.now().strftime(fmt)
