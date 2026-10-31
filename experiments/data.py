"""Validation shared by tabular experiments."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def load_table(path):
    """Read headered CSV: unmodified target first, finite numeric features after it."""
    frame = pd.read_csv(path)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("CSV needs samples, a target column and at least one feature.")
    if frame.iloc[:, 0].isna().any():
        raise ValueError("Target labels must not be missing.")
    try:
        features = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Feature columns must contain numeric values.") from exc
    if not np.isfinite(features).all():
        raise ValueError("Features contain missing or infinite values.")
    return features, frame.iloc[:, 0].to_numpy(), frame.columns[1:].tolist()


def classification_splits(X, y, folds, seed):
    """Reject folds with missing classes instead of emitting misleading metrics."""
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        raise ValueError("Classification requires at least two classes.")
    if folds < 2 or folds > counts.min():
        raise ValueError("kfolds must be >= 2 and <= the smallest class size.")
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
