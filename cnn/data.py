"""Shared data readers for the historical CNN experiments (no TensorFlow import)."""

from pathlib import Path
import random

import numpy as np


def get_datasets(dataset_path):
    paths, labels = [], []
    for path in sorted(Path(dataset_path).iterdir()):
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                header = handle.readline().split()
            if len(header) < 2:
                raise ValueError(f"Missing class label in {path}")
            paths.append(str(path))
            labels.append(int(header[1]) - 1)
    return paths, labels


def get_peptide_datasets(dataset_path, rate):
    root = Path(dataset_path)
    positive = sorted(p for p in (root / "P").iterdir() if p.is_file())
    negative = sorted(p for p in (root / "N").iterdir() if p.is_file())
    selected = random.sample(negative, rate * len(positive))
    return [str(p) for p in positive + selected], [0] * len(positive) + [1] * len(selected)


def read_data(file_path, data_height, data_width):
    rows = []
    with Path(file_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith(">"):
                rows.append(line.split())
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != data_width:
        raise ValueError(f"Expected a matrix with {data_width} columns: {file_path}")
    matrix = matrix[:data_height]
    matrix = np.pad(matrix, ((0, data_height - len(matrix)), (0, 0)), mode="constant")
    return matrix.reshape(data_height, data_width, 1)
