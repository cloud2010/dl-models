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
