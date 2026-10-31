"""Incremental feature comparison with selection fitted inside every training fold.

This replaces the shared workflow of the historical sci-fs scripts for new runs;
their distinct exploratory experiments remain available as research records.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline

from .classifiers import build_classifier, make_parser
from .data import classification_splits, load_table


def feature_curve(estimator, X, y, counts, *, method="f_classif", folds=5, seed=0, smote=False, neighbors=3):
    """Return OOF metrics for feature counts; this is exploration, not nested model selection."""
    from functools import partial
    functions = {"f_classif": f_classif, "chi2": chi2,
                 "mutual_info_classif": partial(mutual_info_classif, random_state=seed)}
    if method not in functions:
        raise ValueError(f"Unknown feature score: {method}")
    if method == "chi2" and (X < 0).any():
        raise ValueError("chi2 requires nonnegative features.")
    counts = list(counts)
    if not counts or any(count < 1 or count > X.shape[1] for count in counts):
        raise ValueError("Feature counts must be between 1 and the number of columns.")
    splits = classification_splits(X, y, folds, seed)
    results = []
    pipeline_class = Pipeline
    if smote:
        from imblearn.pipeline import Pipeline as pipeline_class
        from imblearn.over_sampling import SMOTE
        if neighbors < 1 or any(np.unique(y[train], return_counts=True)[1].min() <= neighbors for train, _ in splits):
            raise ValueError("SMOTE neighbors exceed a training-fold class size.")
    for count in counts:
        steps = [("select", SelectKBest(functions[method], k=count))]
        if smote:
            steps.append(("smote", SMOTE(k_neighbors=neighbors, random_state=seed)))
        steps.append(("model", estimator))
        predicted = cross_val_predict(pipeline_class(steps), X, y, cv=splits, n_jobs=1)
        results.append({"features": count, "accuracy": accuracy_score(y, predicted),
                        "mcc": matthews_corrcoef(y, predicted)})
    return pd.DataFrame(results)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=("rf", "ert", "gbt", "bayes", "lda", "qda", "ncentroid"), default="rf")
    parser.add_argument("--method", choices=("f_classif", "chi2", "mutual_info_classif"), default="f_classif")
    parser.add_argument("--counts", type=int, nargs="+", required=True)
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--randomseed", type=int, default=0)
    parser.add_argument("--smote", action="store_true")
    parser.add_argument("--kneighbors", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        X, y, _ = load_table(args.datapath)
        defaults = make_parser(args.model).parse_args(["--datapath", args.datapath, "--randomseed", str(args.randomseed)])
        result = feature_curve(build_classifier(args.model, defaults), X, y, args.counts,
                               method=args.method, folds=args.kfolds, seed=args.randomseed,
                               smote=args.smote, neighbors=args.kneighbors)
        result.to_csv(Path(args.output), index=False)
        print(result.to_string(index=False))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
