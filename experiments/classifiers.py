"""Shared CLI and leakage-free cross-validation for the scikit-learn experiments."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import json
import time


def make_parser(model, smote=False):
    parser = ArgumentParser(description=f"{model}: stratified out-of-fold classification",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", required=True, help="Headered CSV: target, then numeric features.")
    parser.add_argument("-k", "--kfolds", type=int, default=10)
    parser.add_argument("-r", "--randomseed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel CV folds (avoid nested parallelism).")
    parser.add_argument("--scale", action="store_true", help="Fit StandardScaler inside each training fold.")
    parser.add_argument("--output", help="Write OOF predictions in original input row order.")
    parser.add_argument("--json", dest="metrics_path", help="Write metrics and experiment settings as JSON.")
    if smote:
        # -n remains KNN's neighbor count in its historical CLI.
        flags = ("--kneighbors",) if model == "knn" else ("-n", "--kneighbors")
        parser.add_argument(*flags, type=int, default=3, help="SMOTE neighbors; must fit every training fold.")
    if model == "rf":
        parser.add_argument("-t", "--ntrees", type=int, default=100)
        parser.add_argument("-d", "--mdepth", type=int, default=None)
        parser.add_argument("-e", "--epochs", type=int, default=100,
                            help="Legacy accepted argument; random forests do not use epochs.")
    if model == "knn":
        parser.add_argument("-n", "--nneighbors", type=int, default=5)
        parser.add_argument("-w", "--weights", choices=("uniform", "distance"), default="uniform")
    if model == "svm":
        parser.add_argument("--kernel", choices=("linear", "poly", "rbf", "sigmoid"), default="sigmoid")
    if model == "tree":
        parser.add_argument("-c", "--classifier", choices=("ext", "ada"), default="ext")
    if model == "linear":
        parser.add_argument("-c", "--classifier", choices=("ridge", "lasso"), default="ridge")
    return parser


def build_classifier(model, args):
    """Import only the selected estimator; no optional model libraries required."""
    seed = args.randomseed
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=args.ntrees, max_depth=args.mdepth, random_state=seed, n_jobs=1)
    if model in ("ert", "tree"):
        from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
        if model == "tree" and args.classifier == "ada":
            return AdaBoostClassifier(random_state=seed)
        return ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    if model == "gbt":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=seed)
    if model == "gpc":
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF
        return GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=seed)
    if model == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=args.nneighbors, weights=args.weights, n_jobs=1)
    if model == "ncentroid":
        from sklearn.neighbors import NearestCentroid
        return NearestCentroid()
    if model == "bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    if model in ("lda", "qda"):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        return LinearDiscriminantAnalysis() if model == "lda" else QuadraticDiscriminantAnalysis()
    if model == "linear":
        if args.classifier == "lasso":
            raise ValueError("Lasso is a regressor, not a classifier. Use --classifier ridge for this classification entry.")
        from sklearn.linear_model import RidgeClassifier
        return RidgeClassifier()
    if model == "sgd":
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss="hinge", penalty="l2", random_state=seed, n_jobs=1)
    if model == "svm":
        from sklearn.svm import SVC
        return SVC(kernel=args.kernel, random_state=seed)
    raise ValueError(f"Unknown classifier: {model}")


def cross_validate(estimator, X, y, *, folds=10, seed=0, smote=False, neighbors=3,
                   scale=False, jobs=1, probabilities=False):
    """Only training rows reach the scaler/sampler; validation keeps original prevalence."""
    import numpy as np
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from .data import classification_splits

    splits = classification_splits(X, y, folds, seed)
    steps = [("scale", StandardScaler())] if scale else []
    if smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        if neighbors < 1 or any(np.unique(y[train], return_counts=True)[1].min() <= neighbors
                                for train, _ in splits):
            raise ValueError("SMOTE kneighbors must be >= 1 and smaller than every training-fold class size.")
        steps.append(("smote", SMOTE(k_neighbors=neighbors, random_state=seed)))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    if probabilities:
        # Fit once per fold; hard predictions come from the same OOF probabilities.
        proba = cross_val_predict(pipeline, X, y, cv=splits, n_jobs=jobs, method="predict_proba")
        return np.unique(y)[proba.argmax(axis=1)], proba
    return cross_val_predict(pipeline, X, y, cv=splits, n_jobs=jobs), None


def main(model, smote=False, roc=False, argv=None):
    parser = make_parser(model, smote)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    from .data import load_table
    from .metrics import model_evaluation
    import numpy as np
    import pandas as pd
    try:
        X, y, _ = load_table(args.datapath)
        estimator = build_classifier(model, args)
        print(f"Dataset shape: {X.shape}; classifier: {estimator}")
        predicted, proba = cross_validate(estimator, X, y, folds=args.kfolds,
                                          seed=args.randomseed, smote=smote,
                                          neighbors=getattr(args, "kneighbors", 3),
                                          scale=args.scale, jobs=args.jobs, probabilities=roc)
        scores = model_evaluation(len(np.unique(y)), y, predicted)
        if roc:
            from sklearn.metrics import roc_auc_score
            classes = np.unique(y)
            scores["auc"] = float(roc_auc_score(y, proba[:, 1]) if len(classes) == 2 else
                                   roc_auc_score(y, proba, labels=classes, multi_class="ovr"))
            print(f"AUC = {scores['auc']:.6f}")
        if args.output:
            pd.DataFrame({"target": y, "prediction": predicted}).to_csv(args.output, index=False)
        if args.metrics_path:
            Path(args.metrics_path).write_text(json.dumps({"model": model, "smote": smote,
                "settings": vars(args), "samples": len(y), "metrics": scores}, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Finished in {time.perf_counter() - started:.3f} seconds")
