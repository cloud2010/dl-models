"""SVR cross-validation with unmodified targets and reproducible splits."""

from argparse import ArgumentParser


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("-k", "--kfolds", type=int, default=10)
    parser.add_argument("-r", "--randomseed", type=int, default=0)
    parser.add_argument("--kernel", choices=("linear", "poly", "rbf", "sigmoid"), default="sigmoid")
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args(argv)
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from .data import load_table
    try:
        X, y, _ = load_table(args.datapath)
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Regression targets must be finite.")
        if not 2 <= args.kfolds <= len(y):
            raise ValueError("kfolds must be between 2 and the number of samples.")
        estimator = SVR(kernel=args.kernel)
        if args.scale:
            estimator = make_pipeline(StandardScaler(), estimator)
        splits = KFold(args.kfolds, shuffle=True, random_state=args.randomseed)
        predicted = cross_val_predict(estimator, X, y, cv=splits, n_jobs=1)
        pd.DataFrame({"The predicted values": predicted}).to_csv(args.output, index=False)
        print(f"RMSE = {np.sqrt(mean_squared_error(y, predicted)):.6f}; "
              f"MAE = {mean_absolute_error(y, predicted):.6f}; R2 = {r2_score(y, predicted):.6f}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
