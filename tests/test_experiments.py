import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef as sklearn_mcc

from experiments.classifiers import cross_validate, main, make_parser, build_classifier
from experiments.data import classification_splits, load_table
from experiments.metrics import matthews_corrcoef, model_evaluation


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(n_samples=60, n_features=5, n_informative=3,
                                            weights=[0.7, 0.3], random_state=5)

    def test_smote_only_sees_training_rows(self):
        from imblearn.over_sampling import SMOTE
        # An identity feature allows us to inspect exactly which rows reach sampling.
        X = np.column_stack([self.X, np.arange(len(self.y))])
        seen = []
        original = SMOTE.fit_resample
        def record(sampler, train_X, train_y, **kwargs):
            seen.append(set(train_X[:, -1].astype(int)))
            return original(sampler, train_X, train_y, **kwargs)
        with patch.object(SMOTE, "fit_resample", record):
            predicted, _ = cross_validate(RandomForestClassifier(n_estimators=5, random_state=0),
                                            X, self.y, folds=3, seed=0, smote=True, neighbors=2)
        splits = classification_splits(X, self.y, 3, 0)
        self.assertEqual(len(seen), 3)
        for rows, (train, test) in zip(seen, splits):
            self.assertEqual(rows, set(train))
            self.assertFalse(rows.intersection(test))
        self.assertEqual(predicted.shape, self.y.shape)

    def test_oof_probabilities_match_predictions_and_original_labels(self):
        labels = np.where(self.y, "peptide", "control")
        predicted, proba = cross_validate(RandomForestClassifier(n_estimators=7, random_state=2),
                                          self.X, labels, folds=3, probabilities=True, scale=True)
        np.testing.assert_array_equal(predicted, np.unique(labels)[proba.argmax(axis=1)])
        np.testing.assert_allclose(proba.sum(axis=1), 1)

    def test_small_classes_and_invalid_smote_fail_early(self):
        with self.assertRaisesRegex(ValueError, "smallest class"):
            cross_validate(RandomForestClassifier(), self.X, self.y, folds=20)
        with self.assertRaisesRegex(ValueError, "SMOTE kneighbors"):
            cross_validate(RandomForestClassifier(), self.X, self.y, folds=3, smote=True, neighbors=30)

    def test_metrics_match_sklearn_and_handle_degenerate_cases(self):
        actual = [1, 3, 3, 9, 9, 9]
        predicted = [3, 3, 3, 9, 1, 9]
        self.assertAlmostEqual(matthews_corrcoef(confusion_matrix(actual, predicted)), sklearn_mcc(actual, predicted))
        self.assertEqual(matthews_corrcoef([[4, 0], [0, 0]]), 0)
        self.assertEqual(matthews_corrcoef(np.zeros((2, 2))), 0)
        with self.assertRaises(ValueError):
            matthews_corrcoef([[1, -1], [2, 3]])
        with contextlib.redirect_stdout(io.StringIO()):
            scores = model_evaluation([1, 3, 9], actual, predicted)
        self.assertAlmostEqual(scores["mcc"], sklearn_mcc(actual, predicted))

    def test_legacy_metric_modules_share_one_function(self):
        from utils import matthews_corrcoef as root_mcc
        from capsNet.utils import matthews_corrcoef as capsule_mcc
        from xgb.utils import matthews_corrcoef as xgb_mcc, generate_confusion_matrix
        self.assertIs(root_mcc, matthews_corrcoef)
        self.assertIs(capsule_mcc, matthews_corrcoef)
        self.assertIs(xgb_mcc, matthews_corrcoef)
        _, matrix = generate_confusion_matrix([0, 1], [1, 1], ["a", "b"], ["a", "b"])
        self.assertEqual(matrix.shape, (2, 2))

    def test_cli_keeps_target_values_and_output_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            labels = np.where(self.y, 8, 3)
            data = directory / "input.csv"
            output = directory / "predictions.csv"
            metrics = directory / "metrics.json"
            pd.DataFrame(np.column_stack([labels, self.X])).to_csv(data, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                main("rf", argv=["--datapath", str(data), "-k", "3", "-t", "5",
                                  "--output", str(output), "--json", str(metrics)])
            np.testing.assert_array_equal(pd.read_csv(output)["target"], labels)
            self.assertTrue(set(pd.read_csv(output)["prediction"]).issubset({3, 8}))
            self.assertTrue(metrics.exists())
            pd.DataFrame({"target": [0, 1], "feature": [1, float("inf")]}).to_csv(data, index=False)
            with self.assertRaisesRegex(ValueError, "infinite"):
                load_table(data)

    def test_lasso_is_not_silently_used_as_classifier(self):
        args = make_parser("linear", True).parse_args(["--datapath", "unused", "-c", "lasso"])
        with self.assertRaisesRegex(ValueError, "regressor"):
            build_classifier("linear", args)

    def test_feature_selection_is_fitted_in_training_folds(self):
        from experiments.feature_selection import feature_curve
        from sklearn.feature_selection import SelectKBest
        seen = []
        original = SelectKBest.fit
        def record(selector, X, y):
            seen.append(len(X))
            return original(selector, X, y)
        with patch.object(SelectKBest, "fit", record):
            result = feature_curve(RandomForestClassifier(n_estimators=3, random_state=0),
                                   self.X, self.y, [1, 3], folds=3)
        self.assertEqual(seen, [40] * 6)
        self.assertEqual(result["features"].tolist(), [1, 3])

    def test_regression_does_not_shift_targets(self):
        from experiments.regression import main as regress
        with tempfile.TemporaryDirectory() as directory:
            input_path, output_path = Path(directory) / "data.csv", Path(directory) / "pred.csv"
            pd.DataFrame({"target": [7.0] * 12, "feature": np.arange(12)}).to_csv(input_path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                regress(["--datapath", str(input_path), "--output", str(output_path), "-k", "3"])
            result = pd.read_csv(output_path)
            self.assertEqual(list(result.columns), ["The predicted values"])
            np.testing.assert_allclose(result.iloc[:, 0], 7.0)


if __name__ == "__main__":
    unittest.main()
