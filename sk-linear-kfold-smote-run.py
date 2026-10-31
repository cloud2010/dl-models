# -*- coding: utf-8 -*-
"""This program implements two linear classifiers for predicting biological sequence datasets.
Date: 2023-11-06
Ref:https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
    https://scikit-learn.org/stable/modules/linear_model.html#lasso

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('linear', smote=True, roc=False)
