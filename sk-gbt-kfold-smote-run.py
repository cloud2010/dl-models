# -*- coding: utf-8 -*-
"""This program implements Gradient Boosting Classifier for predicting biological sequence datasets.
Date: 2024-05-11
Author: Min

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('gbt', smote=True, roc=False)
