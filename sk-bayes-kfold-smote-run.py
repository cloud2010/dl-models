# -*- coding: utf-8 -*-
"""This program implements native_bayes classifier for predicting biological sequence datasets.
Date: 2024-01-31
Author: Min

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('bayes', smote=True, roc=False)
