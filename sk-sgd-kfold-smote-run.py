# -*- coding: utf-8 -*-
"""This program implements Stochastic Gradient Descent classifier for predicting biological sequence datasets.
Date: 2024-03-20
Author: Min

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('sgd', smote=True, roc=False)
