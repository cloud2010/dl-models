# -*- coding: utf-8 -*-
"""This program implements Linear Discriminant Analysis classifier for predicting biological sequence datasets.
Date: 2024-04-11
Author: Min

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('lda', smote=True, roc=False)
