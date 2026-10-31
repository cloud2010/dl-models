# -*- coding: utf-8 -*-
"""This program implements Nearest centroid classifier for predicting biological sequence datasets.
Date: 2024-01-25
Author: Min

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('ncentroid', smote=True, roc=False)
