# -*- coding: utf-8 -*-
"""This program implements Gaussian process classification (GPC) for predicting biological sequence datasets.
Date: 2024-05-11
Author: Min

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('gpc', smote=True, roc=False)
