# -*- coding: utf-8 -*-
"""This program implements extra-trees classifier for predicting biological sequence datasets.
Date: 2024-05-31
Author: Min

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('ert', smote=True, roc=False)
