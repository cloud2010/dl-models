# -*- coding: utf-8 -*-
"""This program implements two ensemble classifiers for predicting biological sequence datasets.
Date: 2023-06-10

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('tree', smote=True, roc=False)
