# -*- coding: utf-8 -*-
"""A knn classifier with SMOTE based on scikit-learn apply it to classify bio datasets.
Date: 2019-10-17

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('knn', smote=True, roc=False)
