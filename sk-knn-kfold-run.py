# -*- coding: utf-8 -*-
"""A knn classifier based on scikit-learn apply it to classify bio datasets.
Date: 2019-04-23

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('knn', smote=False, roc=False)
