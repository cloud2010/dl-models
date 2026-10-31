# -*- coding: utf-8 -*-
"""A SVM classifier based on scikit-learn with SMOTE apply it to classify bio datasets.
Date: 2020-04-01

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('svm', smote=True, roc=False)
