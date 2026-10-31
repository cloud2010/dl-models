# -*- coding: utf-8 -*-
"""A random forest classifier based on scikit-learn apply it to classify bio datasets.
Date: 2019-04-11

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('rf', smote=False, roc=False)
