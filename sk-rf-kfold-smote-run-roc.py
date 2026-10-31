# -*- coding: utf-8 -*-
"""A random forest classifier based on scikit-learn with SMOTE apply it to classify bio datasets (ROC).
Date: 2021-12-31

Legacy entry point; implementation shared in experiments.classifiers.
"""

from experiments.classifiers import main


if __name__ == "__main__":
    main('rf', smote=True, roc=True)
