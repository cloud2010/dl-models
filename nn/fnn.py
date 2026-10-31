# -*- coding: utf-8 -*-
"""Pytorch Implementation of FNN with ReLU Activation.

Author: Liu Min
Date: 2019-4-26
Github: `https://github.com/cloud2010/dl-models`
"""

from .training import FeedForward, init_model, run_model


class FNNs(FeedForward):
    def __init__(self, input_size, hidden_size, dropout_rate, n_layers, num_classes):
        super().__init__(input_size, hidden_size, dropout_rate, n_layers, num_classes, normalization=None)


def run(inputFile, h_units, num_layers, epochs, folds, l_rate, d_rate, w_rate, random_s):
    return run_model(FNNs, inputFile, h_units, num_layers, epochs, folds, l_rate, d_rate, w_rate, random_s)
