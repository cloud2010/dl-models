# -*- coding: utf-8 -*-
"""Pytorch Implementation of the Scaled ELU function and Alpha Dropout.

Author: Liu Min
Date: 2019-1-27
Github: `https://github.com/cloud2010/dl-models`
"""

import numpy as np
from torch import nn
from nn.training import init_model, train_folds
from experiments.data import load_table
from experiments.metrics import model_evaluation

class SNNs(nn.Module):
    """Fully connected neural network with 3 hidden layers and SELU activation

    Args:
        input_size: size of each input sample
        hidden_size: size of hidden layer units
        num_classes: size of each output sample
        dropout_rate: Hidden layer dropout rate
    """

    def __init__(self, input_size, hidden_size, dropout_rate, num_classes):
        super(SNNs, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Alpha Dropout is a type of Dropout that maintains the self-normalizing property.
        # Ref: `https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.AlphaDropout`
        self.dp1 = nn.AlphaDropout(dropout_rate)
        self.selu1 = nn.SELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dp2 = nn.AlphaDropout(dropout_rate)
        self.selu2 = nn.SELU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dp3 = nn.AlphaDropout(dropout_rate)
        self.selu3 = nn.SELU()
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dp1(out)
        out = self.selu1(out)
        out = self.fc2(out)
        out = self.dp2(out)
        out = self.selu2(out)
        out = self.fc3(out)
        out = self.dp3(out)
        out = self.selu3(out)
        out = self.output(out)
        return out


def run(inputFile, h_units, epochs, folds, l_rate, d_rate, w_rate, random_s):
    X, y, _ = load_table(inputFile)
    predicted = train_folds(lambda size, classes: SNNs(size, h_units, d_rate, classes),
                            X, y, epochs=epochs, folds=folds, learning_rate=l_rate,
                            weight_decay=w_rate, seed=random_s)
    model_evaluation(len(np.unique(y)), y, predicted)
    return predicted
