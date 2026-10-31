# -*- coding: utf-8 -*-
"""Bidirectional recurrent neural network for tabular sequence features.

Author: liumin@shmtu.edu.cn
Date: 2019-05-30
The historical entry retains its signature; sequence lengths now preserve rows.
"""

import numpy as np
from torch import nn
from nn.training import train_folds, init_model as init_linear
from experiments.data import load_table
from experiments.metrics import model_evaluation

class BiRNN(nn.Module):
    """Bidirectional recurrent neural network (many-to-one)
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        h0 = x.new_zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = x.new_zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def init_model(module):
    init_linear(module)
    if isinstance(module, nn.LSTM):
        for name, parameter in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)


def run(input_file, h_units, s_length, n_layers, l_rate, epochs, folds, random_s):
    X, y, _ = load_table(input_file)
    if s_length < 1 or X.shape[1] % s_length:
        raise ValueError("Sequence length must be positive and divide the feature count.")
    X = X.reshape(len(y), s_length, X.shape[1] // s_length)
    predicted = train_folds(lambda size, classes: BiRNN(size, h_units, n_layers, classes),
                            X, y, epochs=epochs, folds=folds, learning_rate=l_rate,
                            weight_decay=0, seed=random_s, initializer=init_model)
    model_evaluation(len(np.unique(y)), y, predicted)
    return predicted
