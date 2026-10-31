"""Shared FNN architecture and independent-fold PyTorch training."""

import numpy as np
import torch
from torch import nn

from experiments.data import classification_splits, load_table
from experiments.metrics import model_evaluation


class FeedForward(nn.Sequential):
    """Keep the original layer names/topology, with optional input normalization."""

    def __init__(self, input_size, hidden_size, dropout_rate, n_layers, num_classes, normalization=None):
        if input_size < 1 or hidden_size < 1 or n_layers < 0 or num_classes < 2:
            raise ValueError("Invalid FNN dimensions.")
        super().__init__()
        self.add_module("fc", nn.Linear(input_size, hidden_size))
        if normalization == "batch":
            self.add_module("fc_bn", nn.BatchNorm1d(hidden_size))
        elif normalization == "layer":
            self.add_module("fc_ln", nn.LayerNorm(hidden_size))
        self.add_module("fc_relu", nn.ReLU())
        self.add_module("fc_dp", nn.Dropout(dropout_rate))
        for i in range(1, n_layers + 1):
            self.add_module(f"hidden{i}", nn.Linear(hidden_size, hidden_size))
            self.add_module(f"relu{i}", nn.ReLU())
            self.add_module(f"dp{i}", nn.Dropout(dropout_rate))
        self.add_module("output", nn.Linear(hidden_size, num_classes))
        # Sequential preserves [batch, classes], including a single test sample.


def init_model(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def train_folds(model_factory, X, y, *, epochs, folds, learning_rate, weight_decay, seed, initializer=init_model):
    """Every fold owns a new model, normalization state and optimizer."""
    if epochs < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("epochs/l_rate must be positive; weight decay must be nonnegative.")
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder().fit(y)
    encoded = encoder.transform(y)
    splits = classification_splits(X, encoded, folds, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.as_tensor(X, dtype=torch.float32, device=device)
    targets = torch.as_tensor(encoded, dtype=torch.long, device=device)
    predictions = np.empty(len(y), dtype=np.int64)
    for fold, (train, test) in enumerate(splits, 1):
        torch.manual_seed(seed + fold - 1)
        model = model_factory(X.shape[-1], len(encoder.classes_)).to(device)
        model.apply(initializer)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features[train]), targets[train])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions[test] = model(features[test]).argmax(dim=1).cpu().numpy()
        print(f"Fold {fold}/{folds}: train={len(train)}, test={len(test)}, loss={loss.item():.6f}")
    return encoder.inverse_transform(predictions)


def run_model(model_class, input_file, h_units, num_layers, epochs, folds, l_rate, d_rate, w_rate, random_s):
    X, y, _ = load_table(input_file)
    predicted = train_folds(lambda size, classes: model_class(size, h_units, d_rate, num_layers, classes),
                            X, y, epochs=epochs, folds=folds, learning_rate=l_rate,
                            weight_decay=w_rate, seed=random_s)
    model_evaluation(len(np.unique(y)), y, predicted)
    return predicted
