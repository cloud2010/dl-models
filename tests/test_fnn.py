import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional dependency")
class FNNTests(unittest.TestCase):
    def test_all_variants_keep_single_sample_dimension(self):
        from nn.fnn import FNNs
        from nn.fnn_bn import FNNs as BatchFNN
        from nn.fnn_ln import FNNs as LayerFNN
        for model_class in (FNNs, BatchFNN, LayerFNN):
            model = model_class(4, 6, 0.5, 1, 2).eval()
            x = torch.ones((1, 4))
            self.assertEqual(model(x).shape, (1, 2))
            torch.testing.assert_close(model(x), model(x))

    def test_snn_and_sequence_model_share_independent_fold_training(self):
        from snn.snn_t import SNNs
        from rnn.birnn_torch import BiRNN, init_model
        from nn.training import train_folds
        torch.set_num_threads(1)
        X = np.random.default_rng(5).normal(size=(12, 4))
        y = np.array([2, 9] * 6)
        with contextlib.redirect_stdout(io.StringIO()):
            snn = train_folds(lambda size, classes: SNNs(size, 5, 0.1, classes), X, y,
                              epochs=1, folds=3, learning_rate=0.01, weight_decay=0, seed=0)
            birnn = train_folds(lambda size, classes: BiRNN(size, 3, 1, classes), X.reshape(12, 2, 2), y,
                                epochs=1, folds=3, learning_rate=0.01, weight_decay=0, seed=0, initializer=init_model)
        self.assertEqual(snn.shape, y.shape)
        self.assertEqual(birnn.shape, y.shape)
        self.assertTrue(set(birnn).issubset({2, 9}))

    def test_folds_reset_model_optimizer_and_evaluation_mode(self):
        from nn.fnn_bn import FNNs
        from nn.training import train_folds
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(12, 4))
        y = np.array(["a", "b"] * 6)
        models, optimizers, modes = [], [], []
        def factory(size, classes):
            model = FNNs(size, 5, 0.5, 1, classes)
            model.register_forward_pre_hook(lambda module, inputs: modes.append(module.training))
            models.append(model)
            return model
        Adam = torch.optim.Adam
        def optimizer(*args, **kwargs):
            result = Adam(*args, **kwargs)
            optimizers.append(result)
            return result
        with patch("torch.optim.Adam", side_effect=optimizer), contextlib.redirect_stdout(io.StringIO()):
            predicted = train_folds(factory, X, y, epochs=1, folds=3, learning_rate=0.01, weight_decay=0, seed=0)
        self.assertEqual(len(models), 3)
        self.assertEqual(len({id(o) for o in optimizers}), 3)
        self.assertEqual(modes, [True, False] * 3)
        self.assertTrue(all(model.fc_bn.num_batches_tracked.item() == 1 for model in models))
        self.assertEqual(predicted.shape, y.shape)
        self.assertTrue(set(predicted).issubset({"a", "b"}))


if __name__ == "__main__":
    unittest.main()
