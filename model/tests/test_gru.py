import unittest

import torch

from model.classifiers.gru import GRUClassifier


class TestGRUClassifier(unittest.TestCase):
    def test_forward_shape(self):
        model = GRUClassifier(Dx=3, K=4, H=8, num_layers=1)
        x = torch.randn(1, 12, 3)
        out = model(x)
        self.assertEqual(tuple(out.shape), (1, 12, 4))

    def test_bidirectional_hidden(self):
        model = GRUClassifier(Dx=2, K=5, H=7, num_layers=1)
        self.assertTrue(model.gru.bidirectional)
        self.assertEqual(model.fc.in_features, 14)

    def test_num_layers_param(self):
        model = GRUClassifier(Dx=2, K=3, H=6, num_layers=2)
        self.assertEqual(model.gru.num_layers, 2)
        x = torch.randn(1, 6, 2)
        out = model(x)
        self.assertEqual(tuple(out.shape), (1, 6, 3))


if __name__ == "__main__":
    unittest.main()
