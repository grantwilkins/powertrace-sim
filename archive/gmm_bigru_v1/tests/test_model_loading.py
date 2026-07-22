import tempfile
import unittest
from pathlib import Path

import torch

from model.classifiers.gru import GRUClassifier
from model.classifiers.model_loading import load_gru_classifier


class TestModelLoading(unittest.TestCase):
    def test_load_from_state_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            source = GRUClassifier(Dx=2, K=3, H=8, num_layers=1)
            torch.save(source.state_dict(), checkpoint)

            loaded = load_gru_classifier(
                checkpoint_path=str(checkpoint),
                k=3,
                input_dim=2,
                hidden_dim=8,
                num_layers=1,
                device=torch.device("cpu"),
            )

            self.assertFalse(loaded.training)
            out = loaded(torch.randn(1, 10, 2))
            self.assertEqual(tuple(out.shape), (1, 10, 3))

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "missing.pt"
            with self.assertRaises((FileNotFoundError, OSError)):
                load_gru_classifier(
                    checkpoint_path=str(checkpoint),
                    k=3,
                    input_dim=2,
                    hidden_dim=8,
                    num_layers=1,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
