import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.pipeline.data_loading import load_config_data
from model.utils.io import write_json as _write_json


class TestPipelineDataLoading(unittest.TestCase):
    def _write_dataset(self, path: Path) -> None:
        power = np.asarray(
            [
                np.asarray([100.0, 102.0, 103.0, 104.0], dtype=np.float64),
                np.asarray([98.0, 99.0, 101.0, 102.0], dtype=np.float64),
                np.asarray([96.0, 97.0, 100.0, 101.0], dtype=np.float64),
            ],
            dtype=object,
        )
        active = np.asarray(
            [
                np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
                np.asarray([0.0, 1.0, 2.0, 1.0], dtype=np.float64),
                np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            ],
            dtype=object,
        )
        np.savez(
            path,
            pair_key=np.asarray(["p0", "p1", "p2"], dtype=object),
            power=power,
            active_requests=active,
        )

    def test_load_config_data_valid_npz_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.npz"
            split = root / "split.json"
            norm = root / "norm.json"
            self._write_dataset(dataset)
            _write_json(split, {"train_indices": [0], "val_indices": [1], "test_indices": [2]})
            _write_json(norm, {"active_mean": 0.0, "active_std": 1.0})

            payload, err = load_config_data(
                "toy_H100_tp1",
                {
                    "dataset_npz": str(dataset),
                    "split_json": str(split),
                    "norm_params_json": str(norm),
                },
                manifest_dir=str(root),
                feature_set="f2",
            )

            self.assertIsNone(err)
            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(len(payload["raw"]["train"]), 1)
            self.assertEqual(len(payload["raw"]["val"]), 1)
            self.assertEqual(len(payload["raw"]["test"]), 1)

    def test_load_config_data_missing_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "split.json"
            norm = root / "norm.json"
            _write_json(split, {"train_indices": [0], "val_indices": [1], "test_indices": [2]})
            _write_json(norm, {"active_mean": 0.0, "active_std": 1.0})

            payload, err = load_config_data(
                "toy_H100_tp1",
                {
                    "dataset_npz": str(root / "missing.npz"),
                    "split_json": str(split),
                    "norm_params_json": str(norm),
                },
                manifest_dir=str(root),
                feature_set="f2",
            )

            self.assertIsNone(payload)
            self.assertEqual(err, "missing_dataset_npz")

    def test_load_config_data_corrupt_split_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.npz"
            split = root / "split.json"
            norm = root / "norm.json"
            self._write_dataset(dataset)
            split.write_text("{ not valid json")
            _write_json(norm, {"active_mean": 0.0, "active_std": 1.0})

            payload, err = load_config_data(
                "toy_H100_tp1",
                {
                    "dataset_npz": str(dataset),
                    "split_json": str(split),
                    "norm_params_json": str(norm),
                },
                manifest_dir=str(root),
                feature_set="f2",
            )

            self.assertIsNone(payload)
            self.assertIsNotNone(err)
            assert err is not None
            self.assertTrue(err.startswith("split_json_error:"))

    def test_load_config_data_empty_split_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.npz"
            split = root / "split.json"
            norm = root / "norm.json"
            self._write_dataset(dataset)
            _write_json(split, {"train_indices": [], "val_indices": [1], "test_indices": [2]})
            _write_json(norm, {"active_mean": 0.0, "active_std": 1.0})

            payload, err = load_config_data(
                "toy_H100_tp1",
                {
                    "dataset_npz": str(dataset),
                    "split_json": str(split),
                    "norm_params_json": str(norm),
                },
                manifest_dir=str(root),
                feature_set="f2",
            )

            self.assertIsNone(payload)
            self.assertEqual(err, "empty_train_split")


if __name__ == "__main__":
    unittest.main()
