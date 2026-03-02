import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

from model.scripts.train_gmm_bigru import run_training_from_manifest, train_one_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


class TestContinuousV1GMMBiGRUTrain(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.torch = torch

    def test_train_one_config_writes_checkpoint_and_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "checkpoints" / "cfg_best.pt"
            curve_path = root / "training_curves" / "cfg_curve.csv"

            x = np.stack(
                [
                    np.linspace(-1.0, 1.0, 12, dtype=np.float32),
                    np.linspace(0.4, -0.2, 12, dtype=np.float32),
                ],
                axis=1,
            )
            y = np.mod(np.arange(12), 3).astype(np.int64)
            trace = {
                "pair_key": "trace-0",
                "features_norm": x.astype(np.float32),
                "state_labels": y.astype(np.int64),
                "target_power_w": np.linspace(100.0, 120.0, 12, dtype=np.float64),
            }
            config_data = {"train": [trace], "val": [trace], "test": [trace]}

            out = train_one_config(
                config_id="toy_H100_tp1",
                config_data=config_data,
                k=3,
                input_dim=2,
                hidden_dim=8,
                num_layers=1,
                n_epochs=3,
                lr=1e-3,
                patience=3,
                scheduler_patience=2,
                scheduler_factor=0.5,
                seed=7,
                device="cpu",
                checkpoint_path=str(checkpoint_path),
                curve_path=str(curve_path),
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(curve_path.exists())
            self.assertGreaterEqual(len(out["history"]), 1)
            self.assertTrue(np.isfinite(out["best_val_loss"]))

            with open(curve_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue({"epoch", "train_loss", "val_loss", "lr"}.issubset(set(rows[0].keys())))
            self.assertTrue(np.isfinite(float(rows[-1]["val_loss"])))

    def test_run_training_from_manifest_trains_and_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp_root = root / "results" / "experimental_continuous_v1"
            datasets_dir = exp_root / "datasets"
            splits_dir = exp_root / "splits"
            norms_dir = exp_root / "norm_params"
            datasets_dir.mkdir(parents=True, exist_ok=True)
            splits_dir.mkdir(parents=True, exist_ok=True)
            norms_dir.mkdir(parents=True, exist_ok=True)

            config_good = "toy-good_H100_tp1"
            slug_good = "toy-good_H100_tp1"

            power0 = np.array([100.0, 102.0, 105.0, 103.0, 106.0, 108.0], dtype=np.float64)
            power1 = np.array([98.0, 99.0, 101.0, 100.0, 103.0, 104.0, 102.0], dtype=np.float64)
            active0 = np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0], dtype=np.float64)
            active1 = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0], dtype=np.float64)
            tlog0 = np.log1p(np.array([0.0, 16.0, 0.0, 8.0, 0.0, 0.0], dtype=np.float64))
            tlog1 = np.log1p(np.array([0.0, 0.0, 12.0, 0.0, 6.0, 0.0, 0.0], dtype=np.float64))

            dataset_path = datasets_dir / f"{slug_good}.npz"
            np.savez(
                dataset_path,
                config_id=np.array([config_good], dtype=object),
                dt=np.array([0.25], dtype=np.float64),
                pair_key=np.asarray(["p0", "p1"], dtype=object),
                power=np.asarray([power0, power1], dtype=object),
                active_requests=np.asarray([active0, active1], dtype=object),
                t_arrive_log=np.asarray([tlog0, tlog1], dtype=object),
            )
            split_path = splits_dir / f"{slug_good}.json"
            _write_json(
                split_path,
                {
                    "config_id": config_good,
                    "train_indices": [0],
                    "val_indices": [1],
                    "test_indices": [1],
                },
            )
            norm_path = norms_dir / f"{slug_good}.json"
            _write_json(
                norm_path,
                {
                    "config_id": config_good,
                    "dt": 0.25,
                    "power_mean": 101.0,
                    "power_std": 3.0,
                    "active_mean": 1.0,
                    "active_std": 1.0,
                    "t_arrive_log_mean": 0.2,
                    "t_arrive_log_std": 0.5,
                    "power_min": 90.0,
                    "power_max": 130.0,
                },
            )

            config_missing = "toy-missing_H100_tp1"
            manifest_path = exp_root / "manifest.json"
            _write_json(
                manifest_path,
                {
                    "schema_version": "experimental-continuous-v1",
                    "configs": {
                        config_good: {
                            "written": True,
                            "dataset_npz": str(dataset_path),
                            "split_json": str(split_path),
                            "norm_params_json": str(norm_path),
                        },
                        config_missing: {
                            "written": True,
                            "dataset_npz": str(exp_root / "datasets" / "missing.npz"),
                            "split_json": str(split_path),
                            "norm_params_json": str(norm_path),
                        },
                        "toy-ignored_H100_tp1": {
                            "written": False,
                            "dataset_npz": str(dataset_path),
                            "split_json": str(split_path),
                            "norm_params_json": str(norm_path),
                        },
                    },
                },
            )

            out_root = root / "results" / "continuous_v1_gmm_bigru"
            run_manifest = run_training_from_manifest(
                manifest_path=str(manifest_path),
                out_root=str(out_root),
                k=3,
                feature_set="f2",
                hidden_dim=8,
                num_layers=1,
                epochs=2,
                lr=1e-3,
                patience=2,
                scheduler_patience=1,
                scheduler_factor=0.5,
                bic_candidates=[2, 3],
                seed=9,
                device="cpu",
            )

            self.assertEqual(run_manifest["summary"]["num_trained"], 1)
            self.assertGreaterEqual(run_manifest["summary"]["num_skipped"], 1)
            out_dir = Path(run_manifest["defaults"]["out_dir"])
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertTrue((out_dir / "run_summary.csv").exists())

            ckpt = out_dir / "checkpoints" / f"{slug_good}_k3_f2_best.pt"
            curve = out_dir / "training_curves" / f"{slug_good}_k3_f2.csv"
            norm_copy = out_dir / "norm_params" / f"{slug_good}.json"
            gmm_json = out_dir / "gmms" / f"{slug_good}_k3.json"
            self.assertTrue(ckpt.exists())
            self.assertTrue(curve.exists())
            self.assertTrue(norm_copy.exists())
            self.assertTrue(gmm_json.exists())

            with open(norm_copy, "r") as f:
                norm_payload = json.load(f)
            self.assertTrue(np.isfinite(float(norm_payload["delta_A_mean"])))
            self.assertTrue(np.isfinite(float(norm_payload["delta_A_std"])))
            self.assertGreater(float(norm_payload["delta_A_std"]), 0.0)
            self.assertEqual(int(norm_payload["k"]), 3)
            self.assertEqual(str(norm_payload["feature_set"]), "f2")

            with open(gmm_json, "r") as f:
                gmm_payload = json.load(f)
            self.assertEqual(int(gmm_payload["k"]), 3)
            self.assertEqual(len(gmm_payload["means"]), 3)
            self.assertEqual(len(gmm_payload["variances"]), 3)

            with open(out_dir / "run_summary.csv", "r", newline="") as f:
                rows = list(csv.DictReader(f))
            by_config = {r["config_id"]: r for r in rows}
            self.assertEqual(by_config[config_good]["status"], "trained")
            self.assertIn(by_config[config_missing]["status"], {"skipped", "failed"})
            self.assertTrue(np.isfinite(float(by_config[config_good]["best_val_loss"])))


if __name__ == "__main__":
    unittest.main()
