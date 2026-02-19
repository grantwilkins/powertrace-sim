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

from model.scripts.continuous_v1_stateless_train import run_training_from_manifest, train_one_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


class TestContinuousV1StatelessTrain(unittest.TestCase):
    def setUp(self):
        import torch

        torch.manual_seed(0)
        self.torch = torch

    def test_train_one_config_writes_checkpoint_and_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "checkpoints" / "cfg_best.pt"
            curve_path = root / "training_curves" / "cfg_curve.csv"

            p_prev = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(-1, 1)
            A_norm = np.linspace(-0.5, 0.5, 12, dtype=np.float32).reshape(-1, 1)
            dA_norm = np.concatenate(
                [np.zeros((1, 1), dtype=np.float32), np.diff(A_norm, axis=0).astype(np.float32)],
                axis=0,
            )
            p_target = (0.85 * p_prev) + (0.15 * A_norm)
            trace = {
                "pair_key": "trace-0",
                "p_prev_norm": p_prev.astype(np.float32),
                "A_norm": A_norm.astype(np.float32),
                "delta_A_norm": dA_norm.astype(np.float32),
                "p_target_norm": p_target.astype(np.float32),
            }
            config_data = {
                "train": [trace],
                "val": [trace],
                "test": [trace],
            }
            config_norm = {
                "power_mean": 0.0,
                "power_std": 1.0,
                "active_mean": 0.0,
                "active_std": 1.0,
            }

            out = train_one_config(
                config_id="toy_H100_tp1",
                config_data=config_data,
                config_norm=config_norm,
                hidden_dim=8,
                n_epochs=3,
                lr=1e-3,
                patience=3,
                scheduler_patience=2,
                scheduler_factor=0.5,
                lambda_mu=0.1,
                seed=7,
                device="cpu",
                checkpoint_path=str(checkpoint_path),
                curve_path=str(curve_path),
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(curve_path.exists())
            self.assertGreaterEqual(len(out["history"]), 1)
            self.assertTrue(np.isfinite(out["best_val_loss"]))
            self.assertTrue(np.isfinite(out["best_val_total_loss"]))
            self.assertAlmostEqual(float(out["lambda_mu"]), 0.1, places=8)

            with open(curve_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 1)
            expected_cols = {
                "epoch",
                "train_total_loss",
                "train_nll",
                "train_mu_loss",
                "val_total_loss",
                "val_nll",
                "val_mu_loss",
                "mean_sigma",
                "mean_alpha",
                "lr",
            }
            self.assertTrue(expected_cols.issubset(set(rows[0].keys())))
            self.assertTrue(np.isfinite(float(rows[-1]["val_total_loss"])))

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

            p0 = np.linspace(100.0, 110.0, 11, dtype=np.float64)
            p1 = np.linspace(95.0, 105.0, 13, dtype=np.float64)

            a0 = np.array([0, 1, 2, 2, 1, 0, 1, 3, 2, 1, 0], dtype=np.float64)
            a1 = np.array([0, 0, 1, 2, 2, 2, 1, 1, 0, 1, 2, 1, 0], dtype=np.float64)

            x0 = np.stack(
                [
                    np.linspace(-1.0, 0.0, 10, dtype=np.float32),
                    np.linspace(0.1, 0.5, 10, dtype=np.float32),
                    np.linspace(-0.3, 0.3, 10, dtype=np.float32),
                ],
                axis=1,
            )
            y0 = (0.9 * x0[:, 0]) + 0.1 * x0[:, 1]
            x1 = np.stack(
                [
                    np.linspace(0.0, 1.0, 12, dtype=np.float32),
                    np.linspace(0.4, -0.1, 12, dtype=np.float32),
                    np.linspace(0.2, 0.0, 12, dtype=np.float32),
                ],
                axis=1,
            )
            y1 = (0.85 * x1[:, 0]) + 0.15 * x1[:, 1]

            dataset_path = datasets_dir / f"{slug_good}.npz"
            np.savez(
                dataset_path,
                config_id=np.array([config_good], dtype=object),
                dt=np.array([0.25], dtype=np.float64),
                pair_key=np.asarray(["p0", "p1"], dtype=object),
                power=np.asarray([p0, p1], dtype=object),
                active_requests=np.asarray([a0, a1], dtype=object),
                x_norm=np.asarray([x0.astype(np.float32), x1.astype(np.float32)], dtype=object),
                y_norm=np.asarray([y0.astype(np.float32), y1.astype(np.float32)], dtype=object),
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
                    "power_mean": 0.0,
                    "power_std": 1.0,
                    "active_mean": 0.0,
                    "active_std": 1.0,
                    "t_arrive_log_mean": 0.0,
                    "t_arrive_log_std": 1.0,
                    "power_min": 90.0,
                    "power_max": 120.0,
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

            out_dir = root / "results" / "continuous_v1_stateless"
            run_manifest = run_training_from_manifest(
                manifest_path=str(manifest_path),
                out_dir=str(out_dir),
                hidden_dim=8,
                epochs=2,
                lr=1e-3,
                patience=2,
                scheduler_patience=1,
                scheduler_factor=0.5,
                lambda_mu=0.1,
                seed=9,
                device="cpu",
            )

            self.assertEqual(run_manifest["summary"]["num_trained"], 1)
            self.assertGreaterEqual(run_manifest["summary"]["num_skipped"], 1)
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertTrue((out_dir / "run_summary.csv").exists())

            ckpt = out_dir / "checkpoints" / f"{slug_good}_stateless_best.pt"
            curve = out_dir / "training_curves" / f"{slug_good}_stateless.csv"
            norm_copy = out_dir / "norm_params" / f"{slug_good}.json"
            self.assertTrue(ckpt.exists())
            self.assertTrue(curve.exists())
            self.assertTrue(norm_copy.exists())

            with open(norm_copy, "r") as f:
                norm_payload = json.load(f)
            self.assertTrue(np.isfinite(float(norm_payload["delta_A_mean"])))
            self.assertTrue(np.isfinite(float(norm_payload["delta_A_std"])))
            self.assertGreater(float(norm_payload["delta_A_std"]), 0.0)

            with open(out_dir / "run_summary.csv", "r", newline="") as f:
                rows = list(csv.DictReader(f))
            by_config = {r["config_id"]: r for r in rows}
            self.assertEqual(by_config[config_good]["status"], "trained")
            self.assertEqual(by_config[config_missing]["status"], "skipped")
            self.assertTrue(np.isfinite(float(by_config[config_good]["best_val_total_loss"])))
            self.assertAlmostEqual(float(by_config[config_good]["lambda_mu"]), 0.1, places=8)
            self.assertAlmostEqual(float(run_manifest["defaults"]["lambda_mu"]), 0.1, places=8)


if __name__ == "__main__":
    unittest.main()
