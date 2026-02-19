import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

from model.classifiers.gru import GRUClassifier
from model.scripts.continuous_v1_gmm_bigru_eval import evaluate_from_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_pair_manifest(path: Path, *, pair_key: str, json_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "dataset_dir",
                "dataset_name",
                "status",
                "model_name",
                "hardware",
                "tensor_parallelism",
                "rate",
                "iteration",
                "date_key",
                "pair_key",
                "power_csv_path",
                "json_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "family": "toy",
                "dataset_dir": "toy",
                "dataset_name": "toy",
                "status": "matched",
                "model_name": "toy",
                "hardware": "H100",
                "tensor_parallelism": "1",
                "rate": "1",
                "iteration": "0",
                "date_key": "20260101-000000",
                "pair_key": pair_key,
                "power_csv_path": "",
                "json_path": json_path,
            }
        )


class TestContinuousV1GMMBiGRUEval(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def _build_fixture(self, root: Path, *, include_throughput: bool = True):
        cfg = "toy_H100_tp1"
        pair_key = "tp=1|rate=1|date=20260101-000000"

        checkpoint_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "checkpoints" / "toy_H100_tp1_k3_f2_best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model = GRUClassifier(Dx=2, K=3, H=8, num_layers=1)
        torch.save(model.state_dict(), checkpoint_path)

        norm_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "norm_params" / "toy_H100_tp1.json"
        _write_json(
            norm_path,
            {
                "config_id": cfg,
                "dt": 0.25,
                "k": 3,
                "feature_set": "f2",
                "input_dim": 2,
                "hidden_dim": 8,
                "num_layers": 1,
                "active_mean": 0.0,
                "active_std": 1.0,
                "t_arrive_log_mean": 0.0,
                "t_arrive_log_std": 1.0,
                "delta_A_mean": 0.0,
                "delta_A_std": 1.0,
                "power_mean": 100.0,
                "power_std": 10.0,
                "power_min": 70.0,
                "power_max": 140.0,
            },
        )

        gmm_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "gmms" / "toy_H100_tp1_k3.json"
        _write_json(
            gmm_path,
            {
                "config_id": cfg,
                "k": 3,
                "covariance_type": "full",
                "means": [90.0, 110.0, 130.0],
                "variances": [4.0, 9.0, 16.0],
                "weights": [0.3, 0.4, 0.3],
                "order": [0, 1, 2],
                "label_map": [0, 1, 2],
                "aic": 0.0,
                "bic": 0.0,
            },
        )

        run_manifest_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "run_manifest.json"
        _write_json(
            run_manifest_path,
            {
                "schema_version": "continuous-v1-gmm-bigru-train-run-v1",
                "configs": {
                    cfg: {
                        "status": "trained",
                        "checkpoint_path": str(checkpoint_path),
                        "norm_params_path": str(norm_path),
                        "gmm_params_path": str(gmm_path),
                        "k": 3,
                        "feature_set": "f2",
                        "input_dim": 2,
                        "hidden_dim": 8,
                        "num_layers": 1,
                    }
                },
            },
        )

        power = np.array([100.0, 103.0, 107.0, 106.0, 102.0, 101.0], dtype=np.float64)
        dataset_path = root / "results" / "experimental_continuous_v1" / "datasets" / "toy_H100_tp1.npz"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dataset_path,
            config_id=np.asarray([cfg], dtype=object),
            dt=np.asarray([0.25], dtype=np.float64),
            pair_key=np.asarray([pair_key], dtype=object),
            rate=np.asarray(["1"], dtype=object),
            power_start_epoch_s=np.asarray([1000.0], dtype=np.float64),
            power=np.asarray([power], dtype=object),
        )

        split_path = root / "results" / "experimental_continuous_v1" / "splits" / "toy_H100_tp1.json"
        _write_json(split_path, {"train_indices": [], "val_indices": [], "test_indices": [0]})

        experimental_manifest_path = root / "results" / "experimental_continuous_v1" / "manifest.json"
        _write_json(
            experimental_manifest_path,
            {
                "schema_version": "experimental-continuous-v1",
                "configs": {
                    cfg: {
                        "dataset_npz": str(dataset_path),
                        "split_json": str(split_path),
                        "norm_params_json": str(norm_path),
                        "written": True,
                    }
                },
            },
        )

        throughput_db_path = root / "model" / "config" / "throughput_database.json"
        throughput_cfg = (
            {
                cfg: {
                    "prefill_rate_median_toks_per_s": 100.0,
                    "decode_rate_median_toks_per_s": 50.0,
                }
            }
            if include_throughput
            else {}
        )
        _write_json(
            throughput_db_path,
            {
                "schema_version": "stage0-throughput-v1",
                "configs": throughput_cfg,
            },
        )

        requests_path = root / "data" / "requests.json"
        _write_json(
            requests_path,
            {
                "input_lens": [32, 48],
                "output_lens": [20, 12],
                "request_timestamps": [1000.0, 1000.5],
            },
        )

        pair_manifest_path = root / "results" / "stage0" / "pair_manifest.csv"
        _write_pair_manifest(
            pair_manifest_path,
            pair_key=pair_key,
            json_path=str(requests_path),
        )

        return {
            "config_id": cfg,
            "run_manifest": run_manifest_path,
            "experimental_manifest": experimental_manifest_path,
            "throughput_db": throughput_db_path,
            "pair_manifest": pair_manifest_path,
        }

    def test_evaluate_from_artifacts_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._build_fixture(root, include_throughput=True)
            out_dir = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "eval_metrics"

            run = evaluate_from_artifacts(
                run_manifest=str(fx["run_manifest"]),
                experimental_manifest=str(fx["experimental_manifest"]),
                throughput_db=str(fx["throughput_db"]),
                pair_manifest_csv=str(fx["pair_manifest"]),
                out_dir=str(out_dir),
                config_ids=[fx["config_id"]],
                num_seeds=2,
                base_seed=11,
                device="cpu",
                decode_mode="stochastic",
                median_filter_window=1,
                plots=False,
            )

            self.assertEqual(int(run["summary"]["num_evaluated_configs"]), 1)
            self.assertTrue((out_dir / "per_seed_metrics.csv").exists())
            self.assertTrue((out_dir / "per_trace_metrics.csv").exists())
            self.assertTrue((out_dir / "config_summary.csv").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())

            with open(out_dir / "config_summary.csv", "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "evaluated")
            self.assertEqual(rows[0]["feature_set"], "f2")
            self.assertEqual(rows[0]["decode_mode"], "stochastic")
            self.assertTrue(np.isfinite(float(rows[0]["nrmse_median"])))

    def test_evaluate_from_artifacts_missing_throughput_fails_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._build_fixture(root, include_throughput=False)
            out_dir = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "eval_metrics"

            run = evaluate_from_artifacts(
                run_manifest=str(fx["run_manifest"]),
                experimental_manifest=str(fx["experimental_manifest"]),
                throughput_db=str(fx["throughput_db"]),
                pair_manifest_csv=str(fx["pair_manifest"]),
                out_dir=str(out_dir),
                config_ids=[fx["config_id"]],
                num_seeds=2,
                base_seed=11,
                device="cpu",
                decode_mode="stochastic",
                median_filter_window=1,
                plots=False,
            )

            self.assertEqual(int(run["summary"]["num_failed_configs"]), 1)
            with open(out_dir / "config_summary.csv", "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "failed")
            self.assertIn("throughput", rows[0]["reason"])


if __name__ == "__main__":
    unittest.main()
