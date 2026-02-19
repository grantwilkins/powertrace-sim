import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import torch

from model.classifiers.gru import GRUClassifier
from model.scripts.continuous_v1_gmm_bigru_infer import run_inference_from_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


class TestContinuousV1GMMBiGRUInfer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _build_fixture(self, root: Path, *, include_throughput: bool = True):
        cfg = "toy_H100_tp1"

        ckpt_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "checkpoints" / "toy_H100_tp1_k3_f2_best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model = GRUClassifier(Dx=2, K=3, H=8, num_layers=1)
        torch.save(model.state_dict(), ckpt_path)

        norm_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "norm_params" / "toy_H100_tp1.json"
        _write_json(
            norm_path,
            {
                "config_id": cfg,
                "dt": 0.25,
                "feature_set": "f2",
                "k": 3,
                "input_dim": 2,
                "hidden_dim": 8,
                "num_layers": 1,
                "active_mean": 0.0,
                "active_std": 1.0,
                "t_arrive_log_mean": 0.0,
                "t_arrive_log_std": 1.0,
                "delta_A_mean": 0.0,
                "delta_A_std": 1.0,
                "power_mean": 200.0,
                "power_std": 20.0,
                "power_min": 150.0,
                "power_max": 260.0,
            },
        )

        gmm_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "gmms" / "toy_H100_tp1_k3.json"
        _write_json(
            gmm_path,
            {
                "config_id": cfg,
                "k": 3,
                "covariance_type": "full",
                "means": [180.0, 210.0, 240.0],
                "variances": [9.0, 16.0, 25.0],
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
                        "checkpoint_path": str(ckpt_path),
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

        throughput_path = root / "model" / "config" / "throughput_database.json"
        throughput_payload = {
            "schema_version": "stage0-throughput-v1",
            "configs": {
                cfg: {
                    "prefill_rate_median_toks_per_s": 100.0,
                    "decode_rate_median_toks_per_s": 50.0,
                }
            }
            if include_throughput
            else {},
        }
        _write_json(throughput_path, throughput_payload)

        requests_path = root / "requests.json"
        _write_json(
            requests_path,
            {
                "requests": [
                    {"arrival_time": 0.0, "input_tokens": 32, "output_tokens": 20},
                    {"arrival_time": 0.5, "input_tokens": 64, "output_tokens": 8},
                ]
            },
        )

        return cfg, run_manifest_path, throughput_path, requests_path

    def test_run_inference_from_artifacts_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, run_manifest_path, throughput_path, requests_path = self._build_fixture(root, include_throughput=True)
            out_csv = root / "out" / "trace.csv"

            result = run_inference_from_artifacts(
                config_id=cfg,
                requests_json=str(requests_path),
                out_csv=str(out_csv),
                run_manifest=str(run_manifest_path),
                throughput_db=str(throughput_path),
                device="cpu",
                seed=0,
                decode_mode="stochastic",
                median_filter_window=1,
            )

            self.assertTrue(out_csv.exists())
            with open(out_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), int(result["T"]))
            self.assertEqual(rows[0].keys(), {"t_bin", "time_s", "power_w"})
            self.assertEqual(result["feature_set"], "f2")
            self.assertEqual(int(result["k"]), 3)

    def test_missing_throughput_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, run_manifest_path, throughput_path, requests_path = self._build_fixture(root, include_throughput=False)
            out_csv = root / "out" / "trace.csv"
            with self.assertRaisesRegex(ValueError, "throughput DB"):
                run_inference_from_artifacts(
                    config_id=cfg,
                    requests_json=str(requests_path),
                    out_csv=str(out_csv),
                    run_manifest=str(run_manifest_path),
                    throughput_db=str(throughput_path),
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
