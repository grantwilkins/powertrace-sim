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
from model.utils.io import write_json as _write_json
from model.pipeline.inference import run_inference_from_artifacts
from model.utils.provenance import file_identity


class TestContinuousV1GMMBiGRUInfer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _build_fixture(
        self, root: Path, *, include_throughput: bool = True,
        config_id: str = "toy_H100_tp1", bound_throughput=None,
    ):
        cfg = config_id
        bound = (
            dict(bound_throughput)
            if bound_throughput is not None
            else (
                {"lambda_prefill": 100.0, "lambda_decode": 50.0}
                if include_throughput
                else None
            )
        )

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
                        "artifact_identities": {
                            "checkpoint": file_identity(ckpt_path),
                            "trained_norm": file_identity(norm_path),
                            "gmm": file_identity(gmm_path),
                        },
                        **(
                            {"throughput": bound} if bound is not None else {}
                        ),
                    }
                },
            },
        )

        throughput_path = root / "model" / "throughput_database.json"
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
            self.assertTrue(Path(result["inference_manifest"]).exists())
            with open(out_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), int(result["T"]))
            self.assertEqual(
                set(rows[0].keys()),
                {
                    "t_bin",
                    "time_s",
                    "power_w",
                    "generation_mode",
                    "generation_mode_label",
                },
            )
            self.assertEqual(result["generation_mode"], "iid")
            self.assertEqual(rows[0]["generation_mode"], "iid")
            self.assertEqual(float(rows[0]["time_s"]), 0.25)
            self.assertEqual(result["feature_set"], "f2")
            self.assertEqual(int(result["k"]), 3)

    def test_missing_throughput_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, run_manifest_path, throughput_path, requests_path = self._build_fixture(root, include_throughput=False)
            out_csv = root / "out" / "trace.csv"
            with self.assertRaisesRegex(ValueError, "missing bound train throughput"):
                run_inference_from_artifacts(
                    config_id=cfg,
                    requests_json=str(requests_path),
                    out_csv=str(out_csv),
                    run_manifest=str(run_manifest_path),
                    throughput_db=str(throughput_path),
                    device="cpu",
                )

    def test_mutable_throughput_db_cannot_replace_missing_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, run_manifest_path, throughput_path, requests_path = self._build_fixture(root)
            payload = json.loads(Path(run_manifest_path).read_text())
            del payload["configs"][cfg]["throughput"]
            _write_json(run_manifest_path, payload)

            with self.assertRaisesRegex(ValueError, "missing bound train throughput"):
                run_inference_from_artifacts(
                    config_id=cfg,
                    requests_json=str(requests_path),
                    out_csv=str(root / "out" / "trace.csv"),
                    run_manifest=str(run_manifest_path),
                    throughput_db=str(throughput_path),
                    device="cpu",
                )

    def test_stochastic_inference_requires_seed(self):
        with self.assertRaisesRegex(ValueError, "explicit seed"):
            run_inference_from_artifacts(
                config_id="unused_H100_tp1",
                requests_json="unused.json",
                out_csv="unused.csv",
                seed=None,
            )

    def test_full_hf_config_uses_bound_training_throughput_without_a_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, run_manifest, throughput_db, requests = self._build_fixture(
                root,
                include_throughput=False,
                config_id="org/toy-model_H100_tp1",
                bound_throughput={"lambda_prefill": 100.0, "lambda_decode": 50.0},
            )
            result = run_inference_from_artifacts(
                config_id=cfg,
                requests_json=str(requests),
                out_csv=str(root / "trace.csv"),
                run_manifest=str(run_manifest),
                throughput_db=str(root / "missing-throughput.json"),
                device="cpu",
                seed=7,
            )
            self.assertGreater(result["T"], 0)
            self.assertEqual(result["throughput_source"], "run_manifest_bound")

    def test_recorded_artifact_identity_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, run_manifest, throughput_db, requests = self._build_fixture(root)
            payload = json.loads(Path(run_manifest).read_text())
            norm_path = Path(payload["configs"][cfg]["norm_params_path"])
            norm_path.write_text(norm_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                run_inference_from_artifacts(
                    config_id=cfg,
                    requests_json=str(requests),
                    out_csv=str(root / "trace.csv"),
                    run_manifest=str(run_manifest),
                    throughput_db=str(throughput_db),
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
