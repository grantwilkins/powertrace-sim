import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

from model.scripts.figure_d1_conditional_entropy import (  # noqa: E402
    _parse_csv_list,
    _resolve_legacy_f6_checkpoint,
    build_requests_from_json,
    deterministic_pairkey_json_path,
    estimate_knn_joint_mi_nmi,
    estimate_plugin_mi_nmi,
)


class TestFigureD1ConditionalEntropy(unittest.TestCase):
    def test_plugin_mi_perfect_predictor_near_one(self):
        n = 4000
        z = np.concatenate(
            [
                np.zeros((n // 2,), dtype=np.int64),
                np.ones((n // 2,), dtype=np.int64),
            ]
        )
        x = z.astype(np.float64).reshape(-1, 1)
        out = estimate_plugin_mi_nmi(x, z, n_bins=20)
        self.assertTrue(np.isfinite(out["nmi"]))
        self.assertGreater(float(out["nmi"]), 0.98)

    def test_plugin_mi_independent_predictor_near_zero(self):
        rng = np.random.default_rng(42)
        n = 5000
        z = rng.integers(low=0, high=3, size=n, dtype=np.int64)
        x = rng.normal(size=(n, 1)).astype(np.float64)
        out = estimate_plugin_mi_nmi(x, z, n_bins=20)
        self.assertTrue(np.isfinite(out["nmi"]))
        self.assertLess(float(out["nmi"]), 0.05)

    def test_knn_joint_mi_separable_6d_high(self):
        rng = np.random.default_rng(7)
        feature_traces = []
        label_traces = []
        for _ in range(3):
            n_per_class = 250
            x0 = rng.normal(loc=-3.0, scale=0.35, size=(n_per_class, 6))
            x1 = rng.normal(loc=3.0, scale=0.35, size=(n_per_class, 6))
            y0 = np.zeros((n_per_class,), dtype=np.int64)
            y1 = np.ones((n_per_class,), dtype=np.int64)
            feature_traces.append(np.concatenate([x0, x1], axis=0))
            label_traces.append(np.concatenate([y0, y1], axis=0))

        out = estimate_knn_joint_mi_nmi(feature_traces=feature_traces, label_traces=label_traces, knn_k=15)
        self.assertTrue(np.isfinite(out["nmi"]))
        self.assertGreater(float(out["nmi"]), 0.8)

    def test_entropy_zero_class_safe(self):
        z = np.zeros((300,), dtype=np.int64)
        x = np.random.default_rng(0).normal(size=(300, 2))
        out = estimate_plugin_mi_nmi(x, z, n_bins=20)
        self.assertEqual(float(out["h_z"]), 0.0)
        self.assertEqual(float(out["mi"]), 0.0)
        self.assertEqual(float(out["nmi"]), 0.0)

    def test_pairkey_fallback_resolver(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_id = "gpt-oss-120b_H100_tp8"
            pair_key = "model=gpt-oss-120b|tp=8|rate=1.0|iter=2|date=2025-02-24"
            p = (
                root
                / "data"
                / "extraneous-data"
                / "benchmark-gpt-oss-120b-h100"
                / "tp8"
                / "gpt-oss-120b_tp8_rate1.0_iter2_2025-02-24.json"
            )
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")

            with patch("model.scripts.figure_d1_conditional_entropy._repo_root", return_value=root):
                resolved = deterministic_pairkey_json_path(config_id=config_id, pair_key=pair_key)
            self.assertIsNotNone(resolved)
            self.assertEqual(Path(resolved), p)

    def test_missing_recorded_timestamps_dropped_when_strict(self):
        with tempfile.TemporaryDirectory() as td:
            request_path = Path(td) / "req.json"
            payload = {
                "duration": 30.0,
                "input_lens": [64, 32, 16],
                "output_lens": [16, 8, 4],
            }
            request_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "request_timestamps"):
                build_requests_from_json(
                    request_json_path=request_path,
                    power_start_epoch_s=0.0,
                    trace_duration_s=30.0,
                    dt=0.25,
                    require_recorded_timestamps=True,
                )

            requests = build_requests_from_json(
                request_json_path=request_path,
                power_start_epoch_s=0.0,
                trace_duration_s=30.0,
                dt=0.25,
                require_recorded_timestamps=False,
            )
            self.assertEqual(len(requests), 3)

    def test_legacy_checkpoint_resolver(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = "llama-3-8b_H100_tp2"
            ckpt = root / "new_weights" / "llama-3-8b_h100_tp2.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(b"")

            resolved = _resolve_legacy_f6_checkpoint(cfg, [str(root / "new_weights")])
            self.assertIsNotNone(resolved)
            self.assertEqual(Path(resolved), ckpt)

    def test_parse_csv_list(self):
        out = _parse_csv_list("a,b, c ,,")
        self.assertEqual(out, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
