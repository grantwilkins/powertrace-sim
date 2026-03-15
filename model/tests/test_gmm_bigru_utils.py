import os
import unittest

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

from model.classifiers.gmm_bigru import (
    build_features_from_active,
    build_rollout_features_from_requests,
    build_state_labels,
    fit_power_gmm,
    generate_gmm_bigru_trace,
    gmm_params_to_json_dict,
    load_gmm_params_json_dict,
)


class TestGMMBiGRUUtils(unittest.TestCase):
    def test_fit_power_gmm_sorts_means_and_labels(self):
        rng = np.random.default_rng(123)
        x = np.concatenate(
            [
                rng.normal(90.0, 2.0, size=300),
                rng.normal(140.0, 3.0, size=300),
                rng.normal(220.0, 4.0, size=300),
            ]
        ).astype(np.float64)

        fitted = fit_power_gmm(x, k=3, random_state=7, n_init=5, max_iter=200, reg_covar=1e-6)
        means = np.asarray(fitted["means"], dtype=np.float64)
        self.assertEqual(means.shape[0], 3)
        self.assertTrue(np.all(np.diff(means) >= 0.0))

        labels = build_state_labels(x, fitted)
        self.assertEqual(labels.shape[0], x.shape[0])
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))
        self.assertGreaterEqual(np.unique(labels).size, 2)

    def test_gmm_json_roundtrip(self):
        rng = np.random.default_rng(11)
        x = rng.normal(150.0, 10.0, size=500).astype(np.float64)
        fitted = fit_power_gmm(x, k=2, random_state=4)
        payload = gmm_params_to_json_dict(fitted)
        loaded = load_gmm_params_json_dict(payload)

        self.assertEqual(int(loaded["k"]), 2)
        self.assertEqual(np.asarray(loaded["means"]).shape[0], 2)
        self.assertEqual(np.asarray(loaded["variances"]).shape[0], 2)
        self.assertEqual(np.asarray(loaded["weights"]).shape[0], 2)

    def test_build_features_from_active_f2(self):
        active = np.array([0.0, 1.0, 3.0, 2.0], dtype=np.float64)
        norm = {
            "active_mean": 0.0,
            "active_std": 1.0,
            "delta_A_mean": 0.0,
            "delta_A_std": 1.0,
            "t_arrive_log_mean": 0.0,
            "t_arrive_log_std": 1.0,
        }
        out = build_features_from_active(
            active_requests=active,
            t_arrive_log=None,
            norm=norm,
            feature_set="f2",
            max_length=3,
        )
        x = np.asarray(out["features_norm"], dtype=np.float32)
        self.assertEqual(tuple(x.shape), (3, 2))
        self.assertTrue(np.allclose(x[:, 0], np.array([1.0, 3.0, 2.0], dtype=np.float32)))
        self.assertTrue(np.allclose(x[:, 1], np.array([1.0, 2.0, -1.0], dtype=np.float32)))

    def test_build_rollout_features_shapes_and_finite(self):
        requests = [
            {"arrival_time": 0.0, "input_tokens": 32, "output_tokens": 16},
            {"arrival_time": 0.5, "input_tokens": 64, "output_tokens": 8},
        ]
        throughput = {"lambda_prefill": 128.0, "lambda_decode": 64.0}
        norm = {
            "active_mean": 0.0,
            "active_std": 1.0,
            "t_arrive_log_mean": 0.0,
            "t_arrive_log_std": 1.0,
            "delta_A_mean": 0.0,
            "delta_A_std": 1.0,
        }

        out_f2 = build_rollout_features_from_requests(
            requests=requests,
            throughput=throughput,
            norm=norm,
            T=12,
            dt=0.25,
            feature_set="f2",
        )
        x2 = np.asarray(out_f2["features_norm"], dtype=np.float32)
        self.assertEqual(tuple(x2.shape), (12, 2))
        self.assertTrue(np.all(np.isfinite(x2)))

    def test_generate_trace_seeded_and_clamped(self):
        logits = np.zeros((20, 2), dtype=np.float64)
        gmm = {
            "k": 2,
            "covariance_type": "full",
            "means": np.array([100.0, 220.0], dtype=np.float64),
            "variances": np.array([4.0, 9.0], dtype=np.float64),
            "weights": np.array([0.5, 0.5], dtype=np.float64),
            "order": np.array([0, 1], dtype=np.int64),
            "label_map": np.array([0, 1], dtype=np.int64),
        }
        out_a = generate_gmm_bigru_trace(
            logits=logits,
            gmm_params=gmm,
            seed=123,
            decode_mode="stochastic",
            median_filter_window=1,
            clamp_range=(120.0, 180.0),
        )
        out_b = generate_gmm_bigru_trace(
            logits=logits,
            gmm_params=gmm,
            seed=123,
            decode_mode="stochastic",
            median_filter_window=1,
            clamp_range=(120.0, 180.0),
        )
        p_a = np.asarray(out_a["power_w"], dtype=np.float64)
        p_b = np.asarray(out_b["power_w"], dtype=np.float64)
        self.assertEqual(p_a.shape[0], 20)
        self.assertTrue(np.allclose(p_a, p_b))
        self.assertTrue(np.all(np.isfinite(p_a)))
        lo = 120.0 - (0.05 * (180.0 - 120.0))
        hi = 180.0 + (0.05 * (180.0 - 120.0))
        self.assertTrue(np.all(p_a >= lo - 1e-6))
        self.assertTrue(np.all(p_a <= hi + 1e-6))


if __name__ == "__main__":
    unittest.main()
