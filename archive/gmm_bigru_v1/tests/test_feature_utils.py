import unittest

import numpy as np

from model.classifiers.features import (
    compute_delta_active_requests,
    compute_inference_features,
    normalize_delta_active_requests,
)


class TestFeatureUtils(unittest.TestCase):
    def test_compute_delta_active_requests_basic(self):
        active = np.asarray([0.0, 1.0, 3.0, 2.0], dtype=np.float64)
        out = compute_delta_active_requests(active)
        np.testing.assert_allclose(out, np.asarray([0.0, 1.0, 2.0, -1.0], dtype=np.float64))

    def test_compute_delta_active_requests_empty(self):
        out = compute_delta_active_requests(np.asarray([], dtype=np.float64))
        self.assertEqual(out.shape, (0,))

    def test_compute_delta_active_requests_single(self):
        out = compute_delta_active_requests(np.asarray([5.0], dtype=np.float64))
        np.testing.assert_allclose(out, np.asarray([0.0], dtype=np.float64))

    def test_normalize_delta_active_requests_zscore(self):
        delta = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        out = normalize_delta_active_requests(delta, mean=1.0, std=0.5)
        np.testing.assert_allclose(out, np.asarray([-2.0, 0.0, 2.0], dtype=np.float32))

    def test_normalize_delta_active_requests_tiny_std(self):
        delta = np.asarray([0.0, 1.0], dtype=np.float64)
        out = normalize_delta_active_requests(delta, mean=0.0, std=1e-9)
        np.testing.assert_allclose(out, np.asarray([0.0, 1_000_000.0], dtype=np.float32))

    def test_compute_inference_features_basic_shape(self):
        requests = [
            {"arrival_time": 0.0, "input_tokens": 32, "output_tokens": 16},
            {"arrival_time": 0.5, "input_tokens": 16, "output_tokens": 8},
        ]
        config = {
            "lambda_prefill": 128.0,
            "lambda_decode": 64.0,
            "A_mean": 0.0,
            "A_std": 1.0,
            "T_arrive_log_mean": 0.0,
            "T_arrive_log_std": 1.0,
        }
        out = compute_inference_features(requests=requests, config=config, T=8, dt=0.25)
        self.assertEqual(out.shape, (8, 2))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_compute_inference_features_dt_nonpositive_raises(self):
        config = {
            "lambda_prefill": 1.0,
            "lambda_decode": 1.0,
            "A_mean": 0.0,
            "A_std": 1.0,
            "T_arrive_log_mean": 0.0,
            "T_arrive_log_std": 1.0,
        }
        with self.assertRaises(ValueError):
            compute_inference_features(
                requests=[{"arrival_time": 0.0, "input_tokens": 1, "output_tokens": 1}],
                config=config,
                dt=0.0,
            )

    def test_compute_inference_features_negative_throughput_raises(self):
        config = {
            "lambda_prefill": -1.0,
            "lambda_decode": 1.0,
            "A_mean": 0.0,
            "A_std": 1.0,
            "T_arrive_log_mean": 0.0,
            "T_arrive_log_std": 1.0,
        }
        with self.assertRaises(ValueError):
            compute_inference_features(
                requests=[{"arrival_time": 0.0, "input_tokens": 1, "output_tokens": 1}],
                config=config,
                T=4,
                dt=1.0,
            )

    def test_compute_inference_features_empty_requires_T(self):
        config = {
            "lambda_prefill": 1.0,
            "lambda_decode": 1.0,
            "A_mean": 0.0,
            "A_std": 1.0,
            "T_arrive_log_mean": 0.0,
            "T_arrive_log_std": 1.0,
        }
        with self.assertRaises(ValueError):
            compute_inference_features(requests=[], config=config, T=None, dt=1.0)

    def test_compute_inference_features_auto_T(self):
        requests = [{"arrival_time": 0.0, "input_tokens": 1.0, "output_tokens": 1.0}]
        config = {
            "lambda_prefill": 1.0,
            "lambda_decode": 1.0,
            "A_mean": 0.0,
            "A_std": 1.0,
            "T_arrive_log_mean": 0.0,
            "T_arrive_log_std": 1.0,
        }
        out = compute_inference_features(requests=requests, config=config, T=None, dt=1.0)
        self.assertEqual(out.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
