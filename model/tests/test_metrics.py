import unittest

import numpy as np

from model.classifiers.metrics import (
    _total_energy_from_bins,
    autocorrelation_r2,
    autocorrelation_r2_aggregate,
    compute_aggregate_power_metrics,
    compute_power_metrics,
    ks_statistic,
)


class TestPowerMetrics(unittest.TestCase):
    def test_ks_statistic_identical_returns_zero(self):
        x = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        y = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        self.assertAlmostEqual(ks_statistic(x, y), 0.0, places=9)

    def test_ks_statistic_different_positive(self):
        x = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        y = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        self.assertGreater(ks_statistic(x, y), 0.0)

    def test_ks_statistic_empty_returns_nan(self):
        x = np.asarray([], dtype=np.float64)
        y = np.asarray([1.0], dtype=np.float64)
        self.assertTrue(np.isnan(ks_statistic(x, y)))

    def test_autocorrelation_r2_identical_traces_near_one(self):
        x = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        self.assertAlmostEqual(autocorrelation_r2(x, x, max_lag=3), 1.0, places=9)

    def test_autocorrelation_r2_short_trace(self):
        x = np.asarray([0.0, 1.0], dtype=np.float64)
        y = np.asarray([0.0, 1.0], dtype=np.float64)
        self.assertTrue(np.isnan(autocorrelation_r2(x, y, max_lag=3)))

    def test_delta_energy_pct_uses_total_bin_energy_not_trapezoid(self):
        gt = np.asarray([1.0, 0.0, 1.0], dtype=np.float64)
        pred = np.asarray([0.0, 2.0, 0.0], dtype=np.float64)

        metrics = compute_power_metrics(gt, pred, dt=0.5)

        self.assertAlmostEqual(float(metrics["delta_energy_pct"]), 0.0, places=9)

    def test_delta_energy_pct_is_absolute_total_energy_error(self):
        gt = np.asarray([10.0, 10.0], dtype=np.float64)
        pred = np.asarray([8.0, 8.0], dtype=np.float64)

        metrics = compute_power_metrics(gt, pred, dt=1.0)

        self.assertAlmostEqual(float(metrics["delta_energy_pct"]), 20.0, places=9)

    def test_compute_aggregate_power_metrics_pools_ks_across_heldout_points(self):
        gt_traces = [
            np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        ]
        pred_traces = [
            np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        ]

        metrics = compute_aggregate_power_metrics(gt_traces, pred_traces, dt=1.0)

        self.assertAlmostEqual(float(metrics["ks_stat"]), 0.0, places=9)
        self.assertAlmostEqual(float(metrics["delta_energy_pct"]), 0.0, places=9)

    def test_compute_aggregate_power_metrics_acf_uses_average_trace_acfs(self):
        gt_traces = [
            np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        ]
        pred_traces = [
            np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64),
        ]

        metrics = compute_aggregate_power_metrics(
            gt_traces, pred_traces, dt=1.0, acf_max_lag=3
        )

        self.assertTrue(np.isfinite(float(metrics["acf_r2"])))

    def test_autocorrelation_r2_aggregate_pools_traces(self):
        gt_traces = [
            np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64),
        ]
        pred_traces = [
            np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([1.0, 2.0, 1.0, 2.0, 1.0, 2.0], dtype=np.float64),
        ]

        r2 = autocorrelation_r2_aggregate(gt_traces, pred_traces, max_lag=4)
        self.assertAlmostEqual(float(r2), 1.0, places=9)

    def test_total_energy_from_bins_basic(self):
        values = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
        self.assertAlmostEqual(float(_total_energy_from_bins(values, dt=0.25)), 15.0, places=9)

    def test_compute_power_metrics_all_keys_present(self):
        gt = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        pred = np.asarray([1.1, 2.1, 2.9, 3.9], dtype=np.float64)
        metrics = compute_power_metrics(gt, pred, dt=1.0, acf_max_lag=2)
        self.assertEqual(
            set(metrics.keys()),
            {
                "ks_stat",
                "acf_r2",
                "nrmse",
                "p95_error_pct",
                "p99_error_pct",
                "delta_energy_pct",
            },
        )


if __name__ == "__main__":
    unittest.main()
