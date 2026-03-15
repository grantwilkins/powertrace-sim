import unittest

import numpy as np

from model.classifiers.metrics import (
    compute_aggregate_power_metrics,
    compute_power_metrics,
)


class TestPowerMetrics(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
