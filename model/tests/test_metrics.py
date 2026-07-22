"""
Claim:
Power metrics implement their stated aggregation, energy, resolution, and
load-duration conventions at the correct trace or facility level.

Plausible wrong implementations:
- Integrate per-bin average power with a trapezoid rule and lose endpoint bins.
- Difference native samples while labeling the result as a coarser ramp.
- Round a non-integral resolution ratio and report the wrong time scale.
- Use percentile interpolation instead of exceedance rank for the LDC.
- Pool trace boundaries when computing temporal correlation.
"""

import unittest

import numpy as np

from model.metrics import (
    _total_energy_from_bins,
    autocorrelation_r2,
    autocorrelation_r2_aggregate,
    coefficient_of_variation,
    compute_aggregate_power_metrics,
    compute_power_metrics,
    downsample_mean,
    ks_statistic,
    load_duration_value,
    load_factor,
    ramp_stats,
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


class TestFacilityMetrics(unittest.TestCase):
    """Hand-worked values for the facility metrics unified in Phase A."""

    def test_downsample_mean_250ms_to_1s(self):
        # Four 250 ms samples per second: means are (1+2+3+4)/4 and (5+6+7+8)/4.
        x = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
        out = downsample_mean(x, dt=0.25, resolution_s=1.0)
        np.testing.assert_array_equal(out, np.asarray([2.5, 6.5]))

    def test_downsample_mean_same_resolution_is_identity(self):
        x = np.asarray([3.0, 1.0, 4.0, 1.5], dtype=np.float64)
        np.testing.assert_array_equal(downsample_mean(x, dt=1.0, resolution_s=1.0), x)

    def test_downsample_mean_drops_trailing_partial_window(self):
        x = np.asarray([1, 2, 3, 4, 5], dtype=np.float64)
        out = downsample_mean(x, dt=0.5, resolution_s=1.0)
        np.testing.assert_array_equal(out, np.asarray([1.5, 3.5]))

    def test_downsample_mean_rejects_ambiguous_resolution(self):
        with self.assertRaises(ValueError):
            downsample_mean(np.arange(8), dt=0.3, resolution_s=1.0)
        with self.assertRaises(ValueError):
            downsample_mean(np.arange(8), dt=1.0, resolution_s=0.5)

    def test_ramp_stats_resolution_changes_ramps(self):
        # 250 ms sawtooth [0,10,0,10,...]: native steps are +/-10, but 1 s means
        # are constant 5.0 -> every 1 s ramp is exactly 0. The two resolutions
        # MUST differ by construction; an implicit native diff would return 10.
        x = np.asarray([0.0, 10.0] * 8, dtype=np.float64)
        native = ramp_stats(x, dt=0.25, resolution_s=0.25)
        one_s = ramp_stats(x, dt=0.25, resolution_s=1.0)
        self.assertAlmostEqual(native["ramp_max_up"], 10.0, places=12)
        self.assertAlmostEqual(native["ramp_max_down"], -10.0, places=12)
        self.assertAlmostEqual(one_s["ramp_max_up"], 0.0, places=12)
        self.assertAlmostEqual(one_s["ramp_p95_abs"], 0.0, places=12)
        self.assertEqual(one_s["resolution_s"], 1.0)

    def test_ramp_stats_hand_worked_signed_extremes(self):
        # Diffs of [0, 3, 1, 6]: [+3, -2, +5] -> max up 5, max down -2, p50 = 3.
        x = np.asarray([0.0, 3.0, 1.0, 6.0], dtype=np.float64)
        out = ramp_stats(x, dt=1.0, resolution_s=1.0)
        self.assertAlmostEqual(out["ramp_max_up"], 5.0, places=12)
        self.assertAlmostEqual(out["ramp_max_down"], -2.0, places=12)
        self.assertAlmostEqual(out["ramp_p50"], 3.0, places=12)

    def test_ramp_stats_single_point_is_nan(self):
        out = ramp_stats(np.asarray([5.0]), dt=1.0, resolution_s=1.0)
        self.assertTrue(np.isnan(out["ramp_p50"]))

    def test_load_duration_value_exceedance_rank(self):
        # 10 values 10..100 descending-sorted; frac 0.05 -> floor(0.5) = rank 0
        # (the maximum); frac 0.25 -> floor(2.5) = rank 2 -> 80.
        x = np.arange(10.0, 101.0, 10.0)
        self.assertAlmostEqual(load_duration_value(x, 0.05), 100.0, places=12)
        self.assertAlmostEqual(load_duration_value(x, 0.25), 80.0, places=12)
        self.assertTrue(np.isnan(load_duration_value(np.asarray([]), 0.05)))

    def test_load_duration_rejects_invalid_fraction(self):
        for fraction in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                load_duration_value(np.asarray([1.0]), fraction)

    def test_load_factor_hand_worked(self):
        # mean([2, 4, 6]) / max = 4/6.
        x = np.asarray([2.0, 4.0, 6.0])
        self.assertAlmostEqual(load_factor(x), 4.0 / 6.0, places=12)
        self.assertTrue(np.isnan(load_factor(np.asarray([0.0, -1.0]))))

    def test_coefficient_of_variation_hand_worked(self):
        # [1, 3]: mean 2, population std 1 -> CoV 0.5.
        self.assertAlmostEqual(
            coefficient_of_variation(np.asarray([1.0, 3.0])), 0.5, places=12
        )
        self.assertTrue(np.isnan(coefficient_of_variation(np.asarray([0.0, 0.0]))))


if __name__ == "__main__":
    unittest.main()
