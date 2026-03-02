import unittest
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np

from model.scripts.generate_methods_figures import (
    bic_sweep,
    normalize_rate,
    normalize_bic_values,
    rate_is_one,
    select_seed_nearest_median_nrmse,
    select_trace_by_lowest_mean_power,
    select_trace_by_best_median_nrmse,
    select_transition_dense_window,
)


class TestGenerateMethodsFigures(unittest.TestCase):
    def test_rate_normalization(self):
        self.assertTrue(rate_is_one("1"))
        self.assertTrue(rate_is_one("1.0"))
        self.assertTrue(rate_is_one(1))
        self.assertTrue(rate_is_one(1.0))
        self.assertFalse(rate_is_one("0.5"))
        self.assertFalse(rate_is_one("nan"))
        self.assertIsNone(normalize_rate("abc"))

    def test_trace_selection_dense_config(self):
        per_trace_path = Path("results/continuous_v1_gmm_bigru/k10_f2/eval_metrics/per_trace_metrics.csv")
        self.assertTrue(per_trace_path.exists(), msg=f"Missing test fixture: {per_trace_path}")
        rows = []
        import csv

        with open(per_trace_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))

        cfg = "llama-3-8b_H100_tp1"
        self.assertEqual(select_trace_by_best_median_nrmse(rows, config_id=cfg, rate=0.25), 8)
        self.assertEqual(select_trace_by_best_median_nrmse(rows, config_id=cfg, rate=1.0), 17)
        self.assertEqual(select_trace_by_best_median_nrmse(rows, config_id=cfg, rate=4.0), 59)

    def test_seed_selection_nearest_median(self):
        per_seed_path = Path("results/continuous_v1_gmm_bigru/k10_f2/eval_metrics/per_seed_metrics.csv")
        self.assertTrue(per_seed_path.exists(), msg=f"Missing test fixture: {per_seed_path}")
        rows = []
        import csv

        with open(per_seed_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))

        cfg = "llama-3-8b_H100_tp1"
        self.assertEqual(select_seed_nearest_median_nrmse(rows, config_id=cfg, trace_idx=8), 44)
        self.assertEqual(select_seed_nearest_median_nrmse(rows, config_id=cfg, trace_idx=17), 42)
        self.assertEqual(select_seed_nearest_median_nrmse(rows, config_id=cfg, trace_idx=59), 42)

    def test_bic_sweep_shape_and_bestk(self):
        rng = np.random.default_rng(42)
        a = rng.normal(loc=120.0, scale=8.0, size=250)
        b = rng.normal(loc=240.0, scale=12.0, size=250)
        power = np.concatenate([a, b], axis=0)
        sweep = bic_sweep(power)
        self.assertEqual(len(sweep["k_values"]), 19)
        self.assertEqual(len(sweep["bic_values"]), 19)
        self.assertGreaterEqual(int(sweep["best_k"]), 2)
        self.assertLessEqual(int(sweep["best_k"]), 20)

    def test_normalize_bic_values(self):
        raw = [120.0, 90.0, 75.0, 75.0]
        norm = normalize_bic_values(raw)
        self.assertEqual(len(norm), len(raw))
        self.assertAlmostEqual(float(min(norm)), 0.0, places=9)
        self.assertAlmostEqual(float(max(norm)), 1.0, places=9)
        self.assertAlmostEqual(norm[2], 0.0, places=9)
        self.assertAlmostEqual(norm[3], 0.0, places=9)

    def test_window_selection_prefers_dense_transitions(self):
        # 300 bins at dt=0.25 => 75s window for 300 bins? we only test selector behavior.
        # Build a series where middle region changes every step and edges are mostly flat.
        left = np.zeros((80,), dtype=np.float64)
        middle = (np.arange(160) % 2).astype(np.float64)
        right = np.ones((80,), dtype=np.float64) * 3.0
        a_t = np.concatenate([left, middle, right], axis=0)
        start, end = select_transition_dense_window(a_t, window_bins=120)
        self.assertGreaterEqual(start, 0)
        self.assertLessEqual(end, len(a_t))
        self.assertEqual(end - start, 120)
        # Ensure chosen window intersects the high-transition middle segment.
        self.assertLess(start, 240)
        self.assertGreater(end, 80)

    def test_select_trace_by_lowest_mean_power(self):
        traces = [
            np.array([100.0, 110.0, 120.0], dtype=np.float64),
            np.array([300.0, 310.0], dtype=np.float64),
            np.array([50.0, 60.0, 70.0], dtype=np.float64),
        ]
        idx = select_trace_by_lowest_mean_power(traces, [0, 1, 2])
        self.assertEqual(idx, 2)


if __name__ == "__main__":
    unittest.main()
