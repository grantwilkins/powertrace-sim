import os
import unittest

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

from scripts.eval.feature_sufficiency_figure import (
    bootstrap_median_ci,
    build_trace_feature_subsets,
    compute_info_retention,
    map_config_to_npz_stem,
)


class TestFeatureSufficiencyFigure(unittest.TestCase):
    def test_mapping_deepseek_distill_to_npz_stem(self):
        cfg = "deepseek-r1-distill-70b_H100_tp8"
        stem = map_config_to_npz_stem(cfg)
        self.assertEqual(stem, "vllm-benchmark_deepseek-r1-70b_h100")

    def test_info_retention_formula_sanity(self):
        ir_abs, ir_vs_f6 = compute_info_retention(ce_subset=0.6, ce_null=1.0, ce_f6=0.5)
        self.assertAlmostEqual(ir_abs, 0.4, places=8)
        self.assertAlmostEqual(ir_vs_f6, 0.8, places=8)

        ir_abs_f6, ir_vs_f6_f6 = compute_info_retention(ce_subset=0.5, ce_null=1.0, ce_f6=0.5)
        self.assertAlmostEqual(ir_abs_f6, 0.5, places=8)
        self.assertAlmostEqual(ir_vs_f6_f6, 1.0, places=8)

        ir_abs_worse, ir_vs_f6_worse = compute_info_retention(ce_subset=1.2, ce_null=1.0, ce_f6=0.5)
        self.assertLess(ir_abs_worse, 0.0)
        self.assertLess(ir_vs_f6_worse, 0.0)

    def test_alignment_for_all_subsets(self):
        n = 20
        timestamps = 1000.0 + 0.25 * np.arange(n, dtype=np.float64)
        active = np.array([0, 0, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 1, 0, 1, 2, 1, 1, 0, 0], dtype=np.float64)
        prefill = np.array([0, 0, 8, 4, 0, 0, 0, 5, 0, 0, 4, 0, 0, 0, 6, 0, 0, 0, 0, 0], dtype=np.float64)
        decode = np.array([0, 0, 0, 30, 28, 22, 16, 18, 20, 26, 22, 18, 15, 0, 10, 14, 12, 8, 0, 0], dtype=np.float64)

        req_ts = np.array([1000.5, 1001.3, 1002.2, 1002.8, 1003.7, 1004.6], dtype=np.float64)
        in_tok = np.array([12, 24, 18, 30, 16, 28], dtype=np.float64)
        out_tok = np.array([64, 90, 70, 120, 85, 110], dtype=np.float64)

        feats = build_trace_feature_subsets(
            timestamps=timestamps,
            active_requests=active,
            prefill_tokens=prefill,
            decode_tokens=decode,
            request_timestamps=req_ts,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

        lengths = [int(feats[k].shape[0]) for k in ["A", "ΔA", "F2", "F3", "F6"]]
        self.assertTrue(all(l == (n - 1) for l in lengths))
        self.assertEqual(int(feats["A"].shape[1]), 1)
        self.assertEqual(int(feats["ΔA"].shape[1]), 1)
        self.assertEqual(int(feats["F2"].shape[1]), 2)
        self.assertEqual(int(feats["F3"].shape[1]), 3)
        self.assertEqual(int(feats["F6"].shape[1]), 6)

    def test_bootstrap_determinism(self):
        values = [0.6, 0.8, 0.7, 0.9, 0.5, 0.65]
        out1 = bootstrap_median_ci(values, n_bootstrap=500, seed=42)
        out2 = bootstrap_median_ci(values, n_bootstrap=500, seed=42)
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
