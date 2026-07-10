"""B2 tests: the shared arrival-alignment policies (D5) and the ARCH registry (D3).

Hand-worked claims:
- fold_1800 cancels a whole-hour clock skew exactly (3600 = 2 * 1800) and the
  run passes the [-2 s, 600 s] validity gate; rebase_into_window instead pins
  the earliest arrival to the window start, destroying the residual offset.
- In-window arrivals are untouched by both policies, and rebase reports
  shifted=False so callers can keep their original absolute timestamps
  bit-for-bit.
- The ARCH registry preserves the exact insertion order the ledger cache
  encoded as model_idx.
"""

import unittest

import numpy as np

from model.training_data.alignment import align_arrivals
from model.training_data.arch import ARCH, arch_from_manifest, get_arch


class TestAlignArrivals(unittest.TestCase):
    def test_fold_cancels_whole_hour_skew(self):
        power_t0 = 1_780_000_000.0
        # Arrivals stamped one hour ahead of the power clock, 10..40 s in.
        raw = power_t0 + 3600.0 + np.asarray([10.0, 20.0, 40.0])
        arrivals, ok, shifted = align_arrivals(raw, power_t0, policy="fold_1800")
        self.assertTrue(ok)
        self.assertTrue(shifted)
        np.testing.assert_allclose(arrivals, [10.0, 20.0, 40.0], atol=1e-9)

    def test_fold_rejects_unfixable_offset(self):
        power_t0 = 1_780_000_000.0
        # 700 s residual skew: not a multiple of 1800, outside [-2, 600].
        raw = power_t0 + 700.0 + np.asarray([10.0, 20.0])
        _, ok, _ = align_arrivals(raw, power_t0, policy="fold_1800")
        self.assertFalse(ok)

    def test_rebase_pins_out_of_window_arrivals_to_start(self):
        power_t0 = 1_780_000_000.0
        raw = power_t0 + 3600.0 + np.asarray([10.0, 20.0, 40.0])
        arrivals, ok, shifted = align_arrivals(
            raw, power_t0, policy="rebase_into_window", dt=0.25, trace_duration_s=60.0
        )
        self.assertTrue(ok)
        self.assertTrue(shifted)
        # Earliest arrival pinned to 0: the +10 s residual is destroyed.
        np.testing.assert_allclose(arrivals, [0.0, 10.0, 30.0], atol=1e-9)

    def test_in_window_arrivals_untouched_by_both(self):
        power_t0 = 1_780_000_000.0
        raw = power_t0 + np.asarray([5.0, 15.0, 25.0])
        rebased, _, shifted = align_arrivals(
            raw, power_t0, policy="rebase_into_window", dt=0.25, trace_duration_s=60.0
        )
        self.assertFalse(shifted)
        np.testing.assert_allclose(rebased, [5.0, 15.0, 25.0], atol=1e-9)
        folded, ok, _ = align_arrivals(raw, power_t0, policy="fold_1800")
        self.assertTrue(ok)
        np.testing.assert_allclose(folded, [5.0, 15.0, 25.0], atol=1e-9)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            align_arrivals(np.asarray([0.0]), 0.0, policy="nearest")

    def test_rebase_requires_window_parameters(self):
        with self.assertRaises(ValueError):
            align_arrivals(np.asarray([0.0]), 0.0, policy="rebase_into_window")


class TestArchRegistry(unittest.TestCase):
    def test_insertion_order_is_the_ledger_model_index(self):
        # model_idx = list(ARCH).index(model) is stored in ledger caches;
        # this order is frozen (append-only).
        self.assertEqual(
            list(ARCH),
            [
                "llama-3-8b",
                "deepseek-r1-distill-8b",
                "llama-3-70b",
                "deepseek-r1-distill-70b",
                "llama-3-405b",
                "gpt-oss-120b",
                "gpt-oss-20b",
            ],
        )

    def test_get_arch_known_and_unknown(self):
        self.assertEqual(get_arch("llama-3-70b")["d_model"], 8192)
        with self.assertRaises(KeyError):
            get_arch("mystery-model")

    def test_arch_from_manifest_passthrough_and_missing(self):
        arch = {"family": "dense-70b", "n_layers": 80}
        self.assertIs(arch_from_manifest({"arch": arch}), arch)
        with self.assertRaises(ValueError):
            arch_from_manifest({"arch": {}})


if __name__ == "__main__":
    unittest.main()
