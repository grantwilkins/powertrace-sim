"""
Claim:
The deployed physics kernel evaluates the declared 11-term equations at node
level, multiplies only dynamic power by family efficiency, then applies the
declared causal lag and per-GPU physical cap.

Plausible wrong implementations:
- Apply FP8 scaling in the wrong direction or to byte traffic.
- Apply the family multiplier to idle/link standing power.
- Treat a per-GPU cap as a node-total cap.
- Apply EMA before the moving average or use future samples.
- Lose seed reproducibility in stochastic residual generation.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.classifiers.physics import (
    FEATURE_ORDER,
    SCHEMA_VERSION,
    add_stochastic_residual,
    apply_meter_lag,
    ledger_power_features,
    load_physics_artifact,
    predict_mean_node_power,
)


def _ledger():
    return {
        "pre_tok": np.asarray([0.0, 2.0]),
        "dec_tok": np.asarray([0.0, 3.0]),
        "w_read": np.asarray([0.0, 10.0]),
        "kv_read": np.asarray([0.0, 6.0]),
        "w_read_pre": np.asarray([0.0, 7.0]),
        "kv_write": np.asarray([0.0, 5.0]),
        "comm": np.asarray([0.0, 4.0]),
    }


class TestPhysicsKernel(unittest.TestCase):
    def test_loader_preserves_fit_source_provenance(self):
        """Artifact loading must preserve ledger hashes, sources, and dirty state."""
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "feature_order": list(FEATURE_ORDER),
            "fit_revision": "abc123",
            "fit_revision_dirty": True,
            "training_ledger_artifact": {
                "path": "chosen.npz",
                "sha256": "ledger-hash",
                "run_index_path": "chosen.runs.json",
                "run_index_sha256": "index-hash",
            },
            "hardware": {"H100": {"training_source_ids": ["cfg|pair"]}},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.json"
            path.write_text(json.dumps(artifact))
            loaded = load_physics_artifact(path)
        self.assertTrue(loaded["fit_revision_dirty"])
        self.assertEqual(
            loaded["training_ledger_artifact"]["run_index_sha256"], "index-hash"
        )
        self.assertEqual(loaded["hardware"]["H100"]["training_source_ids"], ["cfg|pair"])

    def test_feature_units_and_fp8_scaling(self):
        bf16 = ledger_power_features(
            _ledger(), {"n_active": 10.0, "fp8": 0}, tp=2, hbm_bandwidth_bytes_s=8.0
        )
        fp8 = ledger_power_features(
            _ledger(), {"n_active": 10.0, "fp8": 1}, tp=2, hbm_bandwidth_bytes_s=8.0
        )
        self.assertEqual(bf16["flops_pre"][1], 40.0)
        self.assertEqual(bf16["flops_dec"][1], 60.0)
        self.assertEqual(fp8["flops_pre"][1], 20.0)
        self.assertEqual(fp8["w_read_pre"][1], 7.0)
        self.assertEqual(bf16["tp_link"][0], 2.0)

    def test_family_multiplier_lag_and_node_cap(self):
        coefficients = {key: 0.0 for key in FEATURE_ORDER}
        coefficients.update({"tp": 10.0, "busy_tp": 20.0})
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "feature_order": list(FEATURE_ORDER),
            "hardware": {
                "H100": {
                    "coefficients": coefficients,
                    "family_multipliers": {"unit": 2.0},
                    "hbm_bandwidth_bytes_s": 8.0,
                    "lag": {"moving_average_bins": 1, "ema_alpha": 1.0},
                    "cap_w_per_gpu": 45.0,
                }
            },
        }
        out = predict_mean_node_power(
            _ledger(), {"n_active": 10.0, "fp8": 0, "family": "unit"},
            tp=2, hardware="H100", artifact=artifact,
        )
        # Idle = 2*10 = 20 W. Busy dynamic = (2*20)*2 family = 80 W.
        # Node cap = 2*45 = 90 W, so [20, 100] becomes [20, 90].
        np.testing.assert_array_equal(out, [20.0, 90.0])

    def test_causal_ma_then_ema(self):
        out = apply_meter_lag(
            np.asarray([0.0, 10.0, 10.0]), moving_average_bins=2, ema_alpha=0.5
        )
        np.testing.assert_allclose(out, [0.0, 2.5, 6.25])

    def test_lag_time_constant_is_preserved_across_timesteps(self):
        coefficients = {key: 0.0 for key in FEATURE_ORDER}
        coefficients["busy_tp"] = 1.0
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "feature_order": list(FEATURE_ORDER),
            "dt_s": 1.0,
            "hardware": {"H100": {
                "coefficients": coefficients,
                "family_multipliers": {},
                "hbm_bandwidth_bytes_s": 1.0,
                "lag": {"moving_average_bins": 1, "ema_alpha": 0.5},
                "cap_w_per_gpu": 100.0,
            }},
        }
        ledger = {key: np.zeros(5) for key in _ledger()}
        ledger["pre_tok"][1:] = 1.0
        out = predict_mean_node_power(
            ledger, {"n_active": 1.0}, tp=1, hardware="H100",
            artifact=artifact, dt_s=0.25,
        )
        expected_alpha = 1.0 - 0.5 ** 0.25
        # The fitted one-bin moving average represents one second, so at 250 ms
        # the first busy sample contributes one quarter before the converted EMA.
        self.assertAlmostEqual(out[1], 0.25 * expected_alpha)
        expected_at_one_second = 0.0
        for moving_average_value in (0.25, 0.5, 0.75, 1.0):
            expected_at_one_second += expected_alpha * (
                moving_average_value - expected_at_one_second
            )
        self.assertAlmostEqual(out[4], expected_at_one_second)

    def test_residual_seed_and_zero_noise(self):
        mean = np.full(8, 100.0)
        np.testing.assert_array_equal(
            add_stochastic_residual(mean, sigma_w=0.0, seed=1), mean
        )
        a = add_stochastic_residual(mean, sigma_w=2.0, phi=0.5, seed=7)
        b = add_stochastic_residual(mean, sigma_w=2.0, phi=0.5, seed=7)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
