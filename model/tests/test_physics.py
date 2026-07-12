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
- Use all-weight rather than decode-weight traffic in M0d.
- Leak filter history across run boundaries.
- Deploy a target-routed or over-budget selected artifact.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.classifiers.physics import (
    FEATURE_ORDER,
    SCHEMA_VERSION,
    SELECTED_SCHEMA_VERSION,
    add_stochastic_residual,
    apply_meter_lag,
    ledger_power_features,
    load_physics_artifact,
    load_selected_physics_artifact,
    physics_design,
    physics_feature_order,
    predict_mean_node_power,
    predict_selected_physics,
)


def _ledger():
    return {
        "pre_tok": np.asarray([0.0, 2.0]),
        "dec_tok": np.asarray([0.0, 3.0]),
        "w_read": np.asarray([0.0, 10.0]),
        "w_read_dec": np.asarray([0.0, 3.0]),
        "kv_read": np.asarray([0.0, 6.0]),
        "w_read_pre": np.asarray([0.0, 7.0]),
        "kv_write": np.asarray([0.0, 5.0]),
        "comm": np.asarray([0.0, 4.0]),
    }


def _selected_artifact(mean_kind="M0", *, residence=False, timing="arrival_only_validated"):
    filters = (
        [{"name": "A_fast", "ema_alpha": 0.03},
         {"name": "A_slow", "ema_alpha": 0.5}]
        if mean_kind == "M4A" else []
    )
    names = physics_feature_order(
        mean_kind, residence=residence,
        state_filter_names=tuple(spec["name"] for spec in filters),
    )
    return {
        "schema_version": SELECTED_SCHEMA_VERSION,
        "hardware": "H100",
        "dt_s": 1.0,
        "timing_contract": timing,
        "architectures": {},
        "hardware_profile": {
            "hbm_bandwidth_bytes_s": 10.0,
            "compute_peak_flops_s": 100.0,
            "link_bandwidth_bytes_s": 20.0,
            "residence_bytes_per_gpu": 10.0,
        },
        "mode": {
            "candidate": mean_kind,
            "mean_kind": mean_kind,
            "residence": residence,
            "coefficients": {name: 0.0 for name in names},
            "lag": {"moving_average_s": 1.0, "ema_alpha": 1.0},
            "state_filters": filters,
            "cap_w_per_gpu": 1000.0,
            "cap_quantile": 0.995,
        },
        "learned_scalar_count": len(names) + 2 + len(filters),
        "provenance": {
            "training_source_ids": ["train"],
            "selection_source_ids": ["development"],
            "excluded_target_source_ids": ["target"],
        },
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

    def test_m0d_residence_design_is_hand_worked(self):
        ledger = {
            "pre_tok": [1.0], "dec_tok": [2.0], "w_read": [90.0],
            "w_read_dec": [30.0], "kv_read": [10.0], "w_read_pre": [7.0],
            "kv_write": [10.0], "comm": [8.0],
        }
        design, names = physics_design(
            ledger, {"n_active": 10.0, "w_bytes": 40.0, "fp8": 0},
            tp=2, hardware_profile=_selected_artifact()["hardware_profile"],
            mean_kind="M0d", residence=True, ema_alpha=1.0,
        )
        values = dict(zip(names, design[0]))
        self.assertEqual(values["tp"], 2.0)
        self.assertEqual(values["tp_link"], 2.0)
        self.assertEqual(values["busy_tp"], 2.0)
        self.assertAlmostEqual(values["sat_bw_a"], 2.0 * (1.0 - np.exp(-2.5 / 0.05)))
        self.assertEqual(values["flops_pre"], 20.0)
        self.assertEqual(values["flops_dec"], 40.0)
        self.assertEqual(values["w_read_pre"], 7.0)
        self.assertEqual(values["kv_write"], 10.0)
        self.assertEqual(values["comm"], 8.0)
        self.assertEqual(values["residence"], 2.0)

    def test_monotone_segments_are_exact_at_the_first_knot(self):
        ledger = {
            "pre_tok": np.asarray([4.9, 5.0, 5.1]), "dec_tok": np.zeros(3),
            "w_read": np.zeros(3), "kv_read": np.zeros(3),
            "w_read_pre": np.zeros(3), "kv_write": np.zeros(3),
            "comm": np.zeros(3),
        }
        design, names = physics_design(
            ledger, {"n_active": 0.5, "w_bytes": 1.0, "fp8": 0}, tp=1,
            hardware_profile=_selected_artifact()["hardware_profile"],
            mean_kind="M0b", ema_alpha=1.0,
        )
        first = design[:, names.index("compute_0_0.05")]
        second = design[:, names.index("compute_0.05_0.15")]
        np.testing.assert_allclose(first, [0.049, 0.05, 0.05])
        np.testing.assert_allclose(second, [0.0, 0.0, 0.001])

    def test_m4a_filters_reset_at_each_run(self):
        ledger = {
            "pre_tok": np.zeros(4), "dec_tok": np.zeros(4),
            "w_read": np.zeros(4), "kv_read": np.zeros(4),
            "w_read_pre": np.zeros(4), "kv_write": np.zeros(4),
            "comm": np.zeros(4), "A_t": np.asarray([0.0, 1.0, 0.0, 1.0]),
        }
        filters = (
            {"name": "A_fast", "ema_alpha": 0.03},
            {"name": "A_slow", "ema_alpha": 0.5},
        )
        design, names = physics_design(
            ledger, {"n_active": 1.0, "w_bytes": 1.0, "fp8": 0}, tp=1,
            hardware_profile=_selected_artifact()["hardware_profile"],
            mean_kind="M4A", run_ids=[0, 0, 1, 1], state_filters=filters,
        )
        np.testing.assert_allclose(
            design[:, names.index("A_fast")], [0.0, 0.03 * np.log(2), 0.0, 0.03 * np.log(2)]
        )
        np.testing.assert_allclose(
            design[:, names.index("A_slow")], [0.0, 0.5 * np.log(2), 0.0, 0.5 * np.log(2)]
        )

    def test_selected_prediction_enforces_hardware_and_node_cap(self):
        artifact = _selected_artifact()
        artifact["mode"]["coefficients"]["tp"] = 10.0
        artifact["mode"]["coefficients"]["busy_tp"] = 40.0
        artifact["mode"]["cap_w_per_gpu"] = 45.0
        ledger = _ledger()
        out = predict_selected_physics(
            ledger, {"n_active": 10.0, "w_bytes": 40.0, "fp8": 0},
            tp=2, hardware="H100", artifact=artifact,
        )
        np.testing.assert_array_equal(out, [20.0, 90.0])
        with self.assertRaisesRegex(ValueError, "not 'A100'"):
            predict_selected_physics(
                ledger, {"n_active": 10.0, "w_bytes": 40.0}, tp=2,
                hardware="A100", artifact=artifact,
            )

    def test_selected_loader_rejects_routing_budget_mismatch_and_leakage(self):
        cases = []
        routed = _selected_artifact()
        routed["modes_by_tp"] = {"8": "M0"}
        cases.append((routed, "routing"))
        over_budget = _selected_artifact()
        over_budget["learned_scalar_count"] = 81
        cases.append((over_budget, "exceeds 80"))
        mismatch = _selected_artifact()
        mismatch["learned_scalar_count"] += 1
        cases.append((mismatch, "count mismatch"))
        leaking = _selected_artifact()
        leaking["provenance"]["training_source_ids"] = ["target"]
        cases.append((leaking, "leaks target"))
        with tempfile.TemporaryDirectory() as td:
            for index, (artifact, message) in enumerate(cases):
                path = Path(td) / f"bad-{index}.json"
                path.write_text(json.dumps(artifact))
                with self.assertRaisesRegex(ValueError, message):
                    load_selected_physics_artifact(path)


class TestM0cMean(unittest.TestCase):
    """Claim: M0c is a concave saturating-ramp response in compute and memory
    utilization with no communication column, an occupancy term that is active
    at idle, and a dtype scale that halves only the declared FP8 fraction.

    Plausible wrong implementations:
    - Reuse the hinge-segment basis, allowing increasing marginal power.
    - Keep a communication column that NNLS exchanges with compute.
    - Apply the blanket 0.5 FP8 scale when a recipe fraction is declared.
    """

    def _profile(self):
        return {
            "hbm_bandwidth_bytes_s": 10.0,
            "compute_peak_flops_s": 100.0,
            "link_bandwidth_bytes_s": 20.0,
            "residence_bytes_per_gpu": 10.0,
        }

    def _m0c_ledger(self):
        return {
            "pre_tok": np.asarray([0.0, 0.0, 8.0]),
            "dec_tok": np.asarray([0.0, 4.0, 0.0]),
            "w_read": np.asarray([0.0, 6.0, 2.0]),
            "w_read_dec": np.asarray([0.0, 6.0, 0.0]),
            "kv_read": np.asarray([0.0, 3.0, 0.0]),
            "w_read_pre": np.asarray([0.0, 0.0, 2.0]),
            "kv_write": np.asarray([0.0, 1.0, 2.0]),
            "comm": np.asarray([0.0, 5.0, 5.0]),
        }

    def test_m0c_design_is_hand_worked(self):
        design, names = physics_design(
            self._m0c_ledger(), {"n_active": 10.0, "w_bytes": 5.0, "fp8": 0},
            tp=2, hardware_profile=self._profile(), mean_kind="M0c",
            residence=True, dt_s=1.0, moving_average_s=1.0, ema_alpha=1.0,
        )
        self.assertEqual(names, (
            "tp", "tp_link", "busy_tp",
            "compute_ramp_0.05", "compute_ramp_0.15", "compute_ramp_0.4",
            "compute_ramp_1", "compute_ramp_1.5",
            "memory_ramp_0.05", "memory_ramp_0.15", "memory_ramp_0.4",
            "memory_ramp_1", "memory_ramp_1.5", "residence",
        ))
        self.assertFalse(any("communication" in name for name in names))
        # u_compute = 2*10*(pre+dec)/(2*100) -> (0, 0.4, 0.8)
        # u_memory = (w_read+kv_read+kv_write)/(2*10) -> (0, 0.5, 0.2)
        np.testing.assert_allclose(design[1, 3:8], 2 * np.asarray([0.05, 0.15, 0.4, 0.4, 0.4]))
        np.testing.assert_allclose(design[2, 3:8], 2 * np.asarray([0.05, 0.15, 0.4, 0.8, 0.8]))
        np.testing.assert_allclose(design[1, 8:13], 2 * np.asarray([0.05, 0.15, 0.4, 0.5, 0.5]))
        np.testing.assert_allclose(design[2, 8:13], 2 * np.asarray([0.05, 0.15, 0.2, 0.2, 0.2]))
        np.testing.assert_allclose(design[:, 2], [0.0, 2.0, 2.0])
        # Occupancy 5/(2*10) = 0.25, busy-gated like the other means.
        np.testing.assert_allclose(design[:, 13], [0.0, 0.5, 0.5])

    def test_m0c_response_is_concave_and_monotone_for_any_nonnegative_fit(self):
        rng = np.random.default_rng(20260711)
        knots = np.asarray([0.05, 0.15, 0.4, 1.0, 1.5])
        for _ in range(25):
            weights = rng.uniform(0.0, 100.0, knots.size)
            grid = np.linspace(0.0, 1.5, 601)
            response = (weights * np.minimum(grid[:, None], knots)).sum(axis=1)
            slopes = np.diff(response) / np.diff(grid)
            self.assertTrue(np.all(slopes >= -1e-9))
            self.assertTrue(np.all(np.diff(slopes) <= 1e-9))

    def test_fp8_flop_frac_scales_only_the_declared_fraction(self):
        ledger = self._m0c_ledger()
        ledger["pre_tok"] = np.asarray([0.0, 0.0, 0.2])
        ledger["dec_tok"] = np.asarray([0.0, 0.1, 0.0])
        blanket, _ = physics_design(
            ledger, {"n_active": 10.0, "w_bytes": 5.0, "fp8": 1},
            tp=2, hardware_profile=self._profile(), mean_kind="M0c",
            residence=False, dt_s=1.0, moving_average_s=1.0, ema_alpha=1.0,
        )
        recipe, _ = physics_design(
            ledger, {"n_active": 10.0, "w_bytes": 5.0, "fp8": 1, "fp8_flop_frac": 0.8},
            tp=2, hardware_profile=self._profile(), mean_kind="M0c",
            residence=False, dt_s=1.0, moving_average_s=1.0, ema_alpha=1.0,
        )
        # Utilizations stay below the first knot, so ramp columns are linear
        # in the dtype scale: (1 - 0.5*0.8) / 0.5 = 1.2.
        np.testing.assert_allclose(recipe[1:, 3:8], blanket[1:, 3:8] * 1.2)
        np.testing.assert_allclose(recipe[:, 8:13], blanket[:, 8:13])

    def test_selected_loader_accepts_m0c_and_rejects_state_filters(self):
        artifact = _selected_artifact("M0c", residence=True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m0c.json"
            path.write_text(json.dumps(artifact))
            loaded = load_selected_physics_artifact(path)
            self.assertEqual(loaded["mode"]["mean_kind"], "M0c")
            broken = json.loads(json.dumps(artifact))
            broken["mode"]["state_filters"] = [{"name": "A_slow", "ema_alpha": 0.5}]
            path.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                load_selected_physics_artifact(path)


if __name__ == "__main__":
    unittest.main()
