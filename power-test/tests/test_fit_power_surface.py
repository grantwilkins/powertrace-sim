"""
Claim:
The shared hardware surface and its batch and thermal ablations are fit only on
finite dense training bins. Mixture-of-experts bins and development-test bins
are scored later but cannot change fitted parameters. The batch coordinate is
node-level, zero at batch zero, monotone, and concave.

Plausible wrong implementations:
- Fit every finite bin on the requested hardware.
- Exclude holdouts but accidentally include MoE training bins.
- Filter by model name rather than the declared architecture family.
- Use request rate instead of realized decode batch.
- Omit or double-apply TP scaling in the batch coordinate.
- Let duplicated active bins or checkpoint bytes move the loaded-idle floor.
- Let longer controlled-probe levels receive more total fitting weight.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fit_power_surface import (
    coefficients_in_design_order,
    fit_hardware,
    select_probe_candidate,
)
from batch_ablation import batch_feature, fit_batch_hardware
from thermal_ablation import fit_thermal_hardware


def _dataset(moe_power, test_power):
    n = 4
    return {
        "hw_idx": np.zeros(n, dtype=int),
        "hw_names": np.asarray(["A100"]),
        "role_idx": np.asarray([0, 0, 0, 1]),
        "role_names": np.asarray(["train", "test_indomain"]),
        "family_idx": np.asarray([0, 0, 1, 0]),
        "family_names": np.asarray(["dense", "moe_gated"]),
        "run_id": np.arange(n),
        "power": np.asarray([100.0, 140.0, moe_power, test_power]),
        "tp": np.ones(n),
        "busy": np.asarray([0.0, 1.0, 1.0, 1.0]),
        "prefill_duty": np.asarray([0.0, 1.0, 0.0, 0.0]),
        "decode_duty": np.asarray([0.0, 0.0, 1.0, 1.0]),
        "n_active": np.full(n, 1e9),
        "w_bytes": np.full(n, 2e9),
        "pre_tok": np.asarray([0.0, 100.0, 0.0, 0.0]),
        "dec_tok": np.asarray([0.0, 0.0, 100.0, 100.0]),
        "w_read": np.asarray([0.0, 1e9, 1e9, 1e9]),
        "kv_read": np.zeros(n),
        "kv_write": np.zeros(n),
        "engine_iterations_rate": np.asarray([0.0, 2.0, 10.0, 10.0]),
        "engine_tokens_per_iteration": np.asarray([0.0, 50.0, 10.0, 10.0]),
        "batch": np.asarray([0.0, 1.0, 4.0, 4.0]),
        "fp8": np.zeros(n),
        "dt_s": np.asarray(0.25),
    }


def test_moe_and_test_targets_cannot_change_dense_training_fit():
    fit_a, _, _, _ = fit_hardware(
        _dataset(300.0, 200.0), "A100", loaded_idle_w_per_gpu=60.0
    )
    fit_b, _, _, _ = fit_hardware(
        _dataset(3000.0, 2000.0), "A100", loaded_idle_w_per_gpu=60.0
    )
    assert fit_a["n_train_bins"] == 2
    assert fit_a["n_moe_train_bins_excluded"] == 1
    assert fit_a["delay_s"] == fit_b["delay_s"]
    assert fit_a["coefficients"] == fit_b["coefficients"]
    assert fit_a["coefficients"]["tp"] == 60.0
    assert fit_a["coefficients"]["tp_link"] == 0.0
    assert fit_a["coefficients"]["resident_weights"] == 0.0


def test_probe_fit_is_invariant_to_duplicate_bins_within_a_level():
    source = _dataset(300.0, 200.0)
    keys = (
        "tp", "busy", "n_active", "w_bytes", "pre_tok", "dec_tok",
        "w_read", "kv_read", "kv_write", "engine_iterations_rate", "fp8",
    )
    probe = {key: np.asarray(source[key][:2]) for key in keys}
    probe |= {
        "hardware": np.asarray(["A100", "A100"]),
        "level_id": np.asarray(["prefill", "prefill"]),
        "power": np.asarray([110.0, 150.0]),
    }
    duplicated = {key: np.repeat(value, 3) for key, value in probe.items()}

    fit_a, _, _, _ = fit_hardware(
        source, "A100", loaded_idle_w_per_gpu=60.0,
        probe_calibration=probe,
    )
    fit_b, _, _, _ = fit_hardware(
        source, "A100", loaded_idle_w_per_gpu=60.0,
        probe_calibration=duplicated,
    )

    assert fit_a["n_probe_levels"] == fit_b["n_probe_levels"] == 1
    for name in fit_a["coefficients"]:
        assert np.isclose(
            fit_a["coefficients"][name], fit_b["coefficients"][name]
        )


def test_probe_candidate_cannot_trade_nrmse_for_energy():
    baseline = {
        "energy_error_pct": 4.0,
        "acf_mae": 0.03,
        "acf_r2": 0.90,
        "nrmse_range": 0.10,
    }
    tradeoff = {**baseline, "energy_error_pct": 3.0, "nrmse_range": 0.11}
    pareto = {**baseline, "energy_error_pct": 3.0, "acf_r2": 0.91}

    assert not select_probe_candidate(baseline, tradeoff)
    assert select_probe_candidate(baseline, pareto)


def test_coefficients_follow_design_names_not_json_key_order():
    fit = {"coefficients": {"memory": 3.0, "idle": 1.0, "compute": 2.0}}
    np.testing.assert_array_equal(
        coefficients_in_design_order(fit, ["idle", "compute", "memory"]),
        [1.0, 2.0, 3.0],
    )


def test_thermal_fit_cannot_use_moe_or_test_targets():
    _, fit_a = fit_thermal_hardware(
        _dataset(300.0, 200.0), "A100", "all_load"
    )
    _, fit_b = fit_thermal_hardware(
        _dataset(3000.0, 2000.0), "A100", "all_load"
    )

    assert fit_a == fit_b


def test_batch_feature_is_node_scaled_monotone_and_concave():
    feature = batch_feature({
        "batch": np.asarray([0.0, 1.0, 2.0]),
        "tp": np.asarray([4.0, 4.0, 4.0]),
    })

    np.testing.assert_allclose(feature, 4.0 * np.log([1.0, 2.0, 3.0]))
    assert feature[1] - feature[0] > feature[2] - feature[1]


def test_batch_fit_cannot_use_moe_or_test_targets():
    _, fit_a = fit_batch_hardware(
        _dataset(300.0, 200.0), "A100"
    )
    _, fit_b = fit_batch_hardware(
        _dataset(3000.0, 2000.0), "A100"
    )

    assert fit_a == fit_b
