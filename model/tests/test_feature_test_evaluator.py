"""
Claim:
The feature evaluator assigns repeats deterministically, reports exact per-run
one-second energy/ACF metrics, resets causal state, and never fits target power.

Plausible wrong implementations:
- Assign repeats in manifest order instead of sorted source identity.
- Pool bins across runs or carry a lagged value across a run boundary.
- Compute energy before four-bin aggregation or ACF over the wrong lag range.
- Read target labels while fitting a transfer artifact.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[2] / "feature-test" / "evaluate_candidates.py"
SPEC = importlib.util.spec_from_file_location("feature_evaluator", PATH)
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def test_repeat_assignment_is_sorted_within_cell():
    rows = [
        {"run_id": 7, "config_id": "c", "rate": 1.0, "source_id": "z"},
        {"run_id": 3, "config_id": "c", "rate": 1.0, "source_id": "a"},
        {"run_id": 5, "config_id": "c", "rate": 1.0, "source_id": "m"},
    ]
    assert ev.assign_repeats(rows) == {3: 0, 5: 1, 7: 2}


def test_trace_metrics_aggregate_four_bins_and_use_sixty_lags():
    one_s = 200 + 30 * np.sin(np.arange(80) / 4)
    measured = np.repeat(one_s, 4)
    predicted = measured * 1.1
    out = ev.trace_metrics(measured, predicted)
    assert np.isclose(out["energy_error_pct"], 10.0)
    assert np.isclose(out["acf_r2"], 1.0)
    assert np.isclose(out["acf_mae"], 0.0)
    assert np.isclose(out["nrmse_mean"], np.sqrt(np.mean((0.1 * one_s) ** 2)) / one_s.mean())


def _tiny_data():
    n = 80
    run_id = np.repeat([0, 1, 2], n)
    x = np.tile(np.linspace(0, 2, n), 3)
    return {"run_id": run_id, "power": 100 + 4 * x, "tp": np.ones(3*n), "hw_idx": np.zeros(3*n),
            "pre_tok": x, "dec_tok": x / 2, "batch": x > 0, "pre_active": x > 1,
            "n_active": np.ones(3*n), "fp8": np.zeros(3*n), "w_read": x, "kv_read": x,
            "w_read_pre": x, "w_read_dec": x, "kv_write": x, "comm": x,
            "w_bytes": np.ones(3*n), "A_t": x,
            "delta_A_t": np.tile(np.r_[0.0, np.diff(np.linspace(0, 2, n))], 3),
            "dt_s": np.asarray(0.25)}


def test_causal_history_resets_at_run_boundary():
    original = _tiny_data()
    changed = _tiny_data()
    changed["A_t"][:80] = 1e6
    changed["delta_A_t"] = np.concatenate([
        np.r_[0.0, np.diff(changed["A_t"][start:start + 80])]
        for start in (0, 80, 160)
    ])
    scale = (np.zeros(2), np.ones(2))
    first, _, _ = ev.causal_design(original, "M1", set(), scale=scale)
    second, _, _ = ev.causal_design(changed, "M1", set(), scale=scale)
    np.testing.assert_array_equal(first[80:160], second[80:160])


def test_causal_design_rejects_noncontiguous_runs():
    d = _tiny_data()
    d["run_id"] = np.tile([0, 1, 0], 80)
    with pytest.raises(ValueError, match="contiguous"):
        ev.causal_design(d, "M1", set(), scale=(np.zeros(2), np.ones(2)))


def test_unidentifiable_link_standing_power_is_pruned():
    d = _tiny_data()
    d["tp"][:] = 2.0
    fit = ev._physics_fit(
        d, {"refit": [0, 1]}, kind="M0", alpha=1.0, hardware="A100"
    )
    assert fit["physics"][0] > 0.0
    assert fit["physics"][1] == 0.0


def test_decode_roofline_does_not_count_prefill_weights_as_decode_bandwidth():
    d = _tiny_data()
    changed = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}
    changed["w_read"] += 1e6
    original = ev._physical_design(d, "M0d", 1.0, "A100")
    perturbed = ev._physical_design(changed, "M0d", 1.0, "A100")
    np.testing.assert_array_equal(original[:, 3:6], perturbed[:, 3:6])


def test_target_mutation_cannot_change_source_selected_artifact():
    d = _tiny_data()
    split = {"refit": [0, 1], "test": [2]}
    scores = {name: {"energy_error_pct_median": 4.0, "energy_error_pct_p90": 6.0,
                     "energy_error_pct_worst": 8.0, "acf_r2_median": 0.9,
                     "acf_mae_p90": 0.05, "nrmse_range_median": 0.1}
              for name in ev.PREFERENCE}
    selected = ev.choose_source_candidate(scores, "S0_A100")
    first = ev._physics_fit(d, split, kind=selected, alpha=0.55, hardware="A100")
    first["training_source_ids"] = ["source-a", "source-b"]
    changed = {k: np.copy(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}
    changed["power"][changed["run_id"] == 2] *= 1000
    selected_again = ev.choose_source_candidate(scores, "S0_A100")
    second = ev._physics_fit(changed, split, kind=selected_again, alpha=0.55, hardware="A100")
    second["training_source_ids"] = ["source-a", "source-b"]
    assert selected == selected_again
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_equal(first[key], second[key])


def test_correction_must_improve_dynamics_without_trading_away_mean():
    physics = {"energy_error_pct_median": 4.0, "energy_error_pct_p90": 8.0,
               "acf_r2_median": 0.80, "acf_mae_median": 0.10,
               "nrmse_range_median": 0.14, "cap_hit_fraction_median": 0.01}
    safe = {**physics, "energy_error_pct_median": 4.4, "energy_error_pct_p90": 8.8,
            "acf_r2_median": 0.86, "nrmse_range_median": 0.13}
    unsafe = {**safe, "energy_error_pct_median": 4.6}
    assert ev.passes_correction_safety(safe, physics)
    assert not ev.passes_correction_safety(unsafe, physics)


def test_b2_comparison_enforces_each_declared_tolerance():
    baseline = {"energy_error_pct_median": 0.5, "acf_r2_median": 0.95,
                "nrmse_range_median": 0.08}
    passing = {"energy_error_pct_median": 1.5, "acf_r2_median": 0.90,
               "nrmse_range_median": 0.10}
    assert ev.passes_b2_comparison(passing, baseline)
    assert not ev.passes_b2_comparison({**passing, "energy_error_pct_median": 1.501}, baseline)


def _patch_meter_kernel():
    ev._METER_KERNEL_CACHE = {
        "A100": {"moving_average_s": 0.25, "ema_alpha": 1.0},
        "H100": {"moving_average_s": 1.0, "ema_alpha": 1.0},
    }


def _m0c_data():
    """Two identical runs with idle, decode-only, and mixed-prefill phases.

    Ground truth per GPU: idle 60 W, memory response 500*min(u_mem, 0.4),
    compute response 600*min(u_comp, 0.4), all below the 400 W/GPU cap.
    Decode bins carry a negligible compute utilization so the phase-staged
    fit is exactly identifiable.
    """
    n = 200
    u_mem = np.zeros(n)
    u_comp = np.zeros(n)
    dec = np.zeros(n)
    pre = np.zeros(n)
    u_mem[50:125] = np.linspace(0.01, 0.3, 75)
    dec[50:125] = 1e-9
    u_mem[125:] = 0.1
    u_comp[125:] = np.linspace(0.02, 0.5, 75)
    dec[125:] = 1e-9
    tp = 2.0
    # u_comp = 2 * n_active * (pre+dec) / (tp * 312e12) with n_active = 156e12,
    # so pre = tp * u_comp.
    pre[125:] = tp * u_comp[125:]
    w_read = u_mem * tp * 2e12
    power = tp * (60.0
                  + 500.0 * np.minimum(u_mem, 0.4)
                  + 600.0 * np.minimum(u_comp, 0.4))
    power[:50] = tp * 60.0
    ones = np.ones(2 * n)
    zeros = np.zeros(2 * n)
    return {
        "run_id": np.repeat([0, 1], n), "power": np.tile(power, 2),
        "tp": tp * ones, "hw_idx": zeros,
        "pre_tok": np.tile(pre, 2), "dec_tok": np.tile(dec, 2),
        "n_active": 156e12 * ones, "fp8": zeros,
        "w_read": np.tile(w_read, 2), "kv_read": zeros, "w_read_pre": zeros,
        "w_read_dec": np.tile(w_read, 2), "kv_write": zeros, "comm": zeros,
        "w_bytes": ones, "A_t": zeros, "delta_A_t": zeros,
        "dt_s": np.asarray(0.25),
    }


def test_m0c_phase_fit_recovers_separated_compute_and_memory_responses():
    _patch_meter_kernel()
    d = _m0c_data()
    fit = ev._physics_fit(d, {"refit": [0, 1]}, kind="M0c", alpha=1.0,
                          hardware="A100", residence=False, cap_quantile="tdp")
    names = ev.physics_feature_order("M0c", residence=False)
    knots = np.asarray([0.05, 0.15, 0.4, 1.0, 1.5])
    compute = np.asarray([n.startswith("compute_ramp") for n in names])
    memory = np.asarray([n.startswith("memory_ramp") for n in names])
    for u, expected in ((0.10, 50.0), (0.28, 140.0)):
        recovered = float(np.minimum(u, knots) @ fit["physics"][memory])
        np.testing.assert_allclose(recovered, expected, rtol=1e-6)
    for u, expected in ((0.10, 60.0), (0.35, 210.0)):
        recovered = float(np.minimum(u, knots) @ fit["physics"][compute])
        np.testing.assert_allclose(recovered, expected, rtol=1e-6)
    np.testing.assert_allclose(fit["physics"][0], 60.0, rtol=1e-6)
    pred = ev._predict(d, {**fit, "candidate": "M0c"})
    np.testing.assert_allclose(pred, d["power"], rtol=1e-6)


def test_m0c_cap_is_the_hardware_power_limit_not_a_training_quantile():
    _patch_meter_kernel()
    d = _m0c_data()
    fit = ev._physics_fit(d, {"refit": [0, 1]}, kind="M0c", alpha=1.0,
                          hardware="A100", residence=False, cap_quantile="tdp")
    scaled = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    scaled["power"] = scaled["power"] * 10.0
    refit = ev._physics_fit(scaled, {"refit": [0, 1]}, kind="M0c", alpha=1.0,
                            hardware="A100", residence=False, cap_quantile="tdp")
    assert fit["cap_quantile"] == "tdp"
    assert fit["cap_w_per_gpu"] == 400.0
    assert refit["cap_w_per_gpu"] == 400.0
    h100 = ev._physics_fit(d, {"refit": [0, 1]}, kind="M0c", alpha=1.0,
                           hardware="H100", residence=False, cap_quantile="tdp")
    assert h100["cap_w_per_gpu"] == 700.0


def test_prefill_influence_covers_meter_window_and_resets_at_run_boundary():
    pre = np.zeros(20)
    pre[2] = 5.0
    run_ids = np.repeat([0, 1], 10)
    influence = ev._prefill_influence(pre, run_ids, 4)
    expected = np.zeros(20, dtype=bool)
    expected[2:6] = True
    np.testing.assert_array_equal(influence, expected)
    # A prefill in the last bin of run 0 must not leak into run 1.
    pre2 = np.zeros(20)
    pre2[9] = 5.0
    influence2 = ev._prefill_influence(pre2, run_ids, 4)
    assert influence2[9]
    assert not influence2[10:].any()


def test_scalar_count_includes_m4_state_alphas_and_normalization():
    fit = {"candidate": "M3", "mean_kind": "M4A", "physics": np.ones(21),
           "residual": np.ones(9), "scale": (np.ones(2), np.ones(2)),
           "cap_w_per_gpu": 500.0, "lag_alpha": 0.5}
    assert ev.model_scalar_count(fit) == 21 + 9 + 4 + 1 + 1 + 2
