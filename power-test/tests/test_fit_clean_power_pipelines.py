"""
Claim:
The clean composite fits dense and MoE surfaces on disjoint, deterministic
run populations using per-GPU targets and equal total weight per run. Repeated
legacy workloads never cross fitting and evaluation roles; all rate-4 runs
remain stress-only. Dense power consumes the physical GEMM, attention, and
memory work already computed by the timing roofline.

Plausible wrong implementations:
- Preserve the old third-repetition development split and leak workload identity.
- Leak rate-4 power into a fit.
- Let MoE targets change dense coefficients or dense targets change MoE coefficients.
- Recompute FLOPs from token counts and discard context-dependent attention.
- Fit node watts, making TP8 errors carry more weight than equal per-GPU errors.
- Give longer runs more influence than shorter runs.
- Apply a per-GPU cap after summing TP power, or omit node conversion entirely.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fit_clean_power_pipelines as clean  # noqa: E402
from fit_clean_power_pipelines import (  # noqa: E402
    fit_run_balanced,
    moe_compute_coordinate,
    moe_feature_basis,
    per_gpu,
    phase_work_utilization,
    predict_dense_node_power,
    resident_fraction,
    source_roles,
    training_run_sets,
)


def _row(model, family, rate, role="holdout_model"):
    return {
        "model": model,
        "hardware": "A100",
        "family": family,
        "tp": 4,
        "rate": rate,
        "legacy_role": role,
    }


def test_repeated_workloads_share_source_role_but_rate4_never_fits():
    metadata = {
        10: _row("gpt-oss-120b", "moe-120b", 1.0),
        11: _row("gpt-oss-120b", "moe-120b", 1.0),
        12: _row("gpt-oss-120b", "moe-120b", 1.0),
        20: _row("gpt-oss-120b", "moe-120b", 4.0),
        21: _row("gpt-oss-120b", "moe-120b", 4.0),
        22: _row("gpt-oss-120b", "moe-120b", 4.0),
    }
    roles = source_roles(metadata)
    assert [roles[run] for run in (10, 11, 12)] == [
        "train_source", "train_source", "train_source"
    ]
    assert {roles[run] for run in (20, 21, 22)} == {"stress_rate4"}


def test_twins_remain_transfer_at_every_rate():
    metadata = {
        1: _row("deepseek-r1-distill-70b", "dense-70b", 1.0, "holdout_twin"),
        2: _row("deepseek-r1-distill-70b", "dense-70b", 4.0, "holdout_twin"),
    }
    assert set(source_roles(metadata).values()) == {"transfer_twin"}


def test_dense_and_moe_training_sets_are_disjoint_and_target_blind():
    metadata = {
        1: _row("llama-3-405b", "dense-405b", 1.0),
        2: _row("llama-3-405b", "dense-405b", 1.0),
        3: _row("llama-3-405b", "dense-405b", 1.0),
        4: _row("gpt-oss-120b", "moe-120b", 1.0),
        5: _row("gpt-oss-120b", "moe-120b", 1.0),
        6: _row("gpt-oss-120b", "moe-120b", 1.0),
    }
    roles = source_roles(metadata)
    runs = training_run_sets(metadata, roles)
    assert runs == {"dense": {1, 2, 3}, "moe": {4, 5, 6}}


def test_phase_work_utilization_divides_physical_work_by_tp_capacity_once():
    sub = {
        "tp": np.asarray([2.0]),
        "prefill_gemm_flops_rate": np.asarray([234e12]),
        "decode_gemm_flops_rate": np.asarray([78e12]),
        "prefill_attn_flops_rate": np.asarray([78e12]),
        "decode_attn_flops_rate": np.asarray([78e12]),
        "w_read": np.asarray([2e12]),
        "prefill_attn_bytes_rate": np.asarray([0.25e12]),
        "decode_attn_bytes_rate": np.asarray([0.75e12]),
    }
    prefill, decode, memory = phase_work_utilization(sub, "A100")
    np.testing.assert_allclose(prefill, [0.5])
    np.testing.assert_allclose(decode, [0.25])
    np.testing.assert_allclose(memory, [0.75])


def test_per_gpu_conversion_divides_design_and_target_once():
    design, target = per_gpu(
        [[4.0, 8.0], [8.0, 16.0]], [400.0, 1600.0], [4.0, 8.0]
    )
    np.testing.assert_array_equal(design, [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_array_equal(target, [100.0, 200.0])


def test_resident_fraction_exposes_model_size_per_gpu():
    fraction = resident_fraction([16e9, 64e9], [2, 4], "A100")
    np.testing.assert_allclose(fraction, [0.1, 0.2])


def test_moe_basis_drops_link_floor_when_it_aliases_idle():
    design = np.arange(12.0).reshape(2, 6)
    reduced, names = moe_feature_basis(design, [4, 8])
    np.testing.assert_array_equal(reduced, design[:, [0, 2, 3, 4, 5]])
    assert "multi_gpu_floor" not in names


def test_moe_compute_uses_exact_work_not_token_count_proxy():
    sub = {
        "gemm_flops_rate": np.asarray([156e12, 156e12]),
        "attn_flops_rate": np.asarray([0.0, 156e12]),
        "tp": np.asarray([1.0, 1.0]),
        "busy": np.asarray([1.0, 1.0]),
    }

    coordinate = moe_compute_coordinate(sub)

    np.testing.assert_allclose(coordinate, [np.sqrt(0.5), 1.0])


def test_dense_artifact_does_not_clip_without_an_explicit_limit(monkeypatch):
    design = np.column_stack((
        [50.0, 250.0, 500.0], np.zeros((3, 3))
    ))
    monkeypatch.setattr(
        clean, "dense_design", lambda cache, hardware, delay_s: (
            design, np.ones(3, dtype=bool)
        )
    )
    cache = {
        "run_id": np.asarray([0, 0, 1]),
        "hw_names": np.asarray(["A100"]),
        "tp": np.asarray([2, 4, 4]),
    }
    artifact = {"dense": {"A100": {
        "feature_names": list(clean.DENSE_FEATURES),
        "coefficients": [1.0, 0.0, 0.0, 0.0],
        "delay_s": 0.0,
    }}}

    prediction = predict_dense_node_power(cache, artifact)

    np.testing.assert_array_equal(prediction, [100.0, 1000.0, 2000.0])


def test_run_balanced_fit_is_invariant_to_within_run_duplication():
    design = np.asarray([[1.0], [1.0], [1.0]])
    target = np.asarray([10.0, 20.0, 30.0])
    runs = np.asarray([0, 0, 1])
    original = fit_run_balanced(design, target, runs, {0, 1})
    duplicate = np.repeat([0, 1, 2], [4, 4, 1])
    repeated = fit_run_balanced(
        design[duplicate], target[duplicate], runs[duplicate], {0, 1}
    )
    np.testing.assert_allclose(repeated, original, rtol=1e-12, atol=1e-12)


def test_nontraining_targets_cannot_move_coefficients():
    design = np.asarray([[1.0], [2.0], [3.0]])
    runs = np.asarray([0, 0, 1])
    first = fit_run_balanced(design, [10.0, 20.0, 30.0], runs, {0})
    changed = fit_run_balanced(design, [10.0, 20.0, 1e9], runs, {0})
    np.testing.assert_array_equal(first, changed)
