"""
Claim:
The MoE surface fits TP-summed node power from timing-ledger features with
equal run weighting, exact support limits, and no dependence on non-training
power targets.

Plausible wrong implementations:
- Treat per-GPU features as node features and multiply or divide by TP twice.
- Give long runs more fit weight than short runs.
- Let development or holdout targets change the fitted artifact.
- Extrapolate the 20B TP1/2 surface to dense, 120B, TP4/8, or measured routing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_surface_core import (  # noqa: E402
    BIN_S,
    DELAY_S,
    FEATURE_NAMES,
    HARDWARE,
    HBM_BYTES_S,
    MODEL,
    ROUTING_MODE,
    SCHEMA,
    SUPPORTED_TP,
    TDP_W_PER_GPU,
    canonical_design_digest,
    fit_coefficients,
    lag_one_bin_by_run,
    predict,
    supported_run,
    surface_design,
    target_digest,
    validate_artifact_contract,
    validate_cache,
)


def _cache(**overrides):
    cache = {
        "run_id": np.asarray([0]),
        "tp": np.asarray([2.0]),
        "w_read": np.asarray([0.5e12]),
        "kv_read": np.asarray([0.25e12]),
        "kv_write": np.asarray([0.25e12]),
        "engine_iterations_rate": np.asarray([100.0]),
        "batch": np.asarray([3.0]),
    }
    cache.update(overrides)
    return cache


def test_hand_computed_node_feature_row_and_cap():
    np.testing.assert_allclose(
        surface_design(_cache())[0],
        [2.0, 2.0, 0.5, 0.2, 2.0 * np.log(4.0)],
    )
    design = np.asarray([[2.0, 0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        predict(design, np.asarray([1000.0, 0.0, 0.0, 0.0, 0.0]), [2.0]),
        [800.0],
    )


def test_memory_alignment_is_causal_and_resets_at_run_boundaries():
    np.testing.assert_array_equal(
        lag_one_bin_by_run([10.0, 20.0, 30.0, 40.0], [1, 1, 2, 2]),
        [10.0, 10.0, 30.0, 30.0],
    )


def test_fit_weights_runs_equally_under_within_run_duplication():
    design = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ])
    power = np.asarray([10.0, 20.0, 30.0])
    runs = np.asarray([0, 0, 1])
    original = fit_coefficients(design, power, runs, {0, 1})

    duplicate = np.repeat([0, 1, 2], [3, 3, 1])
    repeated = fit_coefficients(
        design[duplicate], power[duplicate], runs[duplicate], {0, 1})
    np.testing.assert_allclose(repeated, original, rtol=1e-12, atol=1e-12)


def test_nontraining_targets_cannot_change_coefficients():
    design = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ])
    runs = np.asarray([0, 0, 1])
    first = fit_coefficients(design, [10.0, 20.0, 30.0], runs, {0})
    changed = fit_coefficients(design, [10.0, 20.0, 1e9], runs, {0})
    np.testing.assert_array_equal(changed, first)


def test_support_is_exact_and_dense_never_matches():
    assert supported_run("gpt-oss-20b", "A100", 1, "uniform")
    assert supported_run("gpt-oss-20b", "A100", 2, "uniform")
    assert not supported_run("gpt-oss-20b", "A100", 4, "uniform")
    assert not supported_run("gpt-oss-20b", "A100", 1.5, "uniform")
    assert not supported_run("gpt-oss-20b", "A100", 2, "measured")
    assert not supported_run("gpt-oss-120b", "A100", 4, "uniform")
    assert not supported_run("llama-3-70b", "A100", 2, "uniform")


def _provenance_cache():
    return {
        "run_id": np.asarray([1, 2]),
        "tp": np.asarray([1.0, 2.0]),
        "rate": np.asarray([0.5, 4.0]),
        "w_read": np.asarray([1e11, 2e11]),
        "kv_read": np.asarray([3e10, 4e10]),
        "kv_write": np.asarray([5e10, 6e10]),
        "engine_iterations_rate": np.asarray([100.0, 200.0]),
        "batch": np.asarray([2.0, 4.0]),
        "power": np.asarray([100.0, 200.0]),
        "power_valid": np.asarray([True, True]),
        "model_idx": np.asarray([0, 0]),
        "model_names": np.asarray([MODEL]),
        "hw_idx": np.asarray([0, 0]),
        "hw_names": np.asarray([HARDWARE]),
        "role_idx": np.asarray([0, 1]),
        "role_names": np.asarray(["train", "holdout_rate"]),
        "dt_s": np.asarray(0.25),
        "moe_routing_mode": np.asarray(ROUTING_MODE),
    }


def test_provenance_digests_bind_only_selected_design_and_targets():
    cache = _provenance_cache()
    design = canonical_design_digest(cache, {1})
    target = target_digest(cache, {1})

    changed_other = {key: value.copy() for key, value in cache.items()}
    changed_other["batch"][1] = 1e9
    changed_other["power"][1] = 1e9
    assert canonical_design_digest(changed_other, {1}) == design
    assert target_digest(changed_other, {1}) == target

    changed_design = {key: value.copy() for key, value in cache.items()}
    changed_design["batch"][0] += 1
    assert canonical_design_digest(changed_design, {1}) != design

    changed_target = {key: value.copy() for key, value in cache.items()}
    changed_target["power"][0] += 1
    assert target_digest(changed_target, {1}) != target


def test_cache_validation_rejects_fractional_tp_and_moe_role_drift():
    cache = _provenance_cache()
    index = {
        1: {"config_id": "gpt-oss-20b_A100_tp1"},
        2: {"config_id": "gpt-oss-20b_A100_tp2"},
    }
    splits = {1: "train", 2: "holdout_rate"}
    validate_cache(cache, index, splits)

    fractional = {key: value.copy() for key, value in cache.items()}
    fractional["tp"][0] = 1.5
    with pytest.raises(ValueError, match="positive integers"):
        validate_cache(fractional, index, splits)

    with pytest.raises(ValueError, match="split manifest"):
        validate_cache(cache, index, {1: "holdout_rate", 2: "holdout_rate"})


def test_artifact_contract_rejects_support_drift_and_invalid_coefficients():
    artifact = {
        "schema_version": SCHEMA,
        "model": MODEL,
        "hardware": HARDWARE,
        "supported_tp": list(SUPPORTED_TP),
        "routing_mode": ROUTING_MODE,
        "target_units": "TP-summed node watts",
        "feature_names": list(FEATURE_NAMES),
        "response_delay_s": DELAY_S,
        "input_bin_s": BIN_S,
        "hbm_bytes_s": HBM_BYTES_S,
        "tdp_w_per_gpu": TDP_W_PER_GPU,
        "fit_role": "train",
        "coefficients": [1.0] * len(FEATURE_NAMES),
    }
    np.testing.assert_array_equal(
        validate_artifact_contract(artifact), artifact["coefficients"])

    changed_support = dict(artifact, supported_tp=[1, 2, 4])
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_artifact_contract(changed_support)

    changed_coefficients = dict(
        artifact, coefficients=[1.0] * (len(FEATURE_NAMES) - 1) + [np.nan])
    with pytest.raises(ValueError, match="coefficients"):
        validate_artifact_contract(changed_coefficients)
