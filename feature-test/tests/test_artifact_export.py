"""Claim: export produces one strict, routing-free artifact for one hardware."""

import numpy as np

from artifact_export import selected_physics_artifact
from model.classifiers.physics import (
    _validate_selected_artifact,
    physics_design,
    predict_selected_physics,
)
from model.training_data.arch import get_arch


def test_selected_export_is_hardware_local_and_named():
    fit = {
        "physics": np.arange(11, dtype=float),
        "lag_alpha": 0.6,
        "cap_w_per_gpu": 300.0,
        "training_source_ids": ["source-0", "source-1"],
    }
    rows = [{"hardware": "A100", "model": "gpt-oss-20b", "tp": 2}]
    artifact = selected_physics_artifact(
        "A100", "M0", fit, rows, selection_source_ids=["selection"],
        excluded_target_source_ids=["target"],
    )
    validated = _validate_selected_artifact(artifact)
    assert validated["hardware"] == "A100"
    assert validated["learned_scalar_count"] == 13
    assert "selected_by_split" not in validated
    assert validated["parallelism_support"] == {"kind": "TP_only", "calibrated_tp": [2]}
    assert validated["provenance"]["role"] == "production_refit_not_transfer_evidence"
    assert list(validated["mode"]["coefficients"])[0] == "tp"


def test_exported_kernel_matches_shared_design_with_run_resets():
    run_ids = np.repeat([0, 1], 4)
    ledger = {
        "pre_tok": np.tile([0.0, 2.0, 0.0, 1.0], 2),
        "dec_tok": np.tile([0.0, 1.0, 3.0, 0.0], 2),
        "w_read": np.tile([0.0, 2e10, 3e10, 1e10], 2),
        "w_read_dec": np.tile([0.0, 1e10, 3e10, 0.0], 2),
        "kv_read": np.tile([0.0, 1e9, 2e9, 0.0], 2),
        "w_read_pre": np.tile([0.0, 1e10, 0.0, 1e10], 2),
        "kv_write": np.tile([0.0, 2e8, 3e8, 1e8], 2),
        "comm": np.tile([0.0, 1e8, 2e8, 1e8], 2),
        "A_t": np.tile([0.0, 1.0, 2.0, 0.0], 2),
    }
    coefficients = np.array([10, 5, 3, 1, 2, 3, 1e-12, 2e-12, 1e-10, 1e-10, 1e-10])
    fit = {"physics": coefficients, "lag_alpha": 0.6, "cap_w_per_gpu": 1e6,
           "training_source_ids": ["source"]}
    artifact = selected_physics_artifact(
        "A100", "M0", fit,
        [{"hardware": "A100", "model": "gpt-oss-20b", "tp": 2}],
    )
    arch = get_arch("gpt-oss-20b")
    design, names = physics_design(
        ledger, arch, tp=2, hardware_profile=artifact["hardware_profile"],
        mean_kind="M0", dt_s=0.25, moving_average_s=0.25,
        ema_alpha=0.6, run_ids=run_ids,
    )
    expected = design @ np.array([artifact["mode"]["coefficients"][name] for name in names])
    actual = predict_selected_physics(
        ledger, arch, tp=2, hardware="A100", artifact=artifact,
        dt_s=0.25, run_ids=run_ids,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
