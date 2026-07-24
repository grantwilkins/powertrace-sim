"""
Claim:
Qwen transfer applies each frozen source feature contract at node power, changes
only the per-GPU idle intercept for idle calibration, and scales hardware work
terms by energy per unit work rather than raw fitted watts.

Plausible wrong implementations:
- Drop the GPT-OSS-20B multi-GPU floor because the target contains only TP2.
- Apply a per-GPU idle change once at node level instead of once per GPU.
- Change dynamic coefficients while labeling a candidate idle-only.
- Scale hardware transfer by coefficient ratios while ignoring peak throughput.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "analyze_qwen3_30b_moe_transfer.py"
SPEC = importlib.util.spec_from_file_location("qwen_transfer", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_idle_calibration_is_exact_per_gpu_intercept_shift():
    names = ["idle", "dynamic"]
    design = np.asarray([[1.0, 0.0], [1.0, 3.0]])
    source = np.asarray([70.0, 5.0])
    updated = MODULE.replace_idle(source, names, 120.0)

    source_node = MODULE.predict_moe_node(design, source, tp=2, h100_meter=False)
    updated_node = MODULE.predict_moe_node(design, updated, tp=2, h100_meter=False)

    np.testing.assert_allclose(updated_node - source_node, [100.0, 100.0])
    np.testing.assert_allclose(updated[1:], source[1:])


def test_source_contract_controls_multi_gpu_floor_column(monkeypatch):
    ledger = {"run_id": np.asarray([0, 0]), "tp": np.asarray([2.0, 2.0])}
    node_design = np.arange(10.0).reshape(2, 5)
    monkeypatch.setattr(MODULE, "moe_design", lambda _: node_design)
    monkeypatch.setattr(MODULE, "moe_compute_coordinate", lambda _: np.asarray([5.0, 6.0]))
    fit = {"feature_names": list(MODULE.MOE_FEATURES)}

    design, names = MODULE.design_for_source_fit(ledger, fit)

    assert names == list(MODULE.MOE_FEATURES)
    np.testing.assert_allclose(design[:, 1], node_design[:, 1] / 2.0)


def test_hardware_ratio_is_energy_per_work_not_raw_watts(monkeypatch):
    monkeypatch.setitem(MODULE.HARDWARE, "A100", {
        "hbm_bandwidth_bytes_s": 2.0, "compute_peak_flops_s": 4.0,
    })
    monkeypatch.setitem(MODULE.HARDWARE, "H100", {
        "hbm_bandwidth_bytes_s": 8.0, "compute_peak_flops_s": 16.0,
    })
    fit = {"dense": {
        "A100": {
            "feature_names": ["duty_sqrt_memory_util", "compute_util"],
            "coefficients": [10.0, 20.0],
        },
        "H100": {
            "feature_names": ["duty_sqrt_memory_util", "compute_util"],
            "coefficients": [20.0, 40.0],
        },
    }}

    ratios = MODULE.hardware_energy_ratios(fit)

    assert ratios["logical_memory_util_lag_250ms"] == 0.5
    assert ratios["duty_sqrt_exact_compute_util"] == 0.5
