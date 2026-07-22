"""
Claim:
The paper transfer panels compare matched one-second per-GPU traces, label the
dense result as request-only zero-shot, and disclose every MoE adaptation and
its retrospective evidence status.

Plausible wrong implementations:
- Plot TP-summed node watts while labeling the axis per GPU.
- Average measured and predicted traces over different windows.
- Substitute measured engine state into a supposedly generative panel.
- Change dynamic terms while presenting either update as idle-only calibration.
- Present either target-idle adaptation as sealed zero-shot transfer.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "plot_transfer_traces.py"
SPEC = importlib.util.spec_from_file_location("transfer_traces", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_one_second_pair_uses_common_window_and_per_gpu_units():
    measured_node = np.asarray([2, 4, 6, 8, 10, 12, 14, 16, 99], float)
    predicted_node = 2.0 * measured_node[:-2]

    measured, predicted = MODULE.per_gpu_one_second_pair(
        measured_node, predicted_node, tp=2, dt_s=0.25
    )

    np.testing.assert_allclose(measured, [2.5])
    np.testing.assert_allclose(predicted, [5.0])


def test_idle_replacement_changes_exact_node_intercept_only():
    names = ["idle_floor", "dynamic"]
    design = np.asarray([[1.0, 2.0], [1.0, 4.0]])
    source = np.asarray([70.0, 5.0])
    updated = MODULE.replace_coefficient(
        source, names, "idle_floor", 90.0
    )

    source_node = MODULE.predict_dense_node(design, source, tp=2)
    updated_node = MODULE.predict_dense_node(design, updated, tp=2)

    np.testing.assert_allclose(updated_node - source_node, [40.0, 40.0])
    np.testing.assert_allclose(updated[1:], source[1:])


def test_candidate_contract_marks_both_idle_updates_retrospective():
    dense = MODULE.CANDIDATE_CONTRACT["dense"]
    moe = MODULE.CANDIDATE_CONTRACT["moe"]

    assert dense["state_source"] == moe["state_source"] == "request_simulation"
    assert "retrospective" in dense["evidence_role"]
    assert "retrospective" in moe["evidence_role"]
    assert any("idle" in change for change in dense["coefficient_changes"])
    assert any("idle" in change for change in moe["coefficient_changes"])
