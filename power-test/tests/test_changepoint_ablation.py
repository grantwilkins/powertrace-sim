"""
Claim:
The changepoint diagnostic finds the exact two-mean boundary, and a held-out
prediction uses step or leaky-work-dose parameters learned only from explicitly
supplied training runs.

Plausible wrong implementations:
- Report the sample before or after the true boundary.
- Prefer a boundary that creates an undersized segment.
- Integrate total/per-GPU power or apply cooling with the wrong time units.
- Read the held-out residual while fitting its dose threshold or correction.
- Add a dose correction that silently changes predicted trace energy.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from changepoint_ablation import apply_step, best_mean_step, fit_step
from dose_ablation import (
    apply_dose_step,
    fit_dose_step,
    leaky_work_dose,
    plot_rows,
)


def test_exact_step_boundary_and_additive_invariance():
    signal = np.asarray([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])

    result = best_mean_step(signal, dt_s=2.0, min_segment_s=4.0)
    shifted = best_mean_step(signal + 17.0, dt_s=2.0, min_segment_s=4.0)

    assert result["split_bin"] == 3
    assert result["split_s"] == 6.0
    assert result["jump_w"] == 4.0
    assert result["step_r2"] == 1.0
    assert shifted["split_bin"] == result["split_bin"]
    assert shifted["jump_w"] == result["jump_w"]


def _record(run_id, residual):
    residual = np.asarray(residual, float)
    baseline = np.full(residual.size, 100.0)
    return {
        "run_id": run_id,
        "duration_s": float(residual.size),
        "baseline": baseline,
        "residual": residual,
        "split_s": 3.0,
    }


def test_fit_and_prediction_do_not_read_held_out_residual():
    train = [_record(1, [-2, -2, -2, 3, 3, 3])]
    test_a = _record(2, [100, 100, 100, -100, -100, -100])
    test_b = _record(2, [-900, 70, 20, 400, 0, -30])

    fit = fit_step(train)
    predicted_a = apply_step(test_a, fit)
    predicted_b = apply_step(test_b, fit)

    assert fit["train_run_ids"] == [1]
    np.testing.assert_allclose(predicted_a, [98, 98, 98, 103, 103, 103])
    np.testing.assert_allclose(predicted_b, predicted_a)


def test_relative_step_scales_with_target_board_power():
    train = [_record(1, [-10, -10, -10, 10, 10, 10])]
    target = _record(2, np.zeros(6))
    target["baseline"] = np.full(6, 200.0)

    predicted = apply_step(target, fit_step(train, relative=True))

    np.testing.assert_allclose(predicted, [180, 180, 180, 220, 220, 220])


def test_leaky_work_dose_has_joule_units_and_cools():
    cumulative = leaky_work_dose(
        np.asarray([2.0, 2.0, 2.0]), dt_s=0.5, tau_s=float("inf")
    )
    cooling = leaky_work_dose(
        np.asarray([2.0, 0.0, 0.0]), dt_s=np.log(2.0), tau_s=1.0
    )

    np.testing.assert_allclose(cumulative, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cooling, [1.0, 0.5, 0.25])


def _dose_record(run_id, dynamic_power, residual, rate=2.0):
    record = _record(run_id, residual)
    record |= {
        "dynamic_node_power_w": np.asarray(dynamic_power, float),
        "split_bin": 3,
        "tp": 2,
        "rate": rate,
        "jump_w": 8.0,
    }
    return record


def test_dose_fit_is_held_out_and_correction_preserves_energy():
    train = [
        _dose_record(1, [1, 1, 1, 1, 1, 1], [-4, -4, -4, 4, 4, 4]),
        _dose_record(2, [1, 1, 1, 1, 1, 1], [-4, -4, -4, 4, 4, 4]),
    ]
    target_a = _dose_record(
        3, [1, 1, 1, 1, 1, 1], [500, -20, 70, -900, 2, 1], rate=4.0
    )
    target_b = _dose_record(
        3, [1, 1, 1, 1, 1, 1], [-8, 6, 2, 400, -30, 90], rate=4.0
    )

    fit = fit_dose_step(train, tau_grid_s=(1.0, float("inf")))
    predicted_a, trigger_a = apply_dose_step(target_a, fit)
    predicted_b, trigger_b = apply_dose_step(target_b, fit)

    assert fit["train_run_ids"] == [1, 2]
    assert np.isinf(fit["tau_s"])
    assert fit["threshold_j"] == 4.0
    assert fit["jump_w_per_gpu_per_rate"] == 2.0
    assert trigger_a == trigger_b == 3.0
    np.testing.assert_allclose(predicted_a, predicted_b)
    assert np.sum(predicted_a - target_a["baseline"]) == 0.0


def test_plot_uses_one_llama_holdout_per_hardware():
    rows = [
        {"hardware": "H100", "model": "deepseek-r1-distill-70b", "run_id": 1},
        {"hardware": "H100", "model": "llama-3-70b", "run_id": 3},
        {"hardware": "H100", "model": "llama-3-70b", "run_id": 2},
        {"hardware": "A100", "model": "llama-3-70b", "run_id": 4},
    ]

    selected = plot_rows(rows)

    assert [(row["hardware"], row["run_id"]) for row in selected] == [
        ("A100", 4),
        ("H100", 2),
    ]
