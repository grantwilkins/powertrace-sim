"""
Claim:
The held-out-model plots contain only model/setup holdouts, cover every rate
including 4 requests/s, choose a central rate-4 repetition, and compare both
traces in per-GPU one-second units.

Plausible wrong implementations:
- Mix training or TP-holdout rows into the model-holdout figure.
- Omit rate 4 from a setup while still drawing a connected rate line.
- Cherry-pick the lowest-error rate-4 repetition.
- Divide only measured or predicted node power by TP.
"""
import numpy as np

from plot_coverage_heldout_models import (
    RATES,
    aggregate_cells,
    metric_medoid,
    one_second_run,
)


def _row(run_id, rate, role="heldout_model", energy=2.0, rmse=10.0, acf=0.9):
    return {
        "run_id": run_id, "hardware": "A100", "model": "heldout",
        "tp": 4, "rate": rate, "role": role,
        "energy_error_pct": energy, "rmse_w_per_gpu": rmse,
        "acf_r2": acf, "nrmse_range": 0.1,
    }


def test_cell_aggregation_retains_all_six_rates_including_four():
    rows = [_row(index, rate) for index, rate in enumerate(RATES)]
    cells = aggregate_cells(rows)
    assert {row["rate"] for row in cells} == set(RATES)
    assert next(row for row in cells if row["rate"] == 4.0)["runs"] == 1


def test_rate4_selection_uses_metric_medoid_not_best_run():
    rows = [
        _row(1, 4.0, energy=0.0, rmse=0.0, acf=1.0),
        _row(2, 4.0, energy=2.0, rmse=2.0, acf=0.8),
        _row(3, 4.0, energy=100.0, rmse=100.0, acf=-1.0),
    ]
    assert metric_medoid(rows)["run_id"] == 2


def test_one_second_trace_converts_both_node_series_to_per_gpu():
    cache = {
        "run_id": np.asarray([7, 7, 7, 7]),
        "tp": np.asarray([2, 2, 2, 2]),
        "power": np.asarray([20.0, 24.0, 28.0, 32.0]),
        "dt_s": np.asarray(0.25),
    }
    measured, predicted = one_second_run(
        cache, np.asarray([40.0, 44.0, 48.0, 52.0]), 7
    )
    np.testing.assert_allclose(measured, [13.0])
    np.testing.assert_allclose(predicted, [23.0])
