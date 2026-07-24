"""
Claim:
The audit reports all metrics on the same one-second trace window, with RMSE
in per-GPU watts and KS agreement equal to one minus empirical KS distance.

Plausible wrong implementations:
- Report node RMSE, making identical per-GPU errors grow with TP.
- Use the KS p-value or KS distance while labeling it agreement.
- Reverse ACF/KS goodness directions when constructing failure correlations.
- Aggregate measured and predicted traces with different window boundaries.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_power_metric_audit import (  # noqa: E402
    aggregate_cells,
    correlation_matrix,
    diagnostic_metrics,
    one_second_pair,
)


def test_one_second_pair_uses_only_matched_complete_windows():
    measured = np.arange(10, dtype=float)
    predicted = measured[:8] + 4.0
    y, p = one_second_pair(measured, predicted, 0.25)
    np.testing.assert_allclose(y, [1.5, 5.5])
    np.testing.assert_allclose(p, [5.5, 9.5])


def test_rmse_is_per_gpu_and_ks_agreement_has_exact_endpoints():
    per_gpu = np.tile([100.0, 200.0], 32)
    shifted = per_gpu + 10.0
    metrics_tp1 = diagnostic_metrics(per_gpu, shifted, tp=1, native_dt=1.0)
    metrics_tp4 = diagnostic_metrics(4 * per_gpu, 4 * shifted, tp=4, native_dt=1.0)
    assert np.isclose(metrics_tp1["rmse_w_per_gpu"], 10.0)
    assert np.isclose(metrics_tp4["rmse_w_per_gpu"], 10.0)
    assert np.isclose(metrics_tp1["ks_agreement"], metrics_tp4["ks_agreement"])
    identical = diagnostic_metrics(per_gpu, per_gpu, tp=1, native_dt=1.0)
    disjoint = diagnostic_metrics(per_gpu, per_gpu + 1000.0, tp=1, native_dt=1.0)
    assert identical["ks_agreement"] == 1.0
    assert disjoint["ks_agreement"] == 0.0


def test_failure_correlations_orient_acf_and_ks_as_errors():
    rows = [
        {"rate": 1.0, "acf_r2": 0.9, "ks_agreement": 0.8},
        {"rate": 2.0, "acf_r2": 0.5, "ks_agreement": 0.6},
        {"rate": 4.0, "acf_r2": -1.0, "ks_agreement": 0.2},
    ]
    matrix = correlation_matrix(rows, ["rate"], ["acf_error", "ks_distance"])
    np.testing.assert_allclose(matrix, [[1.0, 1.0]])


def test_cell_aggregation_preserves_discrete_tp_metadata():
    row = {
        "hardware": "A100", "model": "model", "family": "dense",
        "tp": 4, "rate": 1.0, "role": "train", "surface_supported": True,
        "signed_energy_bias_pct": -5.0,
        **{name: 1.0 for name in (
            "energy_error_pct", "soft_dtw_divergence", "nrmse_range",
            "rmse_w_per_gpu", "acf_r2", "ks_agreement",
            "measured_mean_w_per_gpu", "measured_range_w_per_gpu", "busy_mean",
            "compute_util_mean", "memory_util_mean", "engine_iterations_rate_mean",
            "decode_batch_mean", "waiting_requests_mean", "resident_weight_fraction",
            "residual_step_abs_w_per_gpu", "residual_step_r2",
        )},
    }
    cell = aggregate_cells([row])[0]
    assert cell["tp"] == 4
    assert isinstance(cell["tp"], int)
