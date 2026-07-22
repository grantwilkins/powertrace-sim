"""
Claim:
Secondary metrics preserve explicit time windows, ramp resolutions, and LDC ranks.

Plausible wrong implementations:
- Treat power error as energy without window averaging.
- Compare native ramps while labeling them one-second ramps.
- Use an ascending percentile instead of the exceedance-rank LDC estimator.
"""

import numpy as np

from evaluation_core import bootstrap_intervals
from feature_metrics import secondary_trace_metrics


def test_identical_trace_has_zero_errors_and_exact_ldc_rank():
    trace = np.repeat(np.arange(1.0, 41.0), 4)
    result = secondary_trace_metrics(trace, trace, dt_s=0.25, cap_w=40.0)
    assert all(value == 0 for key, value in result.items() if "error" in key)
    assert result["ldc_measured_0.50_w"] == 20.0
    assert result["cap_hit_fraction"] == 4 / 160


def test_constant_offset_changes_energy_and_tail_but_not_ramps():
    measured = np.full(240, 100.0)
    predicted = measured + 10.0
    result = secondary_trace_metrics(measured, predicted, dt_s=0.25)
    assert result["window_energy_30s_median_error_pct"] == 10.0
    assert result["power_p99_abs_error_w"] == 10.0
    assert result["ramp_250ms_p95_abs_error_w"] == 0.0
    assert result["ramp_1s_p95_abs_error_w"] == 0.0
    assert result["ldc_error_0.01_w"] == 10.0


def test_bootstrap_resamples_whole_runs_deterministically():
    rows = [{"split": "S", "candidate": "M0", "energy_error_pct": value,
             "acf_r2": 1 - value / 10, "acf_mae": value / 10,
             "nrmse_range": value / 20,
             "soft_dtw_divergence": value / 30}
            for value in (1.0, 2.0, 3.0)]
    assert bootstrap_intervals(rows, seed=7, samples=20) == bootstrap_intervals(
        rows, seed=7, samples=20)
