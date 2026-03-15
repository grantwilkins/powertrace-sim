"""
Power trace evaluation metrics.

This module contains functions for computing evaluation metrics between
ground truth and generated power traces.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

EPS = 1e-12


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Kolmogorov-Smirnov statistic between two distributions.

    Args:
        x: First sample array.
        y: Second sample array.

    Returns:
        Maximum absolute difference between empirical CDFs.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    ys = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if xs.size == 0 or ys.size == 0:
        return float("nan")
    values = np.concatenate([xs, ys])
    values.sort()
    cdf_x = np.searchsorted(xs, values, side="right") / float(xs.size)
    cdf_y = np.searchsorted(ys, values, side="right") / float(ys.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation function up to max_lag.

    Args:
        values: Time series values.
        max_lag: Maximum lag to compute.

    Returns:
        Array of ACF values from lag 0 to max_lag.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return np.zeros((1,), dtype=np.float64)
    lag = int(max(0, min(max_lag, n - 1)))
    centered = arr - float(np.mean(arr))
    denom = float(np.dot(centered, centered))
    out = np.zeros((lag + 1,), dtype=np.float64)
    out[0] = 1.0
    if lag == 0 or denom <= EPS:
        return out
    for k in range(1, lag + 1):
        out[k] = float(np.dot(centered[:-k], centered[k:]) / denom)
    return out


def _total_energy_from_bins(values: np.ndarray, *, dt: float) -> float:
    """
    Compute total trace energy from per-bin average power samples.

    Args:
        values: Array of power values in watts, one value per time bin.
        dt: Duration of each time bin in seconds.

    Returns:
        Total energy in joules.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.sum(arr) * float(dt))


def autocorrelation_r2(real: np.ndarray, synthetic: np.ndarray, max_lag: int = 50) -> float:
    """
    Compute R^2 between autocorrelation functions of real and synthetic traces.

    Args:
        real: Ground truth time series.
        synthetic: Generated time series.
        max_lag: Maximum lag for ACF computation.

    Returns:
        R^2 value measuring ACF similarity (1.0 = perfect match).
    """
    r = np.asarray(real, dtype=np.float64).reshape(-1)
    s = np.asarray(synthetic, dtype=np.float64).reshape(-1)
    if r.size < 3 or s.size < 3:
        return float("nan")
    lag = int(min(max_lag, r.size - 1, s.size - 1))
    if lag < 1:
        return float("nan")
    r_acf = _acf(r, lag)[1:]
    s_acf = _acf(s, lag)[1:]
    if r_acf.size == 0:
        return float("nan")
    sse = float(np.sum((r_acf - s_acf) ** 2))
    tss = float(np.sum((r_acf - float(np.mean(r_acf))) ** 2))
    if tss <= EPS:
        return float(1.0 if sse <= 1e-10 else 0.0)
    return float(1.0 - (sse / tss))


def autocorrelation_r2_aggregate(
    real_traces: Sequence[np.ndarray],
    synthetic_traces: Sequence[np.ndarray],
    max_lag: int = 50,
) -> float:
    """
    Compare average held-out ACF structure across multiple trace pairs.

    Each trace pair contributes its own ACF up to the pair-specific valid lag.
    Those ACF vectors are then averaged across traces, and a single R^2 score
    is computed on the averaged ACFs.
    """
    real_rows = []
    synth_rows = []
    max_lag_int = int(max(1, max_lag))

    for real, synthetic in zip(real_traces, synthetic_traces):
        r = np.asarray(real, dtype=np.float64).reshape(-1)
        s = np.asarray(synthetic, dtype=np.float64).reshape(-1)
        n = int(min(r.size, s.size))
        if n < 3:
            continue
        r = r[:n]
        s = s[:n]
        lag = int(min(max_lag_int, n - 1))
        if lag < 1:
            continue

        r_acf = _acf(r, lag)[1:]
        s_acf = _acf(s, lag)[1:]
        r_pad = np.full((max_lag_int,), np.nan, dtype=np.float64)
        s_pad = np.full((max_lag_int,), np.nan, dtype=np.float64)
        r_pad[: r_acf.size] = r_acf
        s_pad[: s_acf.size] = s_acf
        real_rows.append(r_pad)
        synth_rows.append(s_pad)

    if len(real_rows) == 0:
        return float("nan")

    real_stack = np.vstack(real_rows)
    synth_stack = np.vstack(synth_rows)
    real_mean = np.full((max_lag_int,), np.nan, dtype=np.float64)
    synth_mean = np.full((max_lag_int,), np.nan, dtype=np.float64)
    real_counts = np.sum(np.isfinite(real_stack), axis=0)
    synth_counts = np.sum(np.isfinite(synth_stack), axis=0)
    real_mask = real_counts > 0
    synth_mask = synth_counts > 0
    real_mean[real_mask] = (
        np.nansum(real_stack[:, real_mask], axis=0) / real_counts[real_mask]
    )
    synth_mean[synth_mask] = (
        np.nansum(synth_stack[:, synth_mask], axis=0) / synth_counts[synth_mask]
    )
    mask = np.isfinite(real_mean) & np.isfinite(synth_mean)
    if int(np.sum(mask)) == 0:
        return float("nan")

    r_acf = real_mean[mask]
    s_acf = synth_mean[mask]
    sse = float(np.sum((r_acf - s_acf) ** 2))
    tss = float(np.sum((r_acf - float(np.mean(r_acf))) ** 2))
    if tss <= EPS:
        return float(1.0 if sse <= 1e-10 else 0.0)
    return float(1.0 - (sse / tss))


def compute_aggregate_power_metrics(
    ground_truth_traces: Sequence[np.ndarray],
    generated_traces: Sequence[np.ndarray],
    *,
    dt: float,
    acf_max_lag: int = 50,
) -> Dict[str, float]:
    """
    Compute held-out aggregate metrics across a collection of trace pairs.

    KS, NRMSE, percentile errors, and energy are computed on all held-out points
    pooled together. ACF R^2 is computed from trace-level ACFs averaged across
    held-out traces, avoiding concatenation artifacts at trace boundaries.
    """
    gt_chunks = []
    pred_chunks = []

    for ground_truth_w, generated_w in zip(ground_truth_traces, generated_traces):
        gt = np.asarray(ground_truth_w, dtype=np.float64).reshape(-1)
        pred = np.asarray(generated_w, dtype=np.float64).reshape(-1)
        n = int(min(len(gt), len(pred)))
        if n <= 0:
            continue
        gt_chunks.append(gt[:n])
        pred_chunks.append(pred[:n])

    if len(gt_chunks) == 0 or len(pred_chunks) == 0:
        raise ValueError("No aligned traces for aggregate metric computation.")

    gt_all = np.concatenate(gt_chunks, axis=0).astype(np.float64)
    pred_all = np.concatenate(pred_chunks, axis=0).astype(np.float64)

    err = pred_all - gt_all
    rmse = float(np.sqrt(np.mean(err**2)))
    scale = float(np.max(gt_all) - np.min(gt_all))
    nrmse = float(rmse / (scale + EPS))

    p95_gt = float(np.percentile(gt_all, 95))
    p95_pred = float(np.percentile(pred_all, 95))
    p99_gt = float(np.percentile(gt_all, 99))
    p99_pred = float(np.percentile(pred_all, 99))

    p95_error_pct = float(100.0 * abs(p95_pred - p95_gt) / (abs(p95_gt) + EPS))
    p99_error_pct = float(100.0 * abs(p99_pred - p99_gt) / (abs(p99_gt) + EPS))

    energy_gt = _total_energy_from_bins(gt_all, dt=float(dt))
    energy_pred = _total_energy_from_bins(pred_all, dt=float(dt))
    delta_energy_pct = float(
        100.0 * abs(energy_pred - energy_gt) / (abs(energy_gt) + EPS)
    )

    return {
        "ks_stat": ks_statistic(gt_all, pred_all),
        "acf_r2": autocorrelation_r2_aggregate(
            gt_chunks, pred_chunks, max_lag=int(acf_max_lag)
        ),
        "nrmse": nrmse,
        "p95_error_pct": p95_error_pct,
        "p99_error_pct": p99_error_pct,
        "delta_energy_pct": delta_energy_pct,
    }


def compute_power_metrics(
    ground_truth_w: np.ndarray,
    generated_w: np.ndarray,
    *,
    dt: float,
    acf_max_lag: int = 50,
) -> Dict[str, float]:
    """
    Compute evaluation metrics between ground truth and generated power traces.

    Args:
        ground_truth_w: Ground truth power trace in watts.
        generated_w: Generated power trace in watts.
        dt: Time step duration in seconds.
        acf_max_lag: Maximum lag for ACF R^2 computation.

    Returns:
        Dictionary with metrics:
            - ks_stat: Kolmogorov-Smirnov statistic
            - acf_r2: Autocorrelation R^2
            - nrmse: Normalized root mean squared error
            - p95_error_pct: Percent error in 95th percentile
            - p99_error_pct: Percent error in 99th percentile
            - delta_energy_pct: Absolute percent error in total trace energy
    """
    gt = np.asarray(ground_truth_w, dtype=np.float64).reshape(-1)
    pred = np.asarray(generated_w, dtype=np.float64).reshape(-1)
    n = int(min(len(gt), len(pred)))
    if n <= 0:
        raise ValueError("No aligned points for metric computation.")
    gt = gt[:n]
    pred = pred[:n]

    err = pred - gt
    rmse = float(np.sqrt(np.mean(err**2)))
    scale = float(np.max(gt) - np.min(gt))
    nrmse = float(rmse / (scale + EPS))

    p95_gt = float(np.percentile(gt, 95))
    p95_pred = float(np.percentile(pred, 95))
    p99_gt = float(np.percentile(gt, 99))
    p99_pred = float(np.percentile(pred, 99))

    p95_error_pct = float(100.0 * abs(p95_pred - p95_gt) / (abs(p95_gt) + EPS))
    p99_error_pct = float(100.0 * abs(p99_pred - p99_gt) / (abs(p99_gt) + EPS))

    # Energy error is intentionally end-of-trace only: compare total joules per
    # held-out trace, not pointwise power deviations over time.
    energy_gt = _total_energy_from_bins(gt, dt=float(dt))
    energy_pred = _total_energy_from_bins(pred, dt=float(dt))
    delta_energy_pct = float(
        100.0 * abs(energy_pred - energy_gt) / (abs(energy_gt) + EPS)
    )

    return {
        "ks_stat": ks_statistic(gt, pred),
        "acf_r2": autocorrelation_r2(gt, pred, max_lag=int(acf_max_lag)),
        "nrmse": nrmse,
        "p95_error_pct": p95_error_pct,
        "p99_error_pct": p99_error_pct,
        "delta_energy_pct": delta_energy_pct,
    }
