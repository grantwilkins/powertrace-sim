"""
Power trace evaluation metrics.

This module contains functions for computing evaluation metrics between
ground truth and generated power traces.
"""
from __future__ import annotations

from typing import Dict

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


def _integrate_trapezoid(values: np.ndarray, *, dx: float) -> float:
    """
    Integrate values using trapezoidal rule.

    Args:
        values: Array of values to integrate.
        dx: Spacing between values.

    Returns:
        Integral approximation.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(arr, dx=float(dx)))
    return float(np.trapz(arr, dx=float(dx)))


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
            - delta_energy_pct: Percent difference in total energy
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

    energy_gt = _integrate_trapezoid(gt, dx=float(dt))
    energy_pred = _integrate_trapezoid(pred, dx=float(dt))
    delta_energy_pct = float(100.0 * (energy_pred - energy_gt) / (abs(energy_gt) + EPS))

    return {
        "ks_stat": ks_statistic(gt, pred),
        "acf_r2": autocorrelation_r2(gt, pred, max_lag=int(acf_max_lag)),
        "nrmse": nrmse,
        "p95_error_pct": p95_error_pct,
        "p99_error_pct": p99_error_pct,
        "delta_energy_pct": delta_energy_pct,
    }
