"""Frozen calibration and acceptance math for disaggregated confirmation."""
from __future__ import annotations

import numpy as np


def fit_positive_time_scale(
    measured_s: np.ndarray,
    predicted_s: np.ndarray,
    *,
    fixed_overhead_s: float = 0.0,
) -> float:
    """Fit one multiplicative service-time scale above a fixed overhead."""
    measured = np.asarray(measured_s, dtype=float) - fixed_overhead_s
    predicted = np.asarray(predicted_s, dtype=float) - fixed_overhead_s
    if (
        measured.ndim != 1
        or predicted.shape != measured.shape
        or measured.size == 0
        or not np.isfinite(measured).all()
        or not np.isfinite(predicted).all()
        or np.any(measured <= 0.0)
        or np.any(predicted <= 0.0)
    ):
        raise ValueError("time-scale fit requires positive aligned service times")
    scale = float(predicted @ measured) / float(predicted @ predicted)
    if scale <= 0.0:
        raise ValueError("fitted timing scale must be positive")
    return scale


def fit_nonnegative_gain(
    measured: np.ndarray, predictor: np.ndarray, *, measured_idle_w: float,
) -> float:
    """Fit ``idle + gain * predictor`` by pointwise squared error."""
    measured = np.asarray(measured, dtype=float)
    predictor = np.asarray(predictor, dtype=float)
    if (
        measured.ndim != 1
        or predictor.shape != measured.shape
        or measured.size == 0
        or not np.isfinite(measured).all()
        or not np.isfinite(predictor).all()
        or not np.isfinite(measured_idle_w)
    ):
        raise ValueError("gain fit requires aligned finite role samples")
    denominator = float(predictor @ predictor)
    if denominator <= 0.0:
        raise ValueError("gain predictor has no variation or activity")
    return max(0.0, float(predictor @ (measured - measured_idle_w)) / denominator)


def pointwise_loss_ratio(
    measured: np.ndarray, predicted: np.ndarray, null: np.ndarray,
) -> float:
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    null = np.asarray(null, dtype=float)
    if (
        measured.ndim != 1
        or predicted.shape != measured.shape
        or null.shape != measured.shape
        or not np.isfinite(measured).all()
        or not np.isfinite(predicted).all()
        or not np.isfinite(null).all()
    ):
        raise ValueError("loss comparison requires aligned finite traces")
    null_loss = float(np.mean((null - measured) ** 2))
    if null_loss <= 0.0:
        raise ValueError("pointwise null loss must be positive")
    return float(np.mean((predicted - measured) ** 2)) / null_loss


def role_acceptance(
    metrics: dict[str, float],
    loss_ratio: float,
    thresholds: dict[str, float],
) -> dict[str, bool]:
    checks = {
        "correlation": metrics["correlation"] >= thresholds["correlation_min"],
        "std_ratio": (
            thresholds["std_ratio_min"]
            <= metrics["std_ratio"]
            <= thresholds["std_ratio_max"]
        ),
        "p95_error": (
            metrics["p95_error_pct"] <= thresholds["p95_error_pct_max"]
        ),
        "duty_null_loss": (
            loss_ratio <= thresholds["model_to_duty_null_loss_ratio_max"]
        ),
    }
    return {**checks, "accepted": all(checks.values())}


def pair_common_bin_samples(
    left_bins: np.ndarray,
    left_values: np.ndarray,
    right_bins: np.ndarray,
    right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair samples by bin and within-bin occurrence without averaging."""
    left_bins = np.asarray(left_bins, dtype=int)
    right_bins = np.asarray(right_bins, dtype=int)
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)
    if (
        left_bins.ndim != 1
        or right_bins.ndim != 1
        or left_values.shape != left_bins.shape
        or right_values.shape != right_bins.shape
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
    ):
        raise ValueError("replay samples require aligned finite vectors")
    left, right = [], []
    for bin_index in sorted(set(left_bins) & set(right_bins)):
        left_bin = left_values[left_bins == bin_index]
        right_bin = right_values[right_bins == bin_index]
        count = min(left_bin.size, right_bin.size)
        left.extend(left_bin[:count])
        right.extend(right_bin[:count])
    if len(left) < 2:
        raise ValueError("replays have fewer than two common native samples")
    return np.asarray(left), np.asarray(right)


def average_trace_rows(
    rows: list[dict], *, bin_s: float,
) -> list[dict[str, float | str]]:
    """Average aligned trace columns in non-overlapping time bins."""
    if not np.isfinite(bin_s) or bin_s <= 0.0 or not rows:
        raise ValueError("trace averaging requires rows and a positive bin")
    value_keys = [
        key for key in rows[0]
        if key not in {"cell", "time_s"}
    ]
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        cell = str(row["cell"])
        time_s = float(row["time_s"])
        if not np.isfinite(time_s):
            raise ValueError("trace times must be finite")
        groups.setdefault((cell, int(np.floor(time_s / bin_s))), []).append(row)
    output = []
    for (cell, bin_index), group in sorted(groups.items()):
        averaged: dict[str, float | str] = {
            "cell": cell,
            "time_s": (bin_index + 0.5) * bin_s,
        }
        for key in value_keys:
            values = np.asarray([row[key] for row in group], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("trace values must be finite")
            averaged[key] = float(values.mean())
        output.append(averaged)
    return output
