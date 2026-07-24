"""Align native power samples to the model's half-open 250 ms bins."""
from __future__ import annotations

import numpy as np


def sample_native_power(
    timestamps: np.ndarray,
    measured: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    start_epoch_s: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    timestamps = np.asarray(timestamps, dtype=float)
    measured = np.asarray(measured, dtype=float)
    predicted = {
        name: np.asarray(values, dtype=float)
        for name, values in predictions.items()
    }
    if timestamps.ndim != 1 or measured.shape[0] != timestamps.size:
        raise ValueError("native timestamps and measured samples must align")
    if measured.ndim not in (1, 2) or not predicted:
        raise ValueError("native measured power and predictions are required")
    if dt_s <= 0.0 or not np.isfinite(start_epoch_s):
        raise ValueError("native sampling needs a finite origin and positive cadence")
    if not np.isfinite(timestamps).all() or not np.isfinite(measured).all():
        raise ValueError("native measurements must be finite")
    if any(values.ndim != 1 or not np.isfinite(values).all()
           for values in predicted.values()):
        raise ValueError("native predictions must be finite vectors")

    bins = np.floor((timestamps - start_epoch_s) / dt_s).astype(int)
    horizon = min(values.size for values in predicted.values())
    keep = (bins >= 0) & (bins < horizon)
    return (
        timestamps[keep] - start_epoch_s,
        measured[keep],
        {name: values[bins[keep]] for name, values in predicted.items()},
    )
