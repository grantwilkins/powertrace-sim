"""Measured power-meter response applied to equilibrium predictions."""
from __future__ import annotations

import numpy as np

MOVING_AVERAGE_S = {"A100": 0.0, "H100": 1.0}


def apply_response(values, *, dt_s: float, hardware: str, delay_s: float):
    out = np.asarray(values, dtype=float).copy()
    shift = int(round(delay_s / dt_s))
    if shift > 0 and out.shape[0]:
        out = np.concatenate((
            np.repeat(out[:1], min(shift, out.shape[0]), axis=0),
            out[:-shift],
        ), axis=0)
    window = int(round(MOVING_AVERAGE_S[hardware] / dt_s))
    if window <= 1 or not out.shape[0]:
        return out
    cumulative = np.cumsum(out, axis=0)
    averaged = np.empty_like(out)
    head = min(window, out.shape[0])
    shape = (-1,) + (1,) * (out.ndim - 1)
    averaged[:head] = cumulative[:head] / np.arange(1, head + 1).reshape(shape)
    if out.shape[0] > window:
        averaged[window:] = (
            cumulative[window:] - cumulative[:-window]
        ) / window
    return averaged

