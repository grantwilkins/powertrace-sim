"""Meter response chain: work-onset delay plus the H100 reading filter.

The H100 counter reports a pure 1.0 s trailing moving average (probe-fitted,
cited arXiv:2312.02741, independently confirmed); the A100 counter is
near-instant at 250 ms bins and gets the delay only. No first-order device
time constant exists for either hardware (no step probe has run), so that
stage is omitted. The chain is linear, so fitting may filter design columns
through it and regress against raw measured power. Always applied per run,
never across run boundaries.
"""
from __future__ import annotations

import numpy as np

MOVING_AVERAGE_S = {"A100": 0.0, "H100": 1.0}


def apply_chain(pred, dt: float, hardware: str, delay_s: float) -> np.ndarray:
    """Delay then (H100 only) trailing moving average, for a single run.

    Works along axis 0, so a whole design matrix can be filtered at once.
    """
    out = np.asarray(pred, float).copy()
    n = out.shape[0]
    shift = int(round(delay_s / dt))
    if shift > 0 and n:
        out = np.concatenate([np.repeat(out[:1], min(shift, n), axis=0),
                              out[:-shift]], axis=0)
    window = int(round(MOVING_AVERAGE_S[hardware] / dt))
    if window > 1 and n:
        # Partial windows at the start use the available prefix.
        cumulative = np.cumsum(out, axis=0)
        counts = np.arange(1, min(window, n) + 1, dtype=float)
        averaged = np.empty_like(out)
        averaged[:window] = cumulative[:window] / counts.reshape(
            (-1,) + (1,) * (out.ndim - 1))
        averaged[window:] = (cumulative[window:] - cumulative[:-window]) / window
        out = averaged
    return out


def apply_chain_by_run(values, run_id, dt: float, hardware: str,
                       delay_s: float) -> np.ndarray:
    """Apply the chain independently to each contiguous run segment."""
    values = np.asarray(values, float)
    run_id = np.asarray(run_id)
    starts = np.r_[0, np.flatnonzero(run_id[1:] != run_id[:-1]) + 1, values.shape[0]]
    out = np.empty_like(values)
    for lo, hi in zip(starts[:-1], starts[1:]):
        out[lo:hi] = apply_chain(values[lo:hi], dt, hardware, delay_s)
    return out
