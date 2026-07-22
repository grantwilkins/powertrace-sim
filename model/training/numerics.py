"""Small numerical helpers shared by selected-model fitting stages."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

HARDWARE = {
    "A100": {"compute_peak_flops_s": 312e12, "hbm_bandwidth_bytes_s": 2.0e12,
              "hbm_capacity_bytes": 80e9},
    "H100": {"compute_peak_flops_s": 990e12, "hbm_bandwidth_bytes_s": 3.35e12,
              "hbm_capacity_bytes": 80e9},
}


def run_slices(run_id: np.ndarray) -> list[tuple[int, int, int]]:
    run_id = np.asarray(run_id)
    starts = np.r_[0, np.flatnonzero(run_id[1:] != run_id[:-1]) + 1, run_id.size]
    slices = [(int(run_id[lo]), int(lo), int(hi))
              for lo, hi in zip(starts[:-1], starts[1:])]
    if len({run for run, _, _ in slices}) != len(slices):
        raise ValueError("Each run must occupy one contiguous cache segment")
    return slices


def nnls_rms_scaled(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.mean(x ** 2, axis=0))
    keep = scale > 0
    scaled, _ = nnls(x[:, keep] / scale[keep], y)
    coefficients = np.zeros(x.shape[1])
    coefficients[keep] = scaled / scale[keep]
    return coefficients

