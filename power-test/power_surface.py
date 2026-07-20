"""Minimal static power surface on executed-work coordinates.

Node power is a non-negative combination of a model-loaded floor, a fabric
floor, active duty, one-hinge concave compute and HBM responses, and engine
iteration rate. Arrival labels and a fixed engine-token budget are absent.
An operating power limit is applied only when a run binds one explicitly.
"""
from __future__ import annotations

import numpy as np

# Cited datasheet peaks, same constants as feature-test/evaluation_core.py.
HARDWARE = {
    "A100": {
        "compute_peak_flops_s": 312e12,
        "hbm_bandwidth_bytes_s": 2.0e12,
        "hbm_capacity_bytes": 80e9,
    },
    "H100": {
        "compute_peak_flops_s": 990e12,
        "hbm_bandwidth_bytes_s": 3.35e12,
        "hbm_capacity_bytes": 80e9,
    },
}
UTIL_HINGE = 0.4
# Conditioning constant only (DESIGN.md 2).
ITER_RATE_SCALE = 1000.0


def thermal_state(dynamic_power, run_id, dt_s: float, tau_s: float) -> np.ndarray:
    """Causal first-order heat state, reset to ambient at each run boundary."""
    power = np.asarray(dynamic_power, float).reshape(-1)
    runs = np.asarray(run_id).reshape(-1)
    if power.size != runs.size:
        raise ValueError("dynamic_power and run_id must have the same length")
    if dt_s <= 0.0 or tau_s <= 0.0:
        raise ValueError("dt_s and tau_s must be positive")
    rho = np.exp(-float(dt_s) / float(tau_s))
    state = np.empty_like(power)
    previous_run = None
    previous_state = 0.0
    for index, (value, run) in enumerate(zip(power, runs)):
        if run != previous_run:
            previous_state = 0.0
            previous_run = run
        previous_state = rho * previous_state + (1.0 - rho) * value
        state[index] = previous_state
    return state


def dtype_scale(d) -> np.ndarray:
    """Fractional FP8 compute scale, 1 - 0.5*clip(fp8_flop_frac, 0, 1)."""
    if "fp8_flop_frac" in d:
        return 1.0 - 0.5 * np.clip(np.asarray(d["fp8_flop_frac"], float), 0.0, 1.0)
    fp8 = np.asarray(d["fp8"], float)
    return np.where(fp8 > 0, 0.5, 1.0)


def surface_design(d, hardware: str) -> tuple[np.ndarray, list[str]]:
    """Design matrix and column names for one hardware."""
    profile = HARDWARE[hardware]
    tp = np.asarray(d["tp"], float)
    tokens = np.asarray(d["pre_tok"], float) + np.asarray(d["dec_tok"], float)
    if {
        "transformer_active_params", "output_head_params", "logit_tokens_rate"
    } <= set(d):
        compute_flops = (
            dtype_scale(d)
            * 2.0
            * np.asarray(d["transformer_active_params"], float)
            * tokens
            + 2.0
            * np.asarray(d["output_head_params"], float)
            * np.asarray(d["logit_tokens_rate"], float)
        )
    else:
        compute_flops = (
            dtype_scale(d) * 2.0 * np.asarray(d["n_active"], float) * tokens
        )
    u_compute = compute_flops / (tp * profile["compute_peak_flops_s"])
    u_memory = ((np.asarray(d["w_read"], float) + np.asarray(d["kv_read"], float)
                 + np.asarray(d["kv_write"], float))
                / (tp * profile["hbm_bandwidth_bytes_s"]))
    iters = np.asarray(d["engine_iterations_rate"], float)
    columns = [
        tp,
        tp * (tp > 1),
        np.asarray(d["w_bytes"], float) / profile["hbm_capacity_bytes"],
        tp * np.asarray(d["busy"], float),
    ]
    names = ["tp", "tp_link", "resident_weights", "busy_tp"]
    for prefix, u in (("compute", u_compute), ("memory", u_memory)):
        columns += [tp * u, tp * np.minimum(u, UTIL_HINGE)]
        names += [f"{prefix}_linear", f"{prefix}_hinge_{UTIL_HINGE:g}"]
    columns.append(tp * iters / ITER_RATE_SCALE)
    names.append("iter_rate")
    return np.column_stack(columns), names


def predict(design: np.ndarray, coefficients: np.ndarray, tp: np.ndarray,
            hardware: str, *, power_limit_w: np.ndarray | None = None,
            loaded_idle_delta_w_per_gpu: float | np.ndarray = 0.0) -> np.ndarray:
    """Predict node power, applying only an explicitly bound per-GPU limit."""
    raw = (
        design @ np.asarray(coefficients, float)
        + np.asarray(loaded_idle_delta_w_per_gpu, float)
        * np.asarray(tp, float)
    )
    if power_limit_w is None:
        return raw
    return np.minimum(raw, np.asarray(power_limit_w, float) * np.asarray(tp, float))
