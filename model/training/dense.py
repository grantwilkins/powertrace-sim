"""Physical dense-power design, floor calibration, and source-only fitting."""
from __future__ import annotations

import numpy as np

from model.training.numerics import nnls_rms_scaled, run_slices
from model.training.numerics import HARDWARE
from model.power.response import apply_response

DENSE_FEATURES = (
    "idle_floor",
    "active_weight_fraction",
    "compute_util",
    "duty_sqrt_memory_util",
)
DELAY_CANDIDATES_S = (0.0, 0.25, 0.5, 0.75)
FIT_TIMESTEP_S = 1.0
TAIL_QUANTILE = 0.9
IDLE_MINIMUM_S = 4.0
IDLE_SETTLE_S = 2.0


def resident_fraction(weight_bytes, tp, hardware: str) -> np.ndarray:
    return np.clip(
        np.asarray(weight_bytes, float)
        / (np.asarray(tp, float) * HARDWARE[hardware]["hbm_capacity_bytes"]),
        0.0,
        1.0,
    )


def phase_work_utilization(
    sub: dict, hardware: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.asarray(sub["tp"], float)
    compute = tp * HARDWARE[hardware]["compute_peak_flops_s"]
    bandwidth = tp * HARDWARE[hardware]["hbm_bandwidth_bytes_s"]
    prefill = (
        np.asarray(sub["prefill_gemm_flops_rate"], float)
        + np.asarray(sub["prefill_attn_flops_rate"], float)
    ) / compute
    decode = (
        np.asarray(sub["decode_gemm_flops_rate"], float)
        + np.asarray(sub["decode_attn_flops_rate"], float)
    ) / compute
    memory = (
        np.asarray(sub["w_read"], float)
        + np.asarray(sub["prefill_attn_bytes_rate"], float)
        + np.asarray(sub["decode_attn_bytes_rate"], float)
    ) / bandwidth
    return prefill, decode, memory


def raw_design(sub: dict, hardware: str) -> np.ndarray:
    tp = np.asarray(sub["tp"], float)
    prefill, decode, memory = phase_work_utilization(sub, hardware)
    busy = np.clip(np.asarray(sub["busy"], float), 0.0, 1.0)
    active_weight = (
        busy
        * resident_fraction(sub["w_bytes"], tp, hardware)
    )
    return np.column_stack((
        np.ones(tp.size),
        active_weight,
        np.clip(prefill + decode, 0.0, None),
        np.sqrt(busy * np.clip(memory, 0.0, None)),
    ))


def filter_design(design, run_id, dt_s: float, hardware: str,
                  delay_s: float) -> np.ndarray:
    output = np.empty_like(design)
    for _, lo, hi in run_slices(run_id):
        output[lo:hi] = apply_response(
            design[lo:hi], dt_s=dt_s, hardware=hardware, delay_s=delay_s
        )
    return output


def dense_design(cache: dict, hardware: str, delay_s: float) -> tuple[np.ndarray, np.ndarray]:
    hardware_names = list(map(str, cache["hw_names"]))
    selected = cache["hw_idx"] == hardware_names.index(hardware)
    sub = {
        key: value[selected]
        for key, value in cache.items()
        if isinstance(value, np.ndarray) and value.shape == selected.shape
    }
    design = filter_design(
        raw_design(sub, hardware), sub["run_id"], float(cache["dt_s"]),
        hardware, delay_s,
    )
    return design, selected


def sustained_idle_floor(power_pg, busy, run_id, fit_runs: set[int],
                         dt_s: float) -> tuple[float, int]:
    minimum = int(round(IDLE_MINIMUM_S / dt_s))
    settle = int(round(IDLE_SETTLE_S / dt_s))
    if not np.isclose(minimum * dt_s, IDLE_MINIMUM_S) or minimum <= settle:
        raise ValueError("Ledger timestep does not support the idle-floor rule")
    power_pg = np.asarray(power_pg, float)
    busy = np.asarray(busy, float)
    run_id = np.asarray(run_id)
    per_run = []
    for run in sorted(fit_runs):
        selected = run_id == run
        idle = busy[selected] == 0.0
        power = power_pg[selected]
        edges = np.r_[0, np.flatnonzero(idle[1:] != idle[:-1]) + 1, idle.size]
        values = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if idle[lo] and hi - lo >= minimum:
                steady = power[lo + settle:hi]
                values.extend(steady[np.isfinite(steady)])
        if values:
            per_run.append(float(np.median(values)))
    if not per_run:
        raise ValueError("Dense source data have no sustained idle intervals")
    return float(np.median(per_run)), len(per_run)


def one_second_view(design, power, run_id, fit_runs: set[int],
                    dt_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = int(round(FIT_TIMESTEP_S / dt_s))
    if factor < 1 or not np.isclose(factor * dt_s, FIT_TIMESTEP_S):
        raise ValueError("Ledger timestep must divide one second")
    design = np.asarray(design, float)
    power = np.asarray(power, float)
    run_id = np.asarray(run_id)
    x_rows, y_rows, ids = [], [], []
    for run in sorted(fit_runs):
        selected = run_id == run
        n = selected.sum() // factor * factor
        if n == 0:
            raise ValueError(f"Training run {run} has no complete one-second bin")
        x = design[selected][:n].reshape(-1, factor, design.shape[1]).mean(1)
        y_bins = power[selected][:n].reshape(-1, factor)
        count = np.isfinite(y_bins).sum(1)
        valid = count > 0
        y = np.nansum(y_bins, axis=1)[valid] / count[valid]
        x_rows.append(x[valid])
        y_rows.append(y)
        ids.append(np.full(valid.sum(), run, dtype=np.int32))
    return np.vstack(x_rows), np.concatenate(y_rows), np.concatenate(ids)


def tail_balanced_fit(design, power, run_id, fit_runs: set[int],
                      idle_floor: float) -> tuple[np.ndarray, float]:
    weights = np.zeros(power.size)
    for run in sorted(fit_runs):
        selected = run_id == run
        threshold = np.quantile(power[selected], TAIL_QUANTILE)
        groups = (
            selected & (power < threshold),
            selected & (power >= threshold),
        )
        if not all(group.any() for group in groups):
            raise ValueError(f"Training run {run} has no distinct power tail")
        for group in groups:
            weights[group] = 0.5 / group.sum()
    selected = weights > 0.0
    root = np.sqrt(weights[selected])
    coefficients = nnls_rms_scaled(
        design[selected, 1:] * root[:, None],
        (power[selected] - idle_floor) * root,
    )
    prediction = idle_floor + design[:, 1:] @ coefficients
    objective = float(np.sum(weights * (prediction - power) ** 2))
    return np.r_[idle_floor, coefficients], objective


def fit_dense_hardware(cache: dict, hardware: str,
                       fit_runs: set[int]) -> tuple[dict, np.ndarray, np.ndarray]:
    hardware_names = list(map(str, cache["hw_names"]))
    selected = cache["hw_idx"] == hardware_names.index(hardware)
    sub = {
        key: value[selected]
        for key, value in cache.items()
        if isinstance(value, np.ndarray) and value.shape == selected.shape
    }
    target_pg = np.asarray(sub["power"], float) / np.asarray(sub["tp"], float)
    idle_floor, idle_runs = sustained_idle_floor(
        target_pg, sub["busy"], sub["run_id"], fit_runs, float(cache["dt_s"])
    )
    candidates = []
    for delay_s in DELAY_CANDIDATES_S:
        design = filter_design(
            raw_design(sub, hardware), sub["run_id"], float(cache["dt_s"]),
            hardware, delay_s,
        )
        x_fit, y_fit, run_fit = one_second_view(
            design, target_pg, sub["run_id"], fit_runs, float(cache["dt_s"])
        )
        coefficients, objective = tail_balanced_fit(
            x_fit, y_fit, run_fit, fit_runs, idle_floor
        )
        candidates.append((objective, delay_s, coefficients, design))
    objective, delay_s, coefficients, design = min(
        candidates, key=lambda row: (row[0], row[1])
    )
    fit = {
        "feature_names": list(DENSE_FEATURES),
        "coefficients": coefficients.tolist(),
        "fit_run_ids": sorted(fit_runs),
        "delay_s": delay_s,
        "fit_timestep_s": FIT_TIMESTEP_S,
        "tail_quantile": TAIL_QUANTILE,
        "idle_floor_w_per_gpu": idle_floor,
        "idle_floor_source_runs": idle_runs,
        "weighted_objective": objective,
        "delay_candidates_s": list(DELAY_CANDIDATES_S),
    }
    prediction_pg = design @ coefficients
    return fit, prediction_pg, selected
