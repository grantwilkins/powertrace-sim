"""Clean v4 dense and support-bounded MoE power equations."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from model.power.response import apply_response
from model.timing.iteration import HARDWARE_PROFILES

DENSE_FEATURES = (
    "idle_floor",
    "active_weight_fraction",
    "compute_util",
    "duty_sqrt_memory_util",
)
MOE_FEATURES = (
    "idle",
    "multi_gpu_floor",
    "logical_memory_util_lag_250ms",
    "duty_sqrt_exact_compute_util",
    "engine_iterations_rate",
    "log_decode_batch",
)


def _array(ledger: Mapping[str, object], key: str) -> np.ndarray:
    value = np.asarray(ledger[key], dtype=float)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"ledger channel {key!r} must be a finite vector")
    return value


def dense_design(
    ledger: Mapping[str, object], *, arch: Mapping[str, object],
    hardware: str, tp: int,
) -> np.ndarray:
    profile = HARDWARE_PROFILES[hardware]
    busy = np.clip(_array(ledger, "busy"), 0.0, 1.0)
    weight_fraction = np.clip(
        float(arch["w_bytes"]) / (tp * profile["hbm_capacity"]), 0.0, 1.0
    )
    compute = sum(_array(ledger, key) for key in (
        "prefill_gemm_flops_rate", "decode_gemm_flops_rate",
        "prefill_attn_flops_rate", "decode_attn_flops_rate",
    )) / (tp * profile["peak_flops_s"])
    memory = sum(_array(ledger, key) for key in (
        "w_read", "prefill_attn_bytes_rate", "decode_attn_bytes_rate",
    )) / (tp * profile["hbm_bytes_s"])
    return np.column_stack((
        np.ones(busy.size),
        busy * weight_fraction,
        np.clip(compute, 0.0, None),
        np.sqrt(busy * np.clip(memory, 0.0, None)),
    ))


def _lag(values: np.ndarray) -> np.ndarray:
    return np.r_[values[0], values[:-1]] if values.size else values.copy()


def moe_design(
    ledger: Mapping[str, object], *, hardware: str, tp: int,
    feature_names: list[str],
) -> np.ndarray:
    profile = HARDWARE_PROFILES[hardware]
    busy = np.clip(_array(ledger, "busy"), 0.0, 1.0)
    memory = _lag(sum(_array(ledger, key) for key in (
        "w_read", "kv_read", "kv_write",
    ))) / profile["hbm_bytes_s"]
    compute = sum(_array(ledger, key) for key in (
        "gemm_flops_rate", "attn_flops_rate",
    )) / (tp * profile["peak_flops_s"])
    columns = {
        "idle": np.ones(busy.size),
        "multi_gpu_floor": np.full(busy.size, float(tp > 1)),
        "logical_memory_util_lag_250ms": memory / tp,
        "duty_sqrt_exact_compute_util": np.sqrt(busy * np.clip(compute, 0.0, None)),
        "engine_iterations_rate": _array(ledger, "engine_iterations_rate") / 1000.0,
        "log_decode_batch": np.log1p(_array(ledger, "batch")),
    }
    unknown = set(feature_names) - columns.keys()
    if unknown:
        raise ValueError(f"unknown MoE power features: {sorted(unknown)}")
    return np.column_stack([columns[name] for name in feature_names])


def predict_power(
    ledger: Mapping[str, object], *, arch: Mapping[str, object], model: str,
    hardware: str, tp: int, artifact: Mapping[str, object], dt_s: float,
) -> dict[str, object]:
    family = str(arch["family"])
    if family.startswith("dense"):
        fit = artifact["power"]["dense"][hardware]
        if tuple(fit["feature_names"]) != DENSE_FEATURES:
            raise ValueError("dense artifact feature contract mismatch")
        raw = dense_design(ledger, arch=arch, hardware=hardware, tp=tp)
        design = apply_response(
            raw, dt_s=dt_s, hardware=hardware, delay_s=float(fit["delay_s"])
        )
        surface = "dense"
    else:
        fits = artifact["power"]["moe"]["per_model"]
        if model not in fits:
            raise ValueError(f"no MoE power surface for {model!r}")
        fit = fits[model]
        design = moe_design(
            ledger, hardware=hardware, tp=tp,
            feature_names=list(fit["feature_names"]),
        )
        surface = f"moe:{model}"
    coefficients = np.asarray(fit["coefficients"], dtype=float)
    if design.shape[1] != coefficients.size:
        raise ValueError("power artifact coefficient contract mismatch")
    contributions_pg = design * coefficients
    mean_gpu = contributions_pg.sum(axis=1)
    return {
        "surface": surface,
        "feature_names": list(fit["feature_names"]),
        "contributions_node_w": contributions_pg * tp,
        "mean_gpu_power_w": mean_gpu,
        "node_gpu_power_w": mean_gpu * tp,
    }


def idle_node_power(
    *, arch: Mapping[str, object], model: str, hardware: str, tp: int,
    artifact: Mapping[str, object],
) -> float:
    """Return the calibrated GPU-only node power for an empty request stream."""
    family = str(arch["family"])
    if family.startswith("dense"):
        fit = artifact["power"]["dense"][hardware]
        values = dict(zip(fit["feature_names"], fit["coefficients"]))
        per_gpu = float(values["idle_floor"])
    else:
        fit = artifact["power"]["moe"]["per_model"][model]
        values = dict(zip(fit["feature_names"], fit["coefficients"]))
        per_gpu = float(values["idle"])
        if tp > 1:
            per_gpu += float(values.get("multi_gpu_floor", 0.0))
    return per_gpu * tp
