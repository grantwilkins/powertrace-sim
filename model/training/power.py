"""Fit the frozen clean v4 dense and bounded-MoE power equations."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from model.training.dense import fit_dense_hardware
from model.training.numerics import HARDWARE, nnls_rms_scaled, run_slices

MOE_FEATURES = (
    "idle",
    "multi_gpu_floor",
    "logical_memory_util_lag_250ms",
    "duty_sqrt_exact_compute_util",
    "engine_iterations_rate",
    "log_decode_batch",
)


def load_power_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        cache = {key: data[key] for key in data.files}
    required = {
        "run_id", "model_names", "model_idx", "family_names", "family_idx",
        "hw_names", "hw_idx", "role_names", "role_idx", "tp", "rate",
        "power", "busy", "w_read", "kv_read", "kv_write",
        "gemm_flops_rate", "attn_flops_rate", "engine_iterations_rate", "batch",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"power cache missing fields: {missing}")
    if float(cache["dt_s"]) != 0.25:
        raise ValueError("clean power fitting requires 250 ms cache bins")
    return cache


def _metadata(cache: dict) -> dict[int, dict[str, object]]:
    models = np.asarray(cache["model_names"])[cache["model_idx"]].astype(str)
    families = np.asarray(cache["family_names"])[cache["family_idx"]].astype(str)
    hardware = np.asarray(cache["hw_names"])[cache["hw_idx"]].astype(str)
    legacy_roles = np.asarray(cache["role_names"])[cache["role_idx"]].astype(str)
    output = {}
    for run, lo, hi in run_slices(cache["run_id"]):
        for values in (models, families, hardware, legacy_roles, cache["tp"], cache["rate"]):
            if np.unique(values[lo:hi]).size != 1:
                raise ValueError(f"run {run} has non-constant metadata")
        role = (
            "transfer_twin" if legacy_roles[lo] == "holdout_twin"
            else "stress_rate4" if float(cache["rate"][lo]) == 4.0
            else "train_source" if float(cache["rate"][lo]) < 4.0
            else None
        )
        if role is None:
            raise ValueError(f"run {run} has unsupported source rate")
        output[run] = {
            "model": models[lo], "family": families[lo],
            "hardware": hardware[lo], "tp": int(cache["tp"][lo]), "role": role,
        }
    return output


def _training_runs(metadata: dict[int, dict], family_prefix: str) -> set[int]:
    return {
        run for run, row in metadata.items()
        if str(row["family"]).startswith(family_prefix)
        and row["role"] == "train_source"
    }


def _lag_by_run(values: np.ndarray, run_id: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for _, lo, hi in run_slices(run_id):
        output[lo] = values[lo]
        output[lo + 1:hi] = values[lo:hi - 1]
    return output


def _moe_design(cache: dict, selected: np.ndarray) -> tuple[np.ndarray, list[str]]:
    tp = np.asarray(cache["tp"][selected], dtype=float)
    run_id = cache["run_id"][selected]
    busy = np.clip(np.asarray(cache["busy"][selected], dtype=float), 0.0, 1.0)
    memory = _lag_by_run(sum(
        np.asarray(cache[key][selected], dtype=float)
        for key in ("w_read", "kv_read", "kv_write")
    ), run_id) / (tp * HARDWARE["A100"]["hbm_bandwidth_bytes_s"])
    compute = sum(
        np.asarray(cache[key][selected], dtype=float)
        for key in ("gemm_flops_rate", "attn_flops_rate")
    ) / (tp * HARDWARE["A100"]["compute_peak_flops_s"])
    design = np.column_stack((
        np.ones(tp.size),
        (tp > 1).astype(float),
        memory,
        np.sqrt(busy * np.clip(compute, 0.0, None)),
        np.asarray(cache["engine_iterations_rate"][selected], dtype=float) / 1000.0,
        np.log1p(np.asarray(cache["batch"][selected], dtype=float)),
    ))
    names = list(MOE_FEATURES)
    if np.all(tp > 1):
        design = np.delete(design, 1, axis=1)
        names.pop(1)
    return design, names


def _fit_run_balanced(
    design: np.ndarray, target: np.ndarray, run_id: np.ndarray,
    fit_runs: set[int],
) -> np.ndarray:
    weights = np.zeros(target.size)
    for run in sorted(fit_runs):
        rows = (run_id == run) & np.isfinite(target)
        if not rows.any():
            raise ValueError(f"training run {run} has no finite power")
        weights[rows] = 1.0 / rows.sum()
    selected = weights > 0.0
    root = np.sqrt(weights[selected])
    return nnls_rms_scaled(
        design[selected] * root[:, None], target[selected] * root,
    )


def fit_power_surfaces(cache: dict) -> dict[str, object]:
    metadata = _metadata(cache)
    dense_runs = _training_runs(metadata, "dense")
    moe_runs = _training_runs(metadata, "moe")
    dense = {}
    hardware_names = list(map(str, cache["hw_names"]))
    for hardware in hardware_names:
        selected = cache["hw_idx"] == hardware_names.index(hardware)
        runs = dense_runs & set(map(int, np.unique(cache["run_id"][selected])))
        dense[hardware], _, returned = fit_dense_hardware(cache, hardware, runs)
        if not np.array_equal(selected, returned):
            raise AssertionError("dense hardware selection changed during fitting")
    models = np.asarray(cache["model_names"])[cache["model_idx"]].astype(str)
    per_model = {}
    for model in ("gpt-oss-20b", "gpt-oss-120b"):
        selected = models == model
        design, names = _moe_design(cache, selected)
        tp = np.asarray(cache["tp"][selected], dtype=float)
        target = np.asarray(cache["power"][selected], dtype=float) / tp
        run_id = cache["run_id"][selected]
        runs = moe_runs & set(map(int, np.unique(run_id)))
        coefficients = _fit_run_balanced(design, target, run_id, runs)
        per_model[model] = {
            "feature_names": names,
            "coefficients": coefficients.tolist(),
            "fit_run_ids": sorted(runs),
            "supported_tp": sorted(set(map(int, tp))),
        }
    return {
        "schema_version": "clean-separated-power-surfaces-v4",
        "training_policy": (
            "frozen clean v4: per-GPU one-second run/p90-tail-balanced dense "
            "NNLS and separate run-balanced architecture-specific MoE NNLS"
        ),
        "dense": dense,
        "moe": {
            "hardware": "A100",
            "models": ["gpt-oss-20b", "gpt-oss-120b"],
            "per_model": per_model,
        },
    }

