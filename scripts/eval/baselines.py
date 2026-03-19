#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Effective DGX GPU-only node TDP contribution targets (W), excluding non-GPU overhead.
# These values are intentionally lower than 8x vendor chip TDP and reflect platform-level planning targets.
DEFAULT_NODE_GPU_TDP_W = {
    "A100": 3300.0,
    "H100": 4600.0,
}
DEFAULT_NUM_GPUS_PER_NODE = 8
DEFAULT_NON_GPU_POWER_W = 1000.0

# Effective DGX GPU-only node active targets (W) for TP=4 baseline calibration.
DEFAULT_NODE_GPU_ACTIVE_W = {
    "A100": 1500.0,
    "H100": 2000.0,
}

CONFIG_HW_RE = re.compile(r"^.+_(A100|H100)_tp\d+$")
CONFIG_HW_TP_RE = re.compile(r"^.+_(A100|H100)_tp(\d+)$")


def _load_pipeline_generators() -> Tuple[object, object]:
    from scripts.eval.pipeline_utils import (
        generate_gmm_bigru_trace,
        generate_gmm_bigru_trace_ar1_thresholded,
    )

    return generate_gmm_bigru_trace, generate_gmm_bigru_trace_ar1_thresholded


def _extract_seed(config: Mapping[str, object], rng: Optional[np.random.Generator]) -> Optional[int]:
    if "seed" in config:
        try:
            return int(config["seed"])
        except Exception:
            return None
    if rng is None:
        return None
    return int(rng.integers(0, 2**31 - 1))


def _resolve_hardware(config: Mapping[str, object]) -> str:
    if "hardware" in config:
        hw = str(config["hardware"]).strip().upper()
        if hw in DEFAULT_NODE_GPU_TDP_W:
            return hw
    config_id = str(config.get("config_id", "")).strip()
    match = CONFIG_HW_RE.match(config_id)
    if match:
        return match.group(1)
    raise ValueError(
        "Unable to resolve hardware (expected config['hardware'] or config_id suffix _A100/_H100)."
    )


def _resolve_tp(config: Mapping[str, object]) -> int:
    if "tp" in config:
        try:
            tp = int(config["tp"])
            if tp > 0:
                return tp
        except Exception:
            pass
    config_id = str(config.get("config_id", "")).strip()
    match = CONFIG_HW_TP_RE.match(config_id)
    if match:
        tp = int(match.group(2))
        if tp > 0:
            return tp
    raise ValueError("Unable to resolve TP (expected config['tp'] or config_id suffix _tpX).")


def _resolve_tdp_node_w(config: Mapping[str, object]) -> float:
    if "tdp_node" in config:
        return float(config["tdp_node"])
    hardware = _resolve_hardware(config)
    n_gpus = int(config.get("n_gpus_per_node", DEFAULT_NUM_GPUS_PER_NODE))
    gpu_node_tdp = float(config.get("tdp_gpu_node_w", DEFAULT_NODE_GPU_TDP_W[hardware]))
    if "gpu_tdp_w" in config:
        gpu_node_tdp = float(config["gpu_tdp_w"]) * float(n_gpus)
    else:
        gpu_node_tdp = gpu_node_tdp * (float(n_gpus) / float(DEFAULT_NUM_GPUS_PER_NODE))
    non_gpu_power = float(config.get("non_gpu_power_w", DEFAULT_NON_GPU_POWER_W))
    return float(gpu_node_tdp + non_gpu_power)


def _safe_percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute percentile on empty/invalid array.")
    return float(np.percentile(arr, float(q)))


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _fit_affine_map(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size or x_arr.size == 0:
        raise ValueError("Affine fit inputs must have same non-zero length.")
    A = np.stack([x_arr, np.ones_like(x_arr)], axis=1)
    if np.max(x_arr) - np.min(x_arr) < 1e-12:
        return 0.0, float(np.mean(y_arr))
    sol, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
    a = float(sol[0])
    b = float(sol[1])
    if (not np.isfinite(a)) or (not np.isfinite(b)):
        return 0.0, float(np.mean(y_arr))
    return a, b


SPLITWISE_STYLE_LUT_V1 = "splitwise_style_lut_v1"
SPLITWISE_REMOVED_MESSAGE = (
    "splitwise_lut was removed during the strict-emulation overhaul; use splitwise_strict"
)
SPLITWISE_ALLOWED_STYLE_LUT_MODES = {
    SPLITWISE_STYLE_LUT_V1,
    "splitwise_emulation_strict_v1",
    "legacy_scheduler_strict_v1",
    "strict",
    "splitwise_strict",
    "dgx_fixed_targets_v1",
    "train_phase_matched_v1",
    "train_phase_matched",
}
SPLITWISE_LLAMA_70B_FAMILY = frozenset({"llama2-70b", "llama-3-70b"})


def _normalize_splitwise_model_name(model: object) -> str:
    raw = str(model).strip().lower().replace("_", "-").replace(" ", "")
    aliases = {
        "llama-2-70b": "llama2-70b",
        "llama2-70b": "llama2-70b",
        "llama3-70b": "llama-3-70b",
        "llama-3-70b": "llama-3-70b",
    }
    return aliases.get(raw, raw)


def normalize_splitwise_style_lut_mode(mode: object) -> str:
    mode_raw = str(mode).strip().lower()
    if mode_raw in SPLITWISE_ALLOWED_STYLE_LUT_MODES:
        return SPLITWISE_STYLE_LUT_V1
    return str(mode)


def _splitwise_model_family(model: object) -> str:
    norm = _normalize_splitwise_model_name(model)
    if norm in SPLITWISE_LLAMA_70B_FAMILY:
        return "llama-70b-family"
    return norm


def _normalize_splitwise_hardware_name(hardware: object) -> str:
    return str(hardware).strip().lower().replace("_", "-")


def _safe_row_float(row: Mapping[str, object], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _group_median_by_batch_tokens(
    rows: list[Dict[str, object]],
    value_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    grouped: Dict[float, list[float]] = {}
    for row in rows:
        x = float(row["batch_tokens"])
        y = float(row[value_key])
        if (not np.isfinite(x)) or (not np.isfinite(y)):
            continue
        grouped.setdefault(float(x), []).append(float(y))
    if not grouped:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    x_sorted = np.asarray(sorted(grouped.keys()), dtype=np.float64)
    y_sorted = np.asarray(
        [float(np.median(np.asarray(grouped[float(x)], dtype=np.float64))) for x in x_sorted],
        dtype=np.float64,
    )
    return x_sorted, y_sorted


def _interp_or_extrapolate(
    x_support: np.ndarray,
    y_support: np.ndarray,
    x_query: float,
    *,
    extrapolate: bool,
) -> float:
    x_arr = np.asarray(x_support, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y_support, dtype=np.float64).reshape(-1)
    if x_arr.size == 0 or y_arr.size != x_arr.size:
        raise ValueError("invalid support for interpolation")
    xq = float(x_query)
    if x_arr.size == 1:
        return float(y_arr[0])
    if xq <= float(x_arr[0]):
        if not extrapolate:
            return float(y_arr[0])
        dx = float(x_arr[1] - x_arr[0])
        if abs(dx) < 1e-12:
            return float(y_arr[0])
        slope = float((y_arr[1] - y_arr[0]) / dx)
        return float(y_arr[0] + slope * (xq - float(x_arr[0])))
    if xq >= float(x_arr[-1]):
        if not extrapolate:
            return float(y_arr[-1])
        dx = float(x_arr[-1] - x_arr[-2])
        if abs(dx) < 1e-12:
            return float(y_arr[-1])
        slope = float((y_arr[-1] - y_arr[-2]) / dx)
        return float(y_arr[-1] + slope * (xq - float(x_arr[-1])))
    return float(np.interp(xq, x_arr, y_arr))


def _choose_splitwise_model(rows: list[Dict[str, object]], preferred_model: str) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        model = str(row["model_norm"])
        counts[model] = counts.get(model, 0) + 1
    if preferred_model in counts:
        return preferred_model
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    if not ranked:
        raise ValueError("unable to choose splitwise source model")
    return str(ranked[0][0])


def _read_splitwise_perf_rows(perf_model_csv: str) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    with open(perf_model_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            try:
                model_raw = str(raw_row.get("model", "")).strip()
                hardware_raw = str(raw_row.get("hardware", "")).strip()
                tensor_parallel = int(float(raw_row.get("tensor_parallel", "nan")))
            except Exception:
                continue
            prompt_size = _safe_row_float(raw_row, "prompt_size")
            batch_size = _safe_row_float(raw_row, "batch_size")
            token_size = _safe_row_float(raw_row, "token_size")
            prompt_time_ms = _safe_row_float(raw_row, "prompt_time")
            token_time_ms = _safe_row_float(raw_row, "token_time")
            peak_power = _safe_row_float(raw_row, "peak_power")
            average_power = _safe_row_float(raw_row, "average_power")
            batch_tokens = float(prompt_size) * float(batch_size)
            rows.append(
                {
                    "model_raw": model_raw,
                    "model_norm": _normalize_splitwise_model_name(model_raw),
                    "model_family": _splitwise_model_family(model_raw),
                    "hardware": _normalize_splitwise_hardware_name(hardware_raw),
                    "tensor_parallel": int(tensor_parallel),
                    "prompt_size": float(prompt_size),
                    "batch_size": float(batch_size),
                    "token_size": float(token_size),
                    "batch_tokens": float(batch_tokens),
                    "prompt_time_s": float(prompt_time_ms / 1000.0),
                    "token_time_s": float(token_time_ms / 1000.0),
                    "peak_power": float(peak_power),
                    "average_power": float(average_power),
                }
            )
    if len(rows) == 0:
        raise ValueError(f"perf model csv has no readable rows: {perf_model_csv}")
    return rows


def _resolve_splitwise_source_rows(
    perf_rows: list[Dict[str, object]],
    *,
    requested_model: str,
    requested_hardware: str,
    requested_tp: int,
) -> Tuple[list[Dict[str, object]], Dict[str, object], list[Dict[str, object]]]:
    requested_model_norm = _normalize_splitwise_model_name(requested_model)
    requested_family = _splitwise_model_family(requested_model)
    requested_hw_norm = _normalize_splitwise_hardware_name(requested_hardware)
    requested_tp_int = int(requested_tp)

    hardware_rows = [
        row for row in perf_rows if str(row["hardware"]) == requested_hw_norm
    ]
    if len(hardware_rows) == 0:
        raise ValueError(
            f"No LUT rows for hardware={requested_hw_norm} in perf_model.csv"
        )

    exact_rows = [
        row
        for row in hardware_rows
        if str(row["model_norm"]) == requested_model_norm
        and int(row["tensor_parallel"]) == requested_tp_int
    ]
    if exact_rows:
        return (
            exact_rows,
            {
                "splitwise_source_resolved_model": str(
                    _choose_splitwise_model(exact_rows, requested_model_norm)
                ),
                "splitwise_source_resolved_hardware": str(requested_hw_norm),
                "splitwise_source_resolved_tp": int(requested_tp_int),
                "splitwise_source_match_status": "exact",
            },
            [
                row
                for row in hardware_rows
                if str(row["model_family"]) == requested_family
            ],
        )

    family_rows = [
        row for row in hardware_rows if str(row["model_family"]) == requested_family
    ]
    if len(family_rows) == 0:
        raise ValueError(
            f"No LUT rows for model family={requested_family} on hardware={requested_hw_norm}"
        )

    family_tp_rows = [
        row
        for row in family_rows
        if int(row["tensor_parallel"]) == requested_tp_int
    ]
    if family_tp_rows:
        resolved_model = _choose_splitwise_model(family_tp_rows, requested_model_norm)
        selected_rows = [
            row for row in family_tp_rows if str(row["model_norm"]) == resolved_model
        ]
        return (
            selected_rows,
            {
                "splitwise_source_resolved_model": str(resolved_model),
                "splitwise_source_resolved_hardware": str(requested_hw_norm),
                "splitwise_source_resolved_tp": int(requested_tp_int),
                "splitwise_source_match_status": "family_model_fallback",
            },
            family_rows,
        )

    available_tps = sorted({int(row["tensor_parallel"]) for row in family_rows})
    resolved_tp = min(
        available_tps,
        key=lambda tp_val: (abs(int(tp_val) - requested_tp_int), int(tp_val)),
    )
    nearest_rows = [
        row for row in family_rows if int(row["tensor_parallel"]) == int(resolved_tp)
    ]
    resolved_model = _choose_splitwise_model(nearest_rows, requested_model_norm)
    selected_rows = [
        row for row in nearest_rows if str(row["model_norm"]) == resolved_model
    ]
    match_status = (
        "nearest_tp_fallback"
        if resolved_model == requested_model_norm
        else "family_model_and_nearest_tp_fallback"
    )
    return (
        selected_rows,
        {
            "splitwise_source_resolved_model": str(resolved_model),
            "splitwise_source_resolved_hardware": str(requested_hw_norm),
            "splitwise_source_resolved_tp": int(resolved_tp),
            "splitwise_source_match_status": str(match_status),
        },
        family_rows,
    )


def _filter_timing_rows(rows: list[Dict[str, object]]) -> list[Dict[str, object]]:
    out: list[Dict[str, object]] = []
    for row in rows:
        prompt_time_s = float(row["prompt_time_s"])
        token_time_s = float(row["token_time_s"])
        batch_tokens = float(row["batch_tokens"])
        batch_size = float(row["batch_size"])
        if (
            np.isfinite(prompt_time_s)
            and np.isfinite(token_time_s)
            and np.isfinite(batch_tokens)
            and np.isfinite(batch_size)
            and prompt_time_s > 0.0
            and token_time_s > 0.0
            and batch_tokens > 0.0
            and batch_size > 0.0
        ):
            out.append(row)
    return out


def _filter_power_rows(rows: list[Dict[str, object]]) -> list[Dict[str, object]]:
    out: list[Dict[str, object]] = []
    for row in rows:
        prompt_time_s = float(row["prompt_time_s"])
        token_time_s = float(row["token_time_s"])
        batch_tokens = float(row["batch_tokens"])
        peak = float(row["peak_power"])
        avg = float(row["average_power"])
        if not (
            np.isfinite(prompt_time_s)
            and np.isfinite(token_time_s)
            and prompt_time_s > 0.0
            and token_time_s > 0.0
        ):
            continue
        if not (
            np.isfinite(batch_tokens)
            and batch_tokens > 0.0
            and np.isfinite(peak)
            and np.isfinite(avg)
            and avg > 0.0
            and avg <= peak
            and peak <= 2.0
        ):
            continue
        out.append(row)
    return out


def _build_power_support(
    primary_rows: list[Dict[str, object]],
    family_rows: list[Dict[str, object]],
) -> Dict[str, object]:
    primary_valid = _filter_power_rows(primary_rows)
    primary_x, primary_avg = _group_median_by_batch_tokens(primary_valid, "average_power")
    primary_peak_x, primary_peak = _group_median_by_batch_tokens(primary_valid, "peak_power")
    if primary_x.size >= 3 and primary_peak_x.size >= 3 and np.array_equal(primary_x, primary_peak_x):
        avg_support = np.asarray([float(row["average_power"]) for row in primary_valid], dtype=np.float64)
        idle_ratio = float(np.min(avg_support))
        decode_ratio = float(np.median(avg_support))
        prefill_ratio = float(np.median(np.asarray([float(row["peak_power"]) for row in primary_valid], dtype=np.float64)))
        return {
            "power_support_lookup_mode": "grouped",
            "power_support_batch_tokens": primary_x.astype(np.float64),
            "power_support_average_ratio": primary_avg.astype(np.float64),
            "power_support_peak_ratio": primary_peak.astype(np.float64),
            "power_support_idle_ratio": float(max(1e-6, idle_ratio)),
            "power_support_decode_ratio": float(max(idle_ratio, decode_ratio)),
            "power_support_prefill_ratio": float(max(max(idle_ratio, decode_ratio), prefill_ratio)),
            "power_support_quality_flag": "clean",
            "power_support_status": "slice_grouped",
            "power_support_points": int(primary_x.size),
        }

    family_valid = _filter_power_rows(family_rows)
    family_x, family_avg = _group_median_by_batch_tokens(family_valid, "average_power")
    family_peak_x, family_peak = _group_median_by_batch_tokens(family_valid, "peak_power")
    if family_x.size >= 3 and family_peak_x.size >= 3 and np.array_equal(family_x, family_peak_x):
        avg_support = np.asarray([float(row["average_power"]) for row in family_valid], dtype=np.float64)
        idle_ratio = float(np.min(avg_support))
        decode_ratio = float(np.median(avg_support))
        prefill_ratio = float(np.median(np.asarray([float(row["peak_power"]) for row in family_valid], dtype=np.float64)))
        return {
            "power_support_lookup_mode": "grouped",
            "power_support_batch_tokens": family_x.astype(np.float64),
            "power_support_average_ratio": family_avg.astype(np.float64),
            "power_support_peak_ratio": family_peak.astype(np.float64),
            "power_support_idle_ratio": float(max(1e-6, idle_ratio)),
            "power_support_decode_ratio": float(max(idle_ratio, decode_ratio)),
            "power_support_prefill_ratio": float(max(max(idle_ratio, decode_ratio), prefill_ratio)),
            "power_support_quality_flag": "family_fallback",
            "power_support_status": "family_grouped",
            "power_support_points": int(family_x.size),
        }

    scalar_rows = family_valid if len(family_valid) > 0 else primary_valid
    if len(scalar_rows) == 0:
        raise ValueError("No valid power support rows available for Splitwise strict emulation")
    avg_arr = np.asarray([float(row["average_power"]) for row in scalar_rows], dtype=np.float64)
    peak_arr = np.asarray([float(row["peak_power"]) for row in scalar_rows], dtype=np.float64)
    decode_ratio = float(np.median(avg_arr))
    prefill_ratio = float(np.median(peak_arr))
    idle_ratio = float(min(decode_ratio, 0.7 * decode_ratio))
    return {
        "power_support_lookup_mode": "scalar",
        "power_support_batch_tokens": np.zeros((0,), dtype=np.float64),
        "power_support_average_ratio": np.zeros((0,), dtype=np.float64),
        "power_support_peak_ratio": np.zeros((0,), dtype=np.float64),
        "power_support_idle_ratio": float(max(1e-6, idle_ratio)),
        "power_support_decode_ratio": float(max(idle_ratio, decode_ratio)),
        "power_support_prefill_ratio": float(max(max(idle_ratio, decode_ratio), prefill_ratio)),
        "power_support_quality_flag": "scalar_fallback",
        "power_support_status": "scalar_fallback",
        "power_support_points": int(len(scalar_rows)),
    }


def build_splitwise_style_lut_params(
    config_id: str,
    perf_model_csv: str,
    train_power_flat: np.ndarray,
    *,
    splitwise_source_model: str = "llama-3-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: Optional[int] = None,
    splitwise_style_lut_mode: str = SPLITWISE_STYLE_LUT_V1,
    n_gpus_per_node: int = DEFAULT_NUM_GPUS_PER_NODE,
    per_gpu_tdp_cap_w: Optional[float] = None,
    target_idle_node_gpu_w: Optional[float] = None,
    target_decode_node_gpu_w: Optional[float] = None,
    target_prefill_node_gpu_w: Optional[float] = None,
) -> Dict[str, object]:
    del train_power_flat
    del target_idle_node_gpu_w
    del target_decode_node_gpu_w
    del target_prefill_node_gpu_w

    mode_raw = str(splitwise_style_lut_mode).strip().lower()
    if mode_raw not in SPLITWISE_ALLOWED_STYLE_LUT_MODES:
        raise ValueError(
            f"{SPLITWISE_REMOVED_MESSAGE}; requested LUT mode '{splitwise_style_lut_mode}' is no longer supported"
        )

    try:
        target_hardware = _resolve_hardware({"config_id": str(config_id)})
    except Exception:
        target_hardware = "H100"
    try:
        target_tp = int(_resolve_tp({"config_id": str(config_id)}))
    except Exception:
        target_tp = 4
    resolved_requested_tp = int(splitwise_source_tp) if splitwise_source_tp is not None else int(target_tp)
    if resolved_requested_tp <= 0:
        raise ValueError("splitwise_source_tp must be positive")

    perf_rows = _read_splitwise_perf_rows(perf_model_csv)
    source_rows, resolved_meta, family_rows = _resolve_splitwise_source_rows(
        perf_rows,
        requested_model=str(splitwise_source_model),
        requested_hardware=str(splitwise_source_hardware),
        requested_tp=int(resolved_requested_tp),
    )
    timing_rows = _filter_timing_rows(source_rows)
    if len(timing_rows) == 0:
        raise ValueError(
            "No valid timing rows available for the resolved Splitwise source slice"
        )
    timing_batch_tokens, prompt_time_support_s = _group_median_by_batch_tokens(
        timing_rows, "prompt_time_s"
    )
    token_batch_tokens, token_time_support_s = _group_median_by_batch_tokens(
        timing_rows, "token_time_s"
    )
    if timing_batch_tokens.size == 0 or token_batch_tokens.size == 0:
        raise ValueError("Resolved Splitwise timing support is empty after grouping")
    if not np.array_equal(timing_batch_tokens, token_batch_tokens):
        raise ValueError("Prompt and token timing support batch_tokens do not align")

    power_support = _build_power_support(source_rows, family_rows)
    n_gpus = int(max(1, n_gpus_per_node))
    target_tp = int(max(1, min(int(target_tp), n_gpus)))
    if per_gpu_tdp_cap_w is None:
        per_gpu_tdp_cap_w = float(
            DEFAULT_NODE_GPU_TDP_W[str(target_hardware)] / float(DEFAULT_NUM_GPUS_PER_NODE)
        )

    return {
        "config_id": str(config_id),
        "splitwise_source_model": str(splitwise_source_model),
        "splitwise_source_hardware": str(_normalize_splitwise_hardware_name(splitwise_source_hardware)),
        "splitwise_source_tp": int(resolved_requested_tp),
        "splitwise_style_lut_mode": SPLITWISE_STYLE_LUT_V1,
        "target_hardware": str(target_hardware),
        "target_tp": int(target_tp),
        "n_gpus_per_node": int(n_gpus),
        "per_gpu_tdp_cap_w": float(per_gpu_tdp_cap_w),
        "timing_support_batch_tokens": timing_batch_tokens.astype(np.float64),
        "timing_support_prompt_time_s": prompt_time_support_s.astype(np.float64),
        "timing_support_token_time_s": token_time_support_s.astype(np.float64),
        "timing_support_max_batch_size": int(
            max(1, int(np.max(np.asarray([float(row["batch_size"]) for row in timing_rows], dtype=np.float64))))
        ),
        "timing_support_max_batch_tokens": float(
            np.max(np.asarray([float(row["batch_tokens"]) for row in timing_rows], dtype=np.float64))
        ),
        "timing_support_min_batch_tokens": float(
            np.min(np.asarray([float(row["batch_tokens"]) for row in timing_rows], dtype=np.float64))
        ),
        "timing_support_mode": "grouped_median_interp_extrap",
        "scheduler_defaults_policy": "prompt_biased_preemptive_fifo",
        "scheduler_defaults_max_batch_size": int(
            max(1, int(np.max(np.asarray([float(row["batch_size"]) for row in timing_rows], dtype=np.float64))))
        ),
        "scheduler_defaults_max_batch_tokens": float(
            np.max(np.asarray([float(row["batch_tokens"]) for row in timing_rows], dtype=np.float64))
        ),
        "scheduler_defaults_max_preemptions": 4,
        "scheduler_defaults_max_contiguous_decode_iters": 16,
        "scheduler_defaults_prompt_interrupt_mode": "iteration_boundary",
        "phase_detection_note": (
            "Continuous-time request-centric emulation; no A_raw/delta_A_t phase labeling."
        ),
        "decode_occupancy_note": (
            "Decode uses average_power LUT ratios at the batch level; within-batch occupancy is not modeled."
        ),
        "splitwise_power_quality_flag": str(power_support["power_support_quality_flag"]),
        "splitwise_power_support_status": str(power_support["power_support_status"]),
        "splitwise_power_support_points": int(power_support["power_support_points"]),
        "splitwise_scheduler_policy": "prompt_biased_preemptive_fifo",
        **resolved_meta,
        **power_support,
    }


def build_splitwise_style_lut_trace_params(
    config_id: str,
    perf_model_csv: str,
    train_power_flat: np.ndarray,
    **kwargs: object,
) -> Dict[str, object]:
    return build_splitwise_style_lut_params(
        config_id=config_id,
        perf_model_csv=perf_model_csv,
        train_power_flat=train_power_flat,
        **kwargs,
    )


def _parse_splitwise_requests(requests: object) -> list[Dict[str, float]]:
    out: list[Dict[str, float]] = []
    if requests is None:
        return out
    for idx, req in enumerate(requests):
        if not isinstance(req, Mapping):
            raise ValueError(f"request[{idx}] must be a mapping")
        try:
            arrival_time = float(req["arrival_time"])
            input_tokens = float(req["input_tokens"])
            output_tokens = float(req["output_tokens"])
        except Exception as exc:
            raise ValueError(f"request[{idx}] missing required fields") from exc
        if not (
            np.isfinite(arrival_time)
            and np.isfinite(input_tokens)
            and np.isfinite(output_tokens)
        ):
            continue
        out.append(
            {
                "arrival_time": float(max(0.0, arrival_time)),
                "input_tokens": float(max(0.0, input_tokens)),
                "output_tokens": float(max(0.0, output_tokens)),
                "request_index": float(idx),
            }
        )
    out.sort(key=lambda row: (float(row["arrival_time"]), int(row["request_index"])))
    return out


def _predict_splitwise_timing_s(
    lut_params: Mapping[str, object],
    *,
    batch_tokens: float,
    phase_kind: str,
    generation_meta: Dict[str, object],
) -> float:
    x_support = np.asarray(lut_params["timing_support_batch_tokens"], dtype=np.float64).reshape(-1)
    batch_tokens_safe = float(max(1.0, batch_tokens))
    if x_support.size == 0:
        raise ValueError("missing timing support")
    if (
        batch_tokens_safe < float(np.min(x_support)) - 1e-12
        or batch_tokens_safe > float(np.max(x_support)) + 1e-12
    ):
        generation_meta["splitwise_extrapolation_events"] = int(
            generation_meta.get("splitwise_extrapolation_events", 0)
        ) + 1
    if str(phase_kind) == "decode":
        y_support = np.asarray(lut_params["timing_support_token_time_s"], dtype=np.float64).reshape(-1)
        duration = _interp_or_extrapolate(x_support, y_support, batch_tokens_safe, extrapolate=True)
    else:
        y_support = np.asarray(lut_params["timing_support_prompt_time_s"], dtype=np.float64).reshape(-1)
        duration = _interp_or_extrapolate(x_support, y_support, batch_tokens_safe, extrapolate=True)
        if str(phase_kind) == "mixed":
            duration *= 1.1
    return float(max(1e-6, duration))


def _lookup_splitwise_power_ratio(
    lut_params: Mapping[str, object],
    *,
    batch_tokens: float,
    batch_size: int,
    num_prompt_tasks: int,
    num_decode_tasks: int,
    phase_kind: str,
    generation_meta: Dict[str, object],
) -> float:
    idle_ratio = float(lut_params["power_support_idle_ratio"])
    decode_ratio = float(lut_params["power_support_decode_ratio"])
    prefill_ratio = float(lut_params["power_support_prefill_ratio"])
    lookup_mode = str(lut_params["power_support_lookup_mode"])
    batch_tokens_safe = float(max(1.0, batch_tokens))
    if lookup_mode == "scalar":
        if str(phase_kind) == "decode":
            return float(max(idle_ratio, decode_ratio))
        if str(phase_kind) == "prompt":
            return float(max(decode_ratio, prefill_ratio))
        if str(phase_kind) == "mixed":
            denom = max(1.0, float(batch_size))
            w_prompt = float(max(0.0, float(num_prompt_tasks)) / denom)
            mixed_ratio = (w_prompt * prefill_ratio) + ((1.0 - w_prompt) * decode_ratio)
            return float(max(decode_ratio, mixed_ratio))
        return float(idle_ratio)

    x_support = np.asarray(lut_params["power_support_batch_tokens"], dtype=np.float64).reshape(-1)
    if x_support.size == 0:
        raise ValueError("missing grouped power support")
    clamped_tokens = float(np.clip(batch_tokens_safe, float(x_support[0]), float(x_support[-1])))
    if abs(clamped_tokens - batch_tokens_safe) > 1e-12:
        generation_meta["splitwise_power_clamp_events"] = int(
            generation_meta.get("splitwise_power_clamp_events", 0)
        ) + 1
    if str(phase_kind) == "decode":
        y_support = np.asarray(lut_params["power_support_average_ratio"], dtype=np.float64).reshape(-1)
        ratio = _interp_or_extrapolate(x_support, y_support, clamped_tokens, extrapolate=False)
        return float(max(idle_ratio, ratio))
    if str(phase_kind) == "prompt":
        y_support = np.asarray(lut_params["power_support_peak_ratio"], dtype=np.float64).reshape(-1)
        ratio = _interp_or_extrapolate(x_support, y_support, clamped_tokens, extrapolate=False)
        return float(max(decode_ratio, ratio))
    if str(phase_kind) == "mixed":
        prompt_support = np.asarray(lut_params["power_support_peak_ratio"], dtype=np.float64).reshape(-1)
        decode_support = np.asarray(lut_params["power_support_average_ratio"], dtype=np.float64).reshape(-1)
        prompt_ratio = _interp_or_extrapolate(x_support, prompt_support, clamped_tokens, extrapolate=False)
        decode_batch_ratio = _interp_or_extrapolate(x_support, decode_support, clamped_tokens, extrapolate=False)
        denom = max(1.0, float(batch_size))
        w_prompt = float(max(0.0, float(num_prompt_tasks)) / denom)
        mixed_ratio = (w_prompt * prompt_ratio) + ((1.0 - w_prompt) * decode_batch_ratio)
        return float(max(decode_ratio, mixed_ratio))
    return float(idle_ratio)


def _append_splitwise_segment(
    segments: list[Tuple[float, float, float]],
    *,
    start_time: float,
    end_time: float,
    power_w: float,
    horizon_s: float,
) -> None:
    seg_start = float(max(0.0, min(start_time, horizon_s)))
    seg_end = float(max(seg_start, min(end_time, horizon_s)))
    if seg_end <= seg_start:
        return
    if segments and abs(float(segments[-1][2]) - float(power_w)) <= 1e-9 and abs(float(segments[-1][1]) - seg_start) <= 1e-9:
        prev_start, _, prev_power = segments[-1]
        segments[-1] = (float(prev_start), float(seg_end), float(prev_power))
        return
    segments.append((float(seg_start), float(seg_end), float(power_w)))


def _rasterize_splitwise_segments(
    segments: list[Tuple[float, float, float]],
    *,
    T: int,
    dt: float,
) -> np.ndarray:
    t_horizon = int(T)
    if t_horizon <= 0:
        return np.zeros((0,), dtype=np.float64)
    energy = np.zeros((t_horizon,), dtype=np.float64)
    dt_s = float(dt)
    for seg_start, seg_end, power_w in segments:
        start = float(seg_start)
        end = float(seg_end)
        if end <= start:
            continue
        idx = int(max(0, np.floor(start / dt_s)))
        while idx < t_horizon:
            bin_start = float(idx) * dt_s
            bin_end = bin_start + dt_s
            overlap = min(end, bin_end) - max(start, bin_start)
            if overlap > 0.0:
                energy[idx] += float(power_w) * float(overlap)
            if bin_end >= end - 1e-12:
                break
            idx += 1
    return np.asarray(energy / dt_s, dtype=np.float64).reshape(-1)


def generate_splitwise_style_lut_trace(
    requests: object,
    *,
    T: int,
    dt: float,
    config: Mapping[str, object],
    lut_params: Mapping[str, object],
) -> Tuple[np.ndarray, Dict[str, object]]:
    t_horizon = int(T)
    if t_horizon < 0:
        raise ValueError("T must be non-negative")
    if t_horizon == 0:
        generation_meta = {
            "splitwise_style_lut_mode": str(lut_params.get("splitwise_style_lut_mode", SPLITWISE_STYLE_LUT_V1)),
            "splitwise_power_support_status": str(lut_params.get("power_support_status", "")),
            "splitwise_power_quality_flag": str(lut_params.get("power_support_quality_flag", "")),
            "splitwise_source_resolved_model": str(lut_params.get("splitwise_source_resolved_model", "")),
            "splitwise_source_resolved_hardware": str(lut_params.get("splitwise_source_resolved_hardware", "")),
            "splitwise_source_resolved_tp": int(lut_params.get("splitwise_source_resolved_tp", 0)),
            "splitwise_source_match_status": str(
                lut_params.get("splitwise_source_match_status", "")
            ),
            "splitwise_scheduler_policy": str(
                lut_params.get("scheduler_defaults_policy", "")
            ),
            "splitwise_extrapolation_events": 0,
            "splitwise_power_clamp_events": 0,
            "splitwise_max_batch_tokens_seen": 0.0,
            "splitwise_preemption_events": 0,
            "splitwise_prompt_interrupt_events": 0,
            "splitwise_forced_decode_batches": 0,
        }
        return np.zeros((0,), dtype=np.float64), generation_meta
    dt_s = float(dt)
    if (not np.isfinite(dt_s)) or dt_s <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    parsed_requests = _parse_splitwise_requests(requests)
    n_gpus = int(config.get("n_gpus_per_node", lut_params.get("n_gpus_per_node", DEFAULT_NUM_GPUS_PER_NODE)))
    if n_gpus <= 0:
        raise ValueError("n_gpus_per_node must be positive")
    tp_payload = config.get("tp")
    if tp_payload is None:
        tp_payload = lut_params.get("target_tp")
    if tp_payload is None:
        tp_payload = _resolve_tp(config)
    tp = int(tp_payload)
    tp = int(max(1, min(tp, n_gpus)))
    gpu_tdp_w = float(
        config.get(
            "gpu_tdp_w",
            lut_params.get(
                "per_gpu_tdp_cap_w",
                DEFAULT_NODE_GPU_TDP_W[str(lut_params.get("target_hardware", "H100"))] / float(DEFAULT_NUM_GPUS_PER_NODE),
            ),
        )
    )
    non_gpu_power_w = float(config.get("non_gpu_power_w", 0.0))
    max_batch_size = int(
        max(
            1,
            int(
                lut_params.get(
                    "scheduler_defaults_max_batch_size",
                    lut_params["timing_support_max_batch_size"],
                )
            ),
        )
    )
    max_batch_tokens_timing = float(
        lut_params.get(
            "scheduler_defaults_max_batch_tokens",
            lut_params["timing_support_max_batch_tokens"],
        )
    )
    max_preemptions = int(
        max(1, int(lut_params.get("scheduler_defaults_max_preemptions", 4)))
    )
    max_contiguous_decode_iters = int(
        max(
            1,
            int(
                lut_params.get("scheduler_defaults_max_contiguous_decode_iters", 16)
            ),
        )
    )
    horizon_s = float(t_horizon) * dt_s
    generation_meta: Dict[str, object] = {
        "splitwise_style_lut_mode": str(
            lut_params.get("splitwise_style_lut_mode", SPLITWISE_STYLE_LUT_V1)
        ),
        "splitwise_power_support_status": str(
            lut_params.get("power_support_status", "")
        ),
        "splitwise_power_quality_flag": str(
            lut_params.get("power_support_quality_flag", "")
        ),
        "splitwise_source_resolved_model": str(
            lut_params.get("splitwise_source_resolved_model", "")
        ),
        "splitwise_source_resolved_hardware": str(
            lut_params.get("splitwise_source_resolved_hardware", "")
        ),
        "splitwise_source_resolved_tp": int(
            lut_params.get("splitwise_source_resolved_tp", 0)
        ),
        "splitwise_source_match_status": str(
            lut_params.get("splitwise_source_match_status", "")
        ),
        "splitwise_scheduler_policy": str(
            lut_params.get("scheduler_defaults_policy", "")
        ),
        "splitwise_extrapolation_events": 0,
        "splitwise_power_clamp_events": 0,
        "splitwise_max_batch_tokens_seen": 0.0,
        "splitwise_preemption_events": 0,
        "splitwise_prompt_interrupt_events": 0,
        "splitwise_contiguous_decode_runs": 0,
        "splitwise_contiguous_decode_iters_total": 0,
        "splitwise_forced_decode_batches": 0,
    }

    idle_ratio = float(lut_params["power_support_idle_ratio"])
    idle_node_gpu_power_w = (float(tp) * idle_ratio * gpu_tdp_w) + (
        float(n_gpus - tp) * idle_ratio * gpu_tdp_w
    )
    prompt_queue: list[Dict[str, float]] = []
    decode_queue: list[Dict[str, float]] = []
    blocked_queue: list[Dict[str, float]] = []
    segments: list[Tuple[float, float, float]] = []
    next_req_idx = 0
    current_time = 0.0

    def _enqueue_arrivals_up_to(time_s: float) -> None:
        nonlocal next_req_idx
        while next_req_idx < len(parsed_requests) and float(parsed_requests[next_req_idx]["arrival_time"]) <= time_s + 1e-12:
            req = dict(parsed_requests[next_req_idx])
            req["remaining_decode_tokens"] = float(max(0.0, req["output_tokens"] - 1.0))
            req["num_preemptions"] = 0.0
            prompt_queue.append(req)
            next_req_idx += 1

    def _next_prompt_arrival() -> float:
        if next_req_idx >= len(parsed_requests):
            return float("inf")
        return float(parsed_requests[next_req_idx]["arrival_time"])

    def _candidate_sort_key(item: Dict[str, object]) -> Tuple[float, float]:
        req = item["req"]
        return (float(req["arrival_time"]), float(req["request_index"]))

    while current_time < horizon_s - 1e-12:
        _enqueue_arrivals_up_to(current_time)

        if len(prompt_queue) == 0 and len(decode_queue) == 0 and len(blocked_queue) == 0:
            if next_req_idx >= len(parsed_requests):
                _append_splitwise_segment(
                    segments,
                    start_time=current_time,
                    end_time=horizon_s,
                    power_w=float(idle_node_gpu_power_w + non_gpu_power_w),
                    horizon_s=horizon_s,
                )
                break
            next_arrival = _next_prompt_arrival()
            _append_splitwise_segment(
                segments,
                start_time=current_time,
                end_time=min(next_arrival, horizon_s),
                power_w=float(idle_node_gpu_power_w + non_gpu_power_w),
                horizon_s=horizon_s,
            )
            current_time = float(max(current_time, min(next_arrival, horizon_s)))
            continue

        batch_prompt: list[Dict[str, float]] = []
        batch_decode: list[Dict[str, float]] = []
        request_ids_in_batch: set[float] = set()
        batch_tokens = 0.0

        candidate_tasks: list[Dict[str, object]] = []
        for req in blocked_queue:
            if float(req.get("num_preemptions", 0.0)) >= float(max_preemptions):
                candidate_tasks.append({"priority": 0, "task_type": "decode", "source": "blocked", "req": req})
        for req in prompt_queue:
            candidate_tasks.append({"priority": 1, "task_type": "prompt", "source": "prompt", "req": req})
        for req in blocked_queue:
            if float(req.get("num_preemptions", 0.0)) < float(max_preemptions):
                candidate_tasks.append({"priority": 2, "task_type": "decode", "source": "blocked", "req": req})
        for req in decode_queue:
            if float(req.get("num_preemptions", 0.0)) >= float(max_preemptions):
                candidate_tasks.append({"priority": 0, "task_type": "decode", "source": "decode", "req": req})
            else:
                candidate_tasks.append({"priority": 3, "task_type": "decode", "source": "decode", "req": req})
        candidate_tasks.sort(key=lambda item: (int(item["priority"]), *_candidate_sort_key(item)))

        selected_ids: set[int] = set()
        forced_decode_in_batch = False
        for item in candidate_tasks:
            if (len(batch_prompt) + len(batch_decode)) >= max_batch_size:
                break
            req = item["req"]
            request_id = float(req["request_index"])
            if request_id in request_ids_in_batch:
                continue
            task_tokens = (
                float(max(0.0, req["input_tokens"]))
                if str(item["task_type"]) == "prompt"
                else 1.0
            )
            can_add = batch_tokens + task_tokens <= max_batch_tokens_timing + 1e-12
            if (not can_add) and (len(batch_prompt) + len(batch_decode) > 0):
                continue
            if (not can_add) and (len(batch_prompt) + len(batch_decode) == 0):
                generation_meta["splitwise_extrapolation_events"] = int(
                    generation_meta.get("splitwise_extrapolation_events", 0)
                ) + 1
            request_ids_in_batch.add(request_id)
            selected_ids.add(id(req))
            batch_tokens += task_tokens
            if str(item["task_type"]) == "prompt":
                batch_prompt.append(req)
            else:
                batch_decode.append(req)
                if int(item["priority"]) == 0:
                    forced_decode_in_batch = True

        if forced_decode_in_batch:
            generation_meta["splitwise_forced_decode_batches"] = int(
                generation_meta.get("splitwise_forced_decode_batches", 0)
            ) + 1

        if selected_ids:
            prompt_queue = [req for req in prompt_queue if id(req) not in selected_ids]
            decode_queue = [req for req in decode_queue if id(req) not in selected_ids]
            blocked_queue = [req for req in blocked_queue if id(req) not in selected_ids]

        if len(batch_prompt) == 0 and len(batch_decode) == 0:
            next_arrival = (
                _next_prompt_arrival()
                if next_req_idx < len(parsed_requests)
                else horizon_s
            )
            _append_splitwise_segment(
                segments,
                start_time=current_time,
                end_time=min(next_arrival, horizon_s),
                power_w=float(idle_node_gpu_power_w + non_gpu_power_w),
                horizon_s=horizon_s,
            )
            current_time = float(max(current_time, min(next_arrival, horizon_s)))
            continue

        if len(batch_prompt) > 0 and len(decode_queue) > 0:
            newly_blocked: list[Dict[str, float]] = []
            for req in decode_queue:
                req["num_preemptions"] = float(req.get("num_preemptions", 0.0) + 1.0)
                generation_meta["splitwise_preemption_events"] = int(
                    generation_meta.get("splitwise_preemption_events", 0)
                ) + 1
                newly_blocked.append(req)
            blocked_queue.extend(newly_blocked)
            decode_queue = []

        generation_meta["splitwise_max_batch_tokens_seen"] = float(
            max(float(generation_meta.get("splitwise_max_batch_tokens_seen", 0.0)), batch_tokens)
        )
        if len(batch_prompt) > 0 and len(batch_decode) > 0:
            phase_kind = "mixed"
        elif len(batch_prompt) > 0:
            phase_kind = "prompt"
        else:
            phase_kind = "decode"
        iteration_duration_s = _predict_splitwise_timing_s(
            lut_params,
            batch_tokens=batch_tokens,
            phase_kind=phase_kind,
            generation_meta=generation_meta,
        )
        contiguous_iters = 1
        decode_interrupted = False
        if str(phase_kind) == "decode":
            min_remaining_decode_tokens = float(
                min(float(req.get("remaining_decode_tokens", 0.0)) for req in batch_decode)
            )
            contiguous_iters = int(
                max(
                    1,
                    min(
                        int(np.floor(max(1.0, min_remaining_decode_tokens))),
                        int(max_contiguous_decode_iters),
                    ),
                )
            )
            next_prompt_arrival = _next_prompt_arrival()
            if np.isfinite(next_prompt_arrival):
                window_s = float(contiguous_iters) * float(iteration_duration_s)
                if float(next_prompt_arrival) < float(current_time + window_s) - 1e-12:
                    boundary_iters = int(
                        np.floor(
                            max(0.0, float(next_prompt_arrival - current_time))
                            / float(iteration_duration_s)
                        )
                    ) + 1
                    contiguous_iters = int(max(1, min(contiguous_iters, boundary_iters)))
                    decode_interrupted = True
            generation_meta["splitwise_contiguous_decode_runs"] = int(
                generation_meta.get("splitwise_contiguous_decode_runs", 0)
            ) + 1
            generation_meta["splitwise_contiguous_decode_iters_total"] = int(
                generation_meta.get("splitwise_contiguous_decode_iters_total", 0)
            ) + int(contiguous_iters)

        duration_s = float(iteration_duration_s) * float(contiguous_iters)
        phase_ratio = _lookup_splitwise_power_ratio(
            lut_params,
            batch_tokens=batch_tokens,
            batch_size=len(batch_prompt) + len(batch_decode),
            num_prompt_tasks=len(batch_prompt),
            num_decode_tasks=len(batch_decode),
            phase_kind=phase_kind,
            generation_meta=generation_meta,
        )
        active_gpu_power = float(tp) * float(phase_ratio) * float(gpu_tdp_w)
        idle_gpu_power = float(n_gpus - tp) * float(idle_ratio) * float(gpu_tdp_w)
        node_gpu_power = float(active_gpu_power + idle_gpu_power)
        iteration_end = float(current_time + duration_s)
        _append_splitwise_segment(
            segments,
            start_time=current_time,
            end_time=iteration_end,
            power_w=float(node_gpu_power + non_gpu_power_w),
            horizon_s=horizon_s,
        )
        current_time = iteration_end

        _enqueue_arrivals_up_to(current_time)

        for req in batch_prompt:
            remaining_decode = float(
                max(
                    0.0,
                    req.get(
                        "remaining_decode_tokens",
                        max(0.0, req["output_tokens"] - 1.0),
                    ),
                )
            )
            if remaining_decode > 0.0:
                decode_req = dict(req)
                decode_req["remaining_decode_tokens"] = float(remaining_decode)
                decode_req["num_preemptions"] = 0.0
                decode_queue.append(decode_req)

        for req in batch_decode:
            remaining_decode = float(
                max(
                    0.0,
                    req.get("remaining_decode_tokens", 0.0)
                    - float(contiguous_iters),
                )
            )
            if remaining_decode > 0.0:
                decode_req = dict(req)
                decode_req["remaining_decode_tokens"] = float(remaining_decode)
                if decode_interrupted:
                    decode_req["num_preemptions"] = float(
                        decode_req.get("num_preemptions", 0.0) + 1.0
                    )
                    generation_meta["splitwise_preemption_events"] = int(
                        generation_meta.get("splitwise_preemption_events", 0)
                    ) + 1
                    generation_meta["splitwise_prompt_interrupt_events"] = int(
                        generation_meta.get("splitwise_prompt_interrupt_events", 0)
                    ) + 1
                    if float(decode_req["num_preemptions"]) >= float(max_preemptions):
                        decode_queue.insert(0, decode_req)
                    else:
                        blocked_queue.append(decode_req)
                else:
                    decode_queue.append(decode_req)

    trace = _rasterize_splitwise_segments(segments, T=t_horizon, dt=dt_s)
    power_status = str(generation_meta.get("splitwise_power_support_status", ""))
    clamp_events = int(generation_meta.get("splitwise_power_clamp_events", 0))
    if clamp_events > 0 and power_status and not power_status.endswith("_clamped"):
        generation_meta["splitwise_power_support_status"] = f"{power_status}_clamped"
    return trace.astype(np.float64), generation_meta


def generate_splitwise_lut(
    a_raw: np.ndarray,
    delta_a_raw: np.ndarray,
    config: Mapping[str, object],
    lut_params: Mapping[str, object],
) -> np.ndarray:
    del a_raw
    del delta_a_raw
    del config
    del lut_params
    raise ValueError(SPLITWISE_REMOVED_MESSAGE)


def _resolve_device(config: Mapping[str, object], classifier: torch.nn.Module) -> torch.device:
    raw = config.get("device")
    if raw is not None:
        return torch.device(str(raw))
    first_param = next(classifier.parameters(), None)
    if first_param is not None:
        return first_param.device
    return torch.device("cpu")


def generate_tdp(n_timesteps: int, config: Mapping[str, object]) -> np.ndarray:
    """Every timestep = node-level TDP under conservative all-8-GPUs-active assumption."""
    n = int(n_timesteps)
    if n < 0:
        raise ValueError("n_timesteps must be >= 0")
    tdp_node_w = _resolve_tdp_node_w(config)
    return np.full((n,), tdp_node_w, dtype=np.float64)


def generate_mean(
    n_timesteps: int,
    config: Mapping[str, object],
    train_data: np.ndarray,
) -> np.ndarray:
    """Every timestep = empirical mean from training traces."""
    del config
    n = int(n_timesteps)
    if n < 0:
        raise ValueError("n_timesteps must be >= 0")
    train_arr = np.asarray(train_data, dtype=np.float64).reshape(-1)
    if train_arr.size == 0:
        raise ValueError("train_data must be non-empty")
    mean_power = float(np.mean(train_arr))
    return np.full((n,), mean_power, dtype=np.float64)


def generate_marginal_gmm(
    n_timesteps: int,
    config: Mapping[str, object],
    gmm_params: Mapping[str, object],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample i.i.d. from the marginal GMM (no temporal model)."""
    del config
    n = int(n_timesteps)
    if n < 0:
        raise ValueError("n_timesteps must be >= 0")
    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
    weights = np.asarray(gmm_params["weights"], dtype=np.float64).reshape(-1)
    if means.size == 0:
        raise ValueError("GMM means are empty")
    if variances.size != means.size or weights.size != means.size:
        raise ValueError("GMM parameter shape mismatch")

    weights = np.clip(weights, a_min=1e-12, a_max=None)
    weights = weights / float(np.sum(weights))
    stds = np.sqrt(np.clip(variances, a_min=1e-12, a_max=None))
    local_rng = rng if rng is not None else np.random.default_rng()
    components = local_rng.choice(int(means.size), size=n, p=weights)
    return local_rng.normal(loc=means[components], scale=stds[components]).astype(np.float64)


def generate_ours(
    feature_sequence: np.ndarray,
    config: Mapping[str, object],
    classifier: torch.nn.Module,
    gmm_params: Mapping[str, object],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Full pipeline: BiGRU logits + GMM/AR1 sampling."""
    features = np.asarray(feature_sequence, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"feature_sequence must have shape (T,D); got {features.shape}")
    t_horizon = int(features.shape[0])
    if t_horizon <= 0:
        return np.zeros((0,), dtype=np.float64)

    device = _resolve_device(config, classifier)
    classifier.eval()
    with torch.no_grad():
        try:
            x = torch.from_numpy(features)
        except Exception:
            x = torch.tensor(features.tolist(), dtype=torch.float32)
        x = x.to(device=device, dtype=torch.float32).unsqueeze(0)
        logits = classifier(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    decode_mode = str(config.get("decode_mode", "stochastic"))
    median_filter_window = int(config.get("median_filter_window", 1))
    clamp_range = config.get("clamp_range")
    std_scale = float(config.get("std_scale", 1.0))
    logit_temperature = float(config.get("logit_temperature", 1.0))
    if (not np.isfinite(std_scale)) or std_scale <= 0.0:
        raise ValueError(f"std_scale must be > 0, got {std_scale}")
    if (not np.isfinite(logit_temperature)) or logit_temperature <= 0.0:
        raise ValueError(f"logit_temperature must be > 0, got {logit_temperature}")
    if abs(logit_temperature - 1.0) > 1e-12:
        logits = logits / float(logit_temperature)

    gmm_sampling = dict(gmm_params)
    if abs(std_scale - 1.0) > 1e-12:
        variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
        gmm_sampling["variances"] = np.clip(variances * float(std_scale * std_scale), a_min=1e-12, a_max=None)

    p0 = float(config.get("p0", np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)[0]))
    seed = _extract_seed(config, rng)

    ar1_params = config.get("ar1_params")
    if isinstance(ar1_params, Mapping):
        phi = np.asarray(ar1_params["phi"], dtype=np.float64).reshape(-1)
        sigma_innov = np.asarray(ar1_params["sigma_innov"], dtype=np.float64).reshape(-1) * float(std_scale)
        sigma_marginal_payload = ar1_params.get("sigma_marginal")
        if sigma_marginal_payload is None:
            sigma_marginal = np.sqrt(
                np.clip(np.asarray(gmm_sampling["variances"], dtype=np.float64).reshape(-1), a_min=1e-12, a_max=None)
            )
        else:
            sigma_marginal = np.asarray(sigma_marginal_payload, dtype=np.float64).reshape(-1) * float(std_scale)
        phi_threshold = float(ar1_params.get("phi_threshold", config.get("phi_threshold", 0.3)))
        _, generate_gmm_bigru_trace_ar1_thresholded = _load_pipeline_generators()
        generated = generate_gmm_bigru_trace_ar1_thresholded(
            logits=logits,
            gmm_params=gmm_sampling,
            phi=phi,
            sigma_innov=sigma_innov,
            sigma_marginal=sigma_marginal,
            p0=p0,
            seed=seed,
            decode_mode=decode_mode,
            median_filter_window=median_filter_window,
            phi_threshold=phi_threshold,
            clamp_range=clamp_range,
        )
    else:
        generate_gmm_bigru_trace, _ = _load_pipeline_generators()
        generated = generate_gmm_bigru_trace(
            logits=logits,
            gmm_params=gmm_sampling,
            seed=seed,
            decode_mode=decode_mode,
            median_filter_window=median_filter_window,
            clamp_range=clamp_range,
        )

    power = np.asarray(generated["power_w"], dtype=np.float64).reshape(-1)
    if power.size == t_horizon:
        return power
    if power.size > t_horizon:
        return power[:t_horizon].astype(np.float64)
    out = np.empty((t_horizon,), dtype=np.float64)
    out[: power.size] = power
    out[power.size :] = power[-1] if power.size > 0 else p0
    return out
