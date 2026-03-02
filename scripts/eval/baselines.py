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

from scripts.eval.pipeline_utils import (
    generate_gmm_bigru_trace,
    generate_gmm_bigru_trace_ar1_thresholded,
)

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


def _clamp_per_gpu_phase_powers(
    *,
    idle_w: float,
    decode_w: float,
    prefill_w: float,
    per_gpu_tdp_cap_w: float,
) -> Tuple[float, float, float]:
    idle_out = max(1.0, float(idle_w))
    decode_out = max(idle_out, float(decode_w))
    prefill_out = max(decode_out, float(prefill_w))
    if np.isfinite(per_gpu_tdp_cap_w) and per_gpu_tdp_cap_w > 0.0:
        decode_out = min(decode_out, float(per_gpu_tdp_cap_w))
        prefill_out = min(prefill_out, float(per_gpu_tdp_cap_w))
        decode_out = max(decode_out, idle_out)
        prefill_out = max(prefill_out, decode_out)
    return idle_out, decode_out, prefill_out


def _suppress_short_true_runs(mask: np.ndarray, min_run_len: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size == 0 or int(min_run_len) <= 1:
        return arr
    out = arr.copy()
    i = 0
    n = int(arr.size)
    min_len = int(min_run_len)
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        if (j - i) < min_len:
            out[i:j] = False
        i = j
    return out


def build_splitwise_lut_params(
    config_id: str,
    perf_model_csv: str,
    train_power_flat: np.ndarray,
    *,
    splitwise_source_model: str = "llama2-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: int = 4,
    splitwise_calibration_mode: str = "train_phase_matched_v1",
    n_gpus_per_node: int = DEFAULT_NUM_GPUS_PER_NODE,
    per_gpu_tdp_cap_w: Optional[float] = None,
    target_idle_node_gpu_w: Optional[float] = None,
    target_decode_node_gpu_w: Optional[float] = None,
    target_prefill_node_gpu_w: Optional[float] = None,
) -> Dict[str, float]:
    if int(splitwise_source_tp) <= 0:
        raise ValueError("splitwise_source_tp must be > 0")

    rows: list[Dict[str, object]] = []
    with open(perf_model_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                model = str(row.get("model", "")).strip()
                hardware = str(row.get("hardware", "")).strip()
                tp = int(float(row.get("tensor_parallel", "nan")))
                peak = float(row.get("peak_power", "nan"))
                avg = float(row.get("average_power", "nan"))
            except Exception:
                continue
            if model != str(splitwise_source_model):
                continue
            if hardware != str(splitwise_source_hardware):
                continue
            if tp != int(splitwise_source_tp):
                continue
            if (not np.isfinite(peak)) or (not np.isfinite(avg)):
                continue
            if peak <= 0.0 or avg <= 0.0:
                continue
            rows.append(
                {
                    "peak_power": float(peak),
                    "average_power": float(avg),
                }
            )
    if len(rows) == 0:
        raise ValueError(
            f"No valid LUT rows for model={splitwise_source_model}, hardware={splitwise_source_hardware}, tp={splitwise_source_tp}"
        )

    peak_arr = np.asarray([r["peak_power"] for r in rows], dtype=np.float64)
    avg_arr = np.asarray([r["average_power"] for r in rows], dtype=np.float64)
    idle_norm = _safe_percentile(avg_arr, 10.0)
    decode_norm = float(np.median(avg_arr))
    prefill_norm = float(np.median(peak_arr))
    decode_norm = max(decode_norm, idle_norm)
    prefill_norm = max(prefill_norm, decode_norm)

    config_id_str = str(config_id)
    try:
        target_hardware = _resolve_hardware({"config_id": config_id_str})
    except Exception:
        target_hardware = "H100"
    try:
        target_tp = int(_resolve_tp({"config_id": config_id_str}))
    except Exception:
        target_tp = 4
    n_gpus = int(max(1, n_gpus_per_node))
    target_tp = int(max(1, min(target_tp, n_gpus)))

    train_arr = np.asarray(train_power_flat, dtype=np.float64).reshape(-1)
    train_arr = train_arr[np.isfinite(train_arr)]
    if train_arr.size == 0:
        raise ValueError("train_power_flat must contain finite values.")

    mode_raw = str(splitwise_calibration_mode).strip().lower()
    use_dgx_targets = mode_raw in {
        "dgx_fixed_targets_v1",
        "dgx_fixed_targets",
        "hardware_targets",
        "hardware_targets_v1",
    }
    use_train_phase_matched = mode_raw in {
        "train_phase_matched_v1",
        "train_phase_matched",
    }

    idle_target_w = float("nan")
    decode_target_w = float("nan")
    prefill_target_w = float("nan")
    a = float("nan")
    b = float("nan")

    idle_target_override = _finite_or_none(target_idle_node_gpu_w)
    decode_target_override = _finite_or_none(target_decode_node_gpu_w)
    prefill_target_override = _finite_or_none(target_prefill_node_gpu_w)

    if use_dgx_targets or use_train_phase_matched:
        tdp_node_gpu_target_w = float(DEFAULT_NODE_GPU_TDP_W[target_hardware]) * (
            float(n_gpus) / float(DEFAULT_NUM_GPUS_PER_NODE)
        )
        if per_gpu_tdp_cap_w is not None:
            tdp_node_gpu_target_w = float(per_gpu_tdp_cap_w) * float(n_gpus)

        if use_train_phase_matched:
            decode_node_gpu_target_w = (
                float(decode_target_override)
                if decode_target_override is not None
                else float(np.mean(train_arr))
            )
            idle_node_gpu_target_w = (
                float(idle_target_override)
                if idle_target_override is not None
                else _safe_percentile(train_arr, 10.0)
            )
            prefill_node_gpu_target_w = (
                float(prefill_target_override)
                if prefill_target_override is not None
                else _safe_percentile(train_arr, 95.0)
            )
            splitwise_calibration_mode = "train_phase_matched_v1"
        else:
            decode_node_gpu_target_w = (
                float(decode_target_override)
                if decode_target_override is not None
                else float(DEFAULT_NODE_GPU_ACTIVE_W[target_hardware])
                * (float(n_gpus) / float(DEFAULT_NUM_GPUS_PER_NODE))
            )
            idle_node_gpu_target_w = (
                float(idle_target_override)
                if idle_target_override is not None
                else float(
                    decode_node_gpu_target_w
                    * np.clip(idle_norm / max(decode_norm, 1e-9), 0.45, 0.90)
                )
            )
            prefill_node_gpu_target_w = (
                float(prefill_target_override)
                if prefill_target_override is not None
                else float(
                    decode_node_gpu_target_w
                    * np.clip(prefill_norm / max(decode_norm, 1e-9), 1.00, 2.00)
                )
            )
            splitwise_calibration_mode = "dgx_fixed_targets_v1"

        decode_node_gpu_target_w = max(1.0, float(decode_node_gpu_target_w))
        idle_node_gpu_target_w = max(1.0, float(idle_node_gpu_target_w))
        prefill_node_gpu_target_w = max(1.0, float(prefill_node_gpu_target_w))
        decode_node_gpu_target_w = max(decode_node_gpu_target_w, idle_node_gpu_target_w)
        prefill_node_gpu_target_w = max(prefill_node_gpu_target_w, decode_node_gpu_target_w)
        prefill_node_gpu_target_w = min(prefill_node_gpu_target_w, tdp_node_gpu_target_w)

        idle_w = float(idle_node_gpu_target_w / float(n_gpus))
        decode_w = float(
            (decode_node_gpu_target_w - float(n_gpus - target_tp) * idle_w) / float(target_tp)
        )
        prefill_w = float(
            (prefill_node_gpu_target_w - float(n_gpus - target_tp) * idle_w) / float(target_tp)
        )
        cap_w = (
            float(per_gpu_tdp_cap_w)
            if per_gpu_tdp_cap_w is not None
            else float(tdp_node_gpu_target_w / float(n_gpus))
        )
        idle_w, decode_w, prefill_w = _clamp_per_gpu_phase_powers(
            idle_w=idle_w,
            decode_w=decode_w,
            prefill_w=prefill_w,
            per_gpu_tdp_cap_w=cap_w,
        )
        idle_target_w = float(idle_node_gpu_target_w)
        decode_target_w = float(decode_node_gpu_target_w)
        prefill_target_w = float(prefill_node_gpu_target_w)
    else:
        # Backward-compatibility path: fit normalized LUT to train-trace statistics, then convert to per-GPU.
        idle_target_node_w = _safe_percentile(train_arr, 10.0)
        decode_target_node_w = float(np.mean(train_arr))
        prefill_target_node_w = _safe_percentile(train_arr, 95.0)
        non_gpu_ref_w = float(DEFAULT_NON_GPU_POWER_W)
        idle_target_node_gpu_w = max(1.0, float(idle_target_node_w - non_gpu_ref_w))
        decode_target_node_gpu_w = max(idle_target_node_gpu_w, float(decode_target_node_w - non_gpu_ref_w))
        prefill_target_node_gpu_w = max(
            decode_target_node_gpu_w, float(prefill_target_node_w - non_gpu_ref_w)
        )

        a, b = _fit_affine_map(
            np.asarray([idle_norm, decode_norm, prefill_norm], dtype=np.float64),
            np.asarray(
                [idle_target_node_gpu_w, decode_target_node_gpu_w, prefill_target_node_gpu_w],
                dtype=np.float64,
            ),
        )
        idle_node_gpu_w = float((a * idle_norm) + b)
        decode_node_gpu_w = float((a * decode_norm) + b)
        prefill_node_gpu_w = float((a * prefill_norm) + b)

        idle_w = float(idle_node_gpu_w / float(n_gpus))
        decode_w = float(
            (decode_node_gpu_w - float(n_gpus - target_tp) * idle_w) / float(target_tp)
        )
        prefill_w = float(
            (prefill_node_gpu_w - float(n_gpus - target_tp) * idle_w) / float(target_tp)
        )
        cap_w = (
            float(per_gpu_tdp_cap_w)
            if per_gpu_tdp_cap_w is not None
            else float(DEFAULT_NODE_GPU_TDP_W[target_hardware] / float(max(1, DEFAULT_NUM_GPUS_PER_NODE)))
        )
        idle_w, decode_w, prefill_w = _clamp_per_gpu_phase_powers(
            idle_w=idle_w,
            decode_w=decode_w,
            prefill_w=prefill_w,
            per_gpu_tdp_cap_w=cap_w,
        )
        idle_target_w = float(idle_target_node_gpu_w)
        decode_target_w = float(decode_target_node_gpu_w)
        prefill_target_w = float(prefill_target_node_gpu_w)

    return {
        "config_id": str(config_id),
        "splitwise_source_model": str(splitwise_source_model),
        "splitwise_source_hardware": str(splitwise_source_hardware),
        "splitwise_source_tp": int(splitwise_source_tp),
        "splitwise_calibration_mode": str(splitwise_calibration_mode),
        "target_hardware": str(target_hardware),
        "target_tp": int(target_tp),
        "n_gpus_per_node": int(n_gpus),
        "idle_norm": float(idle_norm),
        "decode_norm": float(decode_norm),
        "prefill_norm": float(prefill_norm),
        "idle_target_w": float(idle_target_w),      # node GPU-only target
        "decode_target_w": float(decode_target_w),  # node GPU-only target
        "prefill_target_w": float(prefill_target_w),  # node GPU-only target
        "affine_a": float(a),
        "affine_b": float(b),
        "idle_w": float(idle_w),       # per-GPU idle power
        "decode_w": float(decode_w),   # per-GPU decode power
        "prefill_w": float(prefill_w),  # per-GPU prefill power
        "prefill_delta_threshold": 0.05,
        "prefill_min_steps": 2,
        "phase_detection_mode": "A_raw_delta_A_thresholded",
        "phase_detection_note": "Approximate Splitwise phase from A_t and delta_A_t, not scheduler task labels.",
        "decode_occupancy_note": "Decode phase uses a single level and ignores occupancy.",
    }


def generate_splitwise_lut(
    a_raw: np.ndarray,
    delta_a_raw: np.ndarray,
    config: Mapping[str, object],
    lut_params: Mapping[str, object],
) -> np.ndarray:
    a = np.asarray(a_raw, dtype=np.float64).reshape(-1)
    da = np.asarray(delta_a_raw, dtype=np.float64).reshape(-1)
    n = int(min(a.size, da.size))
    if n < 0:
        raise ValueError("invalid horizon")
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    n_gpus = int(config.get("n_gpus_per_node", DEFAULT_NUM_GPUS_PER_NODE))
    if n_gpus <= 0:
        raise ValueError("n_gpus_per_node must be positive")
    tp = int(config.get("tp", _resolve_tp(config)))
    tp = int(max(1, min(tp, n_gpus)))
    non_gpu_power_w = float(config.get("non_gpu_power_w", DEFAULT_NON_GPU_POWER_W))
    eps = float(config.get("phase_eps", 1e-6))
    prefill_delta_threshold = float(
        config.get("prefill_delta_threshold", lut_params.get("prefill_delta_threshold", 0.05))
    )
    prefill_min_steps = int(config.get("prefill_min_steps", lut_params.get("prefill_min_steps", 2)))

    idle_w = float(lut_params["idle_w"])
    decode_w = float(lut_params["decode_w"])
    prefill_w = float(lut_params["prefill_w"])
    idle_w = max(1.0, idle_w)
    decode_w = max(idle_w, decode_w)
    prefill_w = max(decode_w, prefill_w)

    a_n = a[:n]
    da_n = da[:n]
    active_mask = a_n > eps
    prefill_mask = active_mask & (da_n > max(eps, prefill_delta_threshold))
    prefill_mask = _suppress_short_true_runs(prefill_mask, min_run_len=prefill_min_steps)
    decode_mask = active_mask & (~prefill_mask)

    phase_power = np.full((n,), idle_w, dtype=np.float64)
    phase_power[decode_mask] = decode_w
    phase_power[prefill_mask] = prefill_w

    active_gpus = np.where(active_mask, tp, 0).astype(np.float64)
    idle_gpus = float(n_gpus) - active_gpus
    power = (active_gpus * phase_power) + (idle_gpus * idle_w) + float(non_gpu_power_w)
    return np.asarray(power, dtype=np.float64).reshape(-1)


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
