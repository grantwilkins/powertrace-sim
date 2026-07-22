#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

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


def _load_pipeline_generator() -> object:
    from scripts.eval.pipeline_utils import generate_gmm_bigru_trace

    return generate_gmm_bigru_trace


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


from scripts.eval.splitwise import (
    SPLITWISE_REMOVED_MESSAGE,
    SPLITWISE_STYLE_LUT_V1,
    build_splitwise_style_lut_params,
    build_splitwise_style_lut_trace_params,
    generate_splitwise_lut,
    generate_splitwise_style_lut_trace,
    normalize_splitwise_style_lut_mode,
)

def _resolve_device(config: Mapping[str, object], classifier: torch.nn.Module) -> torch.device:
    import torch

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
    """Full pipeline: BiGRU logits + IID GMM sampling."""
    import torch

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

    generate_gmm_bigru_trace = _load_pipeline_generator()
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
