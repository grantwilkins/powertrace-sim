#!/usr/bin/env python3
"""
Appendix C1 decomposition fidelity pipeline.

Compares three generation strategies over held-out traces:
- Oracle sequence + within-regime sampling
- Predicted sequence + within-regime sampling
- Marginal GMM sampling (no sequence)

Outputs:
- per-trace metrics CSV
- per-config/strategy summary CSV (median + min/max)
- two-panel Figure C1 PDF
- reproducibility manifest JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.classifiers.gru import GRUClassifier  # noqa: E402
from model.classifiers.metrics import compute_power_metrics  # noqa: E402
from model.classifiers.feature_utils import (  # noqa: E402
    compute_delta_active_requests,
    compute_inference_features,
    normalize_delta_active_requests,
)

DEFAULT_CONFIG_IDS = (
    "deepseek-r1-distill-8b_H100_tp1",
    "llama-3-8b_A100_tp2",
    "llama-3-70b_H100_tp4",
    "gpt-oss-120b_A100_tp8",
)
DEFAULT_MOE_CONFIG_ID = "gpt-oss-120b_A100_tp8"

STRATEGY_ORDER = ("oracle", "predicted", "marginal")
STRATEGY_DISPLAY = {
    "oracle": "Oracle",
    "predicted": "Predicted",
    "marginal": "Marginal",
}
STRATEGY_COLORS = {
    "oracle": "#1f4e79",   # dark blue
    "predicted": "#5b8cc0",  # medium blue
    "marginal": "#c8c8c8",  # light gray
}

AR1_PHI_THRESHOLD_DEFAULT = 0.3
AR1_MIN_RUN_LENGTH = 5
MODEL_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")
EPS = 1e-12


def _safe_std(value: float) -> float:
    out = float(value)
    if (not np.isfinite(out)) or out <= 0.0:
        return 1e-6
    return max(out, 1e-6)


def _extract_norm_value(norm: Mapping[str, float], *keys: str) -> float:
    for key in keys:
        if key in norm:
            return float(norm[key])
    raise KeyError(f"Missing required norm key; expected one of {keys}")


def _validate_feature_set(feature_set: str) -> str:
    value = str(feature_set).strip().lower()
    if value not in {"f2", "f3"}:
        raise ValueError(f"feature_set must be one of {{'f2','f3'}}; got {feature_set}")
    return value


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / np.clip(denom, a_min=EPS, a_max=None)


def _tensor_to_numpy(tensor: torch.Tensor, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Convert torch tensor to numpy array without using torch's .numpy() bridge.

    This avoids runtime failures in environments where torch was built against
    a different NumPy ABI.
    """
    t_cpu = tensor.detach().to(device="cpu")
    return np.asarray(t_cpu.tolist(), dtype=dtype)


def _median_filter_states(states: np.ndarray, window: int) -> np.ndarray:
    z = np.asarray(states, dtype=np.int64).reshape(-1)
    n = int(z.size)
    if n == 0:
        return z.copy()

    w = int(max(1, window))
    if w < 3:
        return z.copy()
    if w % 2 == 0:
        w += 1
    half = w // 2

    out = np.zeros_like(z)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = int(np.median(z[lo:hi]))
    return out


def load_gmm_params_json_dict(payload: Mapping[str, object]) -> Dict[str, object]:
    means = np.asarray(payload.get("means", []), dtype=np.float64).reshape(-1)
    variances = np.asarray(payload.get("variances", []), dtype=np.float64).reshape(-1)
    weights = np.asarray(payload.get("weights", []), dtype=np.float64).reshape(-1)
    order = np.asarray(payload.get("order", []), dtype=np.int64).reshape(-1)
    label_map = np.asarray(payload.get("label_map", []), dtype=np.int64).reshape(-1)

    k = int(payload.get("k", means.size))
    if means.size != k or variances.size != k or weights.size != k:
        raise ValueError("Invalid GMM payload: array lengths must match k.")
    if order.size != k and order.size != 0:
        raise ValueError("Invalid GMM payload: order length must match k.")
    if label_map.size != k and label_map.size != 0:
        raise ValueError("Invalid GMM payload: label_map length must match k.")
    if order.size == 0:
        order = np.arange(k, dtype=np.int64)
    if label_map.size == 0:
        label_map = np.arange(k, dtype=np.int64)

    return {
        "k": int(k),
        "covariance_type": str(payload.get("covariance_type", "full")),
        "means": means.astype(np.float64),
        "variances": np.clip(variances, a_min=1e-12, a_max=None).astype(np.float64),
        "weights": weights.astype(np.float64),
        "order": order.astype(np.int64),
        "label_map": label_map.astype(np.int64),
        "aic": float(payload.get("aic", float("nan"))),
        "bic": float(payload.get("bic", float("nan"))),
    }


def build_rollout_features_from_requests(
    requests: Sequence[Dict[str, object]],
    throughput: Mapping[str, float],
    norm: Mapping[str, float],
    T: Optional[int] = None,
    dt: float = 0.25,
    feature_set: str = "f2",
) -> Dict[str, np.ndarray]:
    feat = _validate_feature_set(feature_set)
    lambda_prefill = _extract_norm_value(
        throughput,
        "lambda_prefill",
        "prefill_rate_median_toks_per_s",
    )
    lambda_decode = _extract_norm_value(
        throughput,
        "lambda_decode",
        "decode_rate_median_toks_per_s",
    )
    if lambda_prefill <= 0.0 or lambda_decode <= 0.0:
        raise ValueError("Throughput rates must be positive.")

    active_mean = _extract_norm_value(norm, "active_mean", "A_mean")
    active_std = _safe_std(_extract_norm_value(norm, "active_std", "A_std"))
    t_mean = _extract_norm_value(norm, "t_arrive_log_mean", "T_arrive_log_mean")
    t_std = _safe_std(_extract_norm_value(norm, "t_arrive_log_std", "T_arrive_log_std"))
    dA_mean = _extract_norm_value(norm, "delta_A_mean")
    dA_std = _safe_std(_extract_norm_value(norm, "delta_A_std"))

    base = compute_inference_features(
        requests=requests,
        config={
            "lambda_prefill": float(lambda_prefill),
            "lambda_decode": float(lambda_decode),
            "A_mean": float(active_mean),
            "A_std": float(active_std),
            "T_arrive_log_mean": float(t_mean),
            "T_arrive_log_std": float(t_std),
        },
        T=T,
        dt=float(dt),
    )
    if base.ndim != 2 or base.shape[1] != 2:
        raise ValueError(f"Expected base features with shape (T,2); got {base.shape}")

    a_norm = np.asarray(base[:, 0], dtype=np.float32)
    t_arrive_norm = np.asarray(base[:, 1], dtype=np.float32)
    a_raw = (a_norm.astype(np.float64) * float(active_std)) + float(active_mean)
    delta_raw = compute_delta_active_requests(a_raw).astype(np.float64)
    delta_norm = normalize_delta_active_requests(delta_raw, mean=dA_mean, std=dA_std).astype(np.float32)

    cols = [a_norm, delta_norm]
    if feat == "f3":
        cols.append(t_arrive_norm)
    features = np.stack(cols, axis=-1).astype(np.float32)

    return {
        "features_norm": features,
        "A_raw": a_raw.astype(np.float64),
        "A_norm": a_norm.astype(np.float32),
        "delta_A_raw": delta_raw.astype(np.float64),
        "delta_A_norm": delta_norm.astype(np.float32),
        "t_arrive_norm": t_arrive_norm.astype(np.float32),
    }


def generate_gmm_bigru_trace(
    logits: Union[np.ndarray, torch.Tensor],
    gmm_params: Mapping[str, object],
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    if isinstance(logits, torch.Tensor):
        z = _tensor_to_numpy(logits, dtype=np.float64)
    else:
        z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 2:
        raise ValueError(f"logits must have shape (T,K) or (1,T,K); got {z.shape}")

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
    k = int(means.size)
    if z.shape[1] != k:
        raise ValueError(f"logits K mismatch: got {z.shape[1]} but GMM has {k}")

    probs = _softmax_np(z)
    mode = str(decode_mode).strip().lower()
    if mode not in {"stochastic", "argmax"}:
        raise ValueError(f"decode_mode must be 'stochastic' or 'argmax'; got {decode_mode}")

    rng = np.random.default_rng(seed)
    if mode == "argmax":
        states_raw = np.argmax(probs, axis=-1).astype(np.int64)
    else:
        states_raw = np.asarray([rng.choice(k, p=probs_t) for probs_t in probs], dtype=np.int64)
    states = _median_filter_states(states_raw, int(median_filter_window))

    std = np.sqrt(np.clip(variances, a_min=1e-12, a_max=None))
    power = rng.normal(loc=means[states], scale=std[states]).astype(np.float64)
    power = _apply_clamp(power, clamp_range)

    return {
        "power_w": power.astype(np.float64),
        "states": states.astype(np.int64),
        "states_raw": states_raw.astype(np.int64),
        "probs": probs.astype(np.float64),
    }


def predict_sorted_gmm_labels_from_params(power_values: np.ndarray, gmm_params: Dict[str, object]) -> np.ndarray:
    y = np.asarray(power_values, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.zeros((0,), dtype=np.int64)

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.clip(np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1), a_min=1e-12, a_max=None)
    weights = np.asarray(gmm_params.get("weights", np.ones_like(means)), dtype=np.float64).reshape(-1)
    if means.size == 0:
        raise ValueError("GMM means are empty")
    if variances.size != means.size or weights.size != means.size:
        raise ValueError("GMM parameter shape mismatch")

    weights = _normalize_weights(weights)
    x = y.reshape(-1, 1)
    log_norm = -0.5 * (
        np.log(2.0 * np.pi * variances).reshape(1, -1)
        + ((x - means.reshape(1, -1)) ** 2) / variances.reshape(1, -1)
    )
    log_prob = log_norm + np.log(weights).reshape(1, -1)
    return np.argmax(log_prob, axis=1).astype(np.int64)


def estimate_ar1_params(
    gmm_params: Dict[str, object],
    training_power_traces: Sequence[np.ndarray],
    training_labels_traces: Sequence[np.ndarray],
    K: int,
    min_run_length: int = AR1_MIN_RUN_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.clip(np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1), a_min=1e-12, a_max=None)
    if means.size != int(K) or variances.size != int(K):
        raise ValueError(f"GMM parameter size mismatch for K={K}")

    state_run_residuals: Dict[int, List[np.ndarray]] = {k: [] for k in range(int(K))}
    run_min = int(max(2, min_run_length))
    for power, labels in zip(training_power_traces, training_labels_traces):
        p = np.asarray(power, dtype=np.float64).reshape(-1)
        s = np.asarray(labels, dtype=np.int64).reshape(-1)
        n = int(min(len(p), len(s)))
        if n < 2:
            continue
        p = p[:n]
        s = s[:n]

        run_start = 0
        for i in range(1, n + 1):
            if i == n or s[i] != s[run_start]:
                state = int(s[run_start])
                run_len = int(i - run_start)
                if run_len >= run_min and 0 <= state < int(K):
                    run_power = p[run_start:i]
                    run_residuals = run_power - float(means[state])
                    state_run_residuals[state].append(run_residuals.astype(np.float64))
                run_start = i

    phi = np.zeros((int(K),), dtype=np.float64)
    sigma_marginal = np.sqrt(variances).astype(np.float64)
    sigma_innov = np.zeros((int(K),), dtype=np.float64)

    for k_idx in range(int(K)):
        runs = state_run_residuals[k_idx]
        if len(runs) == 0:
            phi[k_idx] = 0.0
            sigma_innov[k_idx] = sigma_marginal[k_idx]
            continue

        numer = 0.0
        denom = 0.0
        for r in runs:
            if r.size < 2:
                continue
            numer += float(np.sum(r[:-1] * r[1:]))
            denom += float(np.sum(r[:-1] ** 2))
        if denom > 1e-12:
            phi_raw = numer / denom
            phi[k_idx] = float(np.clip(phi_raw, 0.0, 0.99))
        else:
            phi[k_idx] = 0.0

        sigma_innov[k_idx] = float(
            sigma_marginal[k_idx] * np.sqrt(max(1e-12, 1.0 - (phi[k_idx] ** 2)))
        )

    return phi, sigma_innov, sigma_marginal


def generate_gmm_bigru_trace_ar1_thresholded(
    *,
    logits: Union[np.ndarray, torch.Tensor],
    gmm_params: Dict[str, object],
    phi: np.ndarray,
    sigma_innov: np.ndarray,
    sigma_marginal: np.ndarray,
    p0: float,
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    phi_threshold: float = AR1_PHI_THRESHOLD_DEFAULT,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    if isinstance(logits, torch.Tensor):
        z = _tensor_to_numpy(logits, dtype=np.float64)
    else:
        z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 2:
        raise ValueError(f"logits must have shape (T,K) or (1,T,K); got {z.shape}")

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    k = int(means.size)
    if z.shape[1] != k:
        raise ValueError(f"logits K mismatch: got {z.shape[1]} but GMM has {k}")

    phi_arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    sigma_innov_arr = np.asarray(sigma_innov, dtype=np.float64).reshape(-1)
    sigma_marginal_arr = np.asarray(sigma_marginal, dtype=np.float64).reshape(-1)
    if phi_arr.size != k or sigma_innov_arr.size != k or sigma_marginal_arr.size != k:
        raise ValueError(f"phi/sigma arrays size mismatch for K={k}")

    use_ar1 = phi_arr >= float(phi_threshold)
    phi_gen = np.where(use_ar1, phi_arr, 0.0).astype(np.float64)
    sigma_gen = np.where(use_ar1, sigma_innov_arr, sigma_marginal_arr).astype(np.float64)

    probs = _softmax_np(z)
    mode = str(decode_mode).strip().lower()
    if mode not in {"stochastic", "argmax"}:
        raise ValueError(f"decode_mode must be 'stochastic' or 'argmax'; got {decode_mode}")

    rng = np.random.default_rng(seed)
    if mode == "argmax":
        states_raw = np.argmax(probs, axis=-1).astype(np.int64)
    else:
        states_raw = np.asarray([rng.choice(k, p=probs_t) for probs_t in probs], dtype=np.int64)
    states = _median_filter_states(states_raw, int(median_filter_window))

    t_horizon = int(z.shape[0])
    power = np.zeros((t_horizon,), dtype=np.float64)
    p_prev = float(p0)
    for i in range(t_horizon):
        s = int(states[i])
        mu = float(means[s])
        p_t = float(mu + (phi_gen[s] * (p_prev - mu)) + float(rng.normal(0.0, sigma_gen[s])))
        power[i] = p_t
        p_prev = p_t
    power = _apply_clamp(power, clamp_range)

    return {
        "power_w": power.astype(np.float64),
        "states": states.astype(np.int64),
        "states_raw": states_raw.astype(np.int64),
        "probs": probs.astype(np.float64),
        "use_ar1": use_ar1.astype(bool),
        "phi_gen": phi_gen.astype(np.float64),
        "sigma_gen": sigma_gen.astype(np.float64),
    }


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _write_csv(path: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_existing_path(path_str: str, base_dir: str) -> Optional[str]:
    raw = Path(path_str)
    if raw.is_absolute():
        return str(raw) if raw.exists() else None
    local = Path(path_str)
    if local.exists():
        return str(local)
    from_base = Path(base_dir) / raw
    if from_base.exists():
        return str(from_base)
    return None


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(text).strip())


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _parse_config_ids(config_ids: Optional[Sequence[str]], default_ids: Sequence[str]) -> List[str]:
    if not config_ids:
        return [str(x) for x in default_ids]
    out: List[str] = []
    for token in config_ids:
        if token is None:
            continue
        out.extend([x.strip() for x in str(token).split(",") if x.strip()])
    deduped: List[str] = []
    seen = set()
    for cid in out:
        if cid in seen:
            continue
        deduped.append(cid)
        seen.add(cid)
    return deduped


def _resolve_device(device: Optional[Union[torch.device, str]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _apply_clamp(power_w: np.ndarray, clamp_range: Optional[Tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(power_w, dtype=np.float64).reshape(-1)
    if clamp_range is None:
        return arr
    lo = float(clamp_range[0])
    hi = float(clamp_range[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return arr
    margin = 0.05 * (hi - lo)
    return np.clip(arr, lo - margin, hi + margin).astype(np.float64)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, a_min=1e-12, a_max=None)
    s = float(np.sum(w))
    if s <= EPS:
        return np.full_like(w, 1.0 / float(max(1, w.size)))
    return (w / s).astype(np.float64)


def _coerce_dataset_x_norm_features(
    x_norm_obj: object,
    *,
    input_dim: int,
) -> np.ndarray:
    x = np.asarray(x_norm_obj)
    if x.ndim != 2:
        raise ValueError(f"dataset_x_norm must be 2D; got shape {x.shape}")
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] < int(input_dim):
        raise ValueError(
            f"dataset_x_norm has insufficient feature dims: got {x.shape[1]} < required {int(input_dim)}"
        )
    if x.shape[1] > int(input_dim):
        x = x[:, : int(input_dim)]
    return np.asarray(x, dtype=np.float32)


def _prune_top_energy_rows(
    per_trace_rows: Sequence[Dict[str, object]],
    *,
    top_k: int,
    min_retained: int,
) -> int:
    k = int(max(0, top_k))
    if k <= 0:
        return 0
    keep_min = int(max(1, min_retained))

    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in per_trace_rows:
        if str(row.get("status", "")) != "ok":
            continue
        key = (str(row.get("config_id", "")), str(row.get("strategy", "")))
        groups.setdefault(key, []).append(row)

    num_pruned = 0
    for _, rows in groups.items():
        metric_rows = []
        for row in rows:
            try:
                val = float(row.get("abs_delta_energy_pct", float("nan")))
            except Exception:
                continue
            if np.isfinite(val):
                metric_rows.append((val, row))
        if len(metric_rows) <= keep_min:
            continue

        metric_rows.sort(key=lambda x: x[0], reverse=True)
        max_prunable = max(0, len(metric_rows) - keep_min)
        prune_n = int(min(k, max_prunable))
        if prune_n <= 0:
            continue
        for _, row in metric_rows[:prune_n]:
            row["status"] = "pruned"
            row["reason"] = "pruned_top_energy_error"
            num_pruned += 1

    return int(num_pruned)


def _display_label(config_id: str, moe_config_id: str) -> str:
    m = MODEL_RE.match(str(config_id).strip())
    if m is None:
        return str(config_id)

    family = str(m.group(1)).lower()
    size = int(m.group(2))
    hw = str(m.group(3)).upper()
    tp = int(m.group(4))

    if "llama-3" in family:
        model = f"Llama-3.1-{size}B"
    elif "deepseek-r1-distill" in family:
        model = f"DeepSeek-R1-Distill-{size}B"
    elif "gpt-oss" in family:
        model = f"GPT-OSS-{size}B"
    else:
        model = f"{m.group(1)}-{size}B"

    if str(config_id) == str(moe_config_id):
        model = f"{model} (MoE proxy)"

    return f"{model}\n{hw} TP={tp}"


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "run_manifest": str(repo_root / "results" / "continuous_v1_gmm_bigru" / "k10_f2" / "run_manifest.json"),
        "experimental_manifest": str(repo_root / "results" / "experimental_continuous_v1" / "manifest.json"),
        "throughput_db": str(repo_root / "model" / "config" / "throughput_database.json"),
        "pair_manifest_csv": str(repo_root / "results" / "stage0" / "pair_manifest.csv"),
        "ar1_params_dir": str(repo_root / "results" / "continuous_v1_gmm_bigru" / "k10_f2_ar1_thresh" / "ar1_params"),
        "out_per_trace_csv": str(repo_root / "results" / "eval_paper" / "appendix_c1_decomposition_per_trace.csv"),
        "out_summary_csv": str(repo_root / "results" / "eval_paper" / "appendix_c1_decomposition_summary.csv"),
        "out_manifest_json": str(repo_root / "results" / "eval_paper" / "appendix_c1_manifest.json"),
        "out_figure_pdf": str(repo_root / "figures" / "appendix_c1_decomposition_fidelity.pdf"),
    }


def _load_pair_manifest_map(pair_manifest_csv: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    base_dir = str(Path(pair_manifest_csv).resolve().parent)
    with open(pair_manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip() != "matched":
                continue
            key = str(row.get("pair_key", "")).strip()
            json_path_raw = str(row.get("json_path", "")).strip()
            if key == "" or json_path_raw == "":
                continue
            resolved = _resolve_existing_path(json_path_raw, base_dir)
            if resolved is not None:
                out[key] = resolved
    return out


def _synthesize_request_timestamps(payload: Dict[str, object], n: int) -> Optional[List[float]]:
    if n <= 0:
        return []

    duration = _safe_float(payload.get("duration"))
    if duration is not None and duration > 0:
        step = float(duration) / float(max(n, 1))
        if step > 0:
            values = (np.arange(n, dtype=np.float64) + 0.5) * step + 1.0
            return [float(x) for x in values]

    request_rate = _safe_float(payload.get("request_rate"))
    poisson_rate = _safe_float(payload.get("poisson_rate"))
    rate = request_rate if request_rate is not None else poisson_rate
    if rate is not None and rate > 0:
        step = 1.0 / float(rate)
        values = (np.arange(n, dtype=np.float64) + 1.0) * step + 1.0
        return [float(x) for x in values]
    return None


def _build_requests_from_stage0_json(
    request_json_path: str,
    *,
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
) -> List[Dict[str, float]]:
    payload = _load_json(request_json_path)
    required = ("input_lens", "output_lens")
    missing = [k for k in required if not isinstance(payload.get(k), list)]
    if missing:
        raise ValueError(f"request json missing arrays: {missing}")

    input_lens = payload["input_lens"]
    output_lens = payload["output_lens"]
    n_base = int(min(len(input_lens), len(output_lens)))

    request_timestamps_raw = payload.get("request_timestamps")
    if isinstance(request_timestamps_raw, list):
        n = int(min(n_base, len(request_timestamps_raw)))
        request_timestamps = request_timestamps_raw[:n]
    else:
        n = int(n_base)
        synth = _synthesize_request_timestamps(payload, n)
        if synth is None:
            raise ValueError("request json missing arrays: ['request_timestamps']")
        request_timestamps = synth

    if n <= 0:
        raise ValueError("request arrays are empty after alignment")

    arrivals = np.asarray(request_timestamps[:n], dtype=np.float64) - float(power_start_epoch_s)
    if arrivals.size > 0 and (
        float(np.min(arrivals)) < -float(dt)
        or float(np.max(arrivals)) > float(trace_duration_s) + float(dt)
    ):
        arrivals = arrivals - float(np.min(arrivals))

    out: List[Dict[str, float]] = []
    for i in range(n):
        a = float(arrivals[i])
        nin = float(input_lens[i])
        nout = float(output_lens[i])
        if not (np.isfinite(a) and np.isfinite(nin) and np.isfinite(nout)):
            continue
        out.append(
            {
                "arrival_time": float(a),
                "input_tokens": float(max(0.0, nin)),
                "output_tokens": float(max(0.0, nout)),
            }
        )

    if len(out) == 0:
        raise ValueError("no valid requests after filtering")
    return out


def _extract_norm_for_eval(norm_payload: Dict[str, object]) -> Dict[str, float]:
    required = (
        "active_mean",
        "active_std",
        "t_arrive_log_mean",
        "t_arrive_log_std",
        "power_mean",
        "power_std",
        "power_min",
        "power_max",
        "delta_A_mean",
        "delta_A_std",
    )
    missing = [k for k in required if k not in norm_payload]
    if missing:
        raise ValueError(f"Norm params missing keys: {missing}")

    out = {k: float(norm_payload[k]) for k in required}
    if out["active_std"] <= 0.0:
        raise ValueError("active_std must be positive")
    if out["t_arrive_log_std"] <= 0.0:
        raise ValueError("t_arrive_log_std must be positive")
    if out["power_std"] <= 0.0:
        raise ValueError("power_std must be positive")
    if out["delta_A_std"] <= 0.0:
        raise ValueError("delta_A_std must be positive")
    return out


def _resolve_checkpoint_norm_gmm_paths(config_entry: Dict[str, object], base_dir: str) -> Tuple[str, str, str]:
    checkpoint_raw = str(config_entry.get("checkpoint_path", ""))
    norm_raw = str(config_entry.get("norm_params_path", ""))
    gmm_raw = str(config_entry.get("gmm_params_path", ""))

    checkpoint_path = _resolve_existing_path(checkpoint_raw, base_dir)
    norm_path = _resolve_existing_path(norm_raw, base_dir)
    gmm_path = _resolve_existing_path(gmm_raw, base_dir)

    if checkpoint_path is None:
        raise ValueError(f"Checkpoint path not found: {checkpoint_raw}")
    if norm_path is None:
        raise ValueError(f"Norm params path not found: {norm_raw}")
    if gmm_path is None:
        raise ValueError(f"GMM path not found: {gmm_raw}")
    return checkpoint_path, norm_path, gmm_path


def _resolve_throughput(throughput_payload: Dict[str, object], config_id: str) -> Dict[str, float]:
    cfgs = throughput_payload.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("Invalid throughput database format")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"config_id '{config_id}' not found in throughput DB")

    prefill = float(row.get("prefill_rate_median_toks_per_s", float("nan")))
    decode = float(row.get("decode_rate_median_toks_per_s", float("nan")))
    if (not np.isfinite(prefill)) or prefill <= 0.0:
        raise ValueError(f"Invalid prefill throughput for '{config_id}'")
    if (not np.isfinite(decode)) or decode <= 0.0:
        raise ValueError(f"Invalid decode throughput for '{config_id}'")
    return {"lambda_prefill": prefill, "lambda_decode": decode}


def _resolve_experimental_paths(
    experimental_manifest: Dict[str, object],
    *,
    config_id: str,
    experimental_base: str,
) -> Tuple[str, str]:
    cfgs = experimental_manifest.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("Invalid experimental manifest format")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"config_id '{config_id}' not found in experimental manifest")

    dataset_path = _resolve_existing_path(str(row.get("dataset_npz", "")), experimental_base)
    split_path = _resolve_existing_path(str(row.get("split_json", "")), experimental_base)
    if dataset_path is None:
        raise ValueError(f"Dataset path not found for '{config_id}'")
    if split_path is None:
        raise ValueError(f"Split path not found for '{config_id}'")
    return dataset_path, split_path


def _load_model(
    *,
    checkpoint_path: str,
    k: int,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
) -> GRUClassifier:
    model = GRUClassifier(
        Dx=int(input_dim),
        K=int(k),
        H=int(hidden_dim),
        num_layers=int(max(1, num_layers)),
    ).to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    if isinstance(state, dict) and "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    model.load_state_dict(state)
    model.eval()
    return model


def _generate_oracle_dense(
    *,
    states: np.ndarray,
    gmm_params: Mapping[str, object],
    seed: int,
    clamp_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    z = np.asarray(states, dtype=np.int64).reshape(-1)
    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
    std = np.sqrt(np.clip(variances, a_min=1e-12, a_max=None))

    if z.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    if np.min(z) < 0 or np.max(z) >= means.size:
        raise ValueError("oracle states out of GMM component range")

    rng = np.random.default_rng(int(seed))
    power = rng.normal(loc=means[z], scale=std[z]).astype(np.float64)
    return _apply_clamp(power, clamp_range)


def _generate_oracle_moe_ar1(
    *,
    states: np.ndarray,
    p0: float,
    gmm_params: Mapping[str, object],
    phi: np.ndarray,
    sigma_innov: np.ndarray,
    sigma_marginal: np.ndarray,
    phi_threshold: float,
    seed: int,
    clamp_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    z = np.asarray(states, dtype=np.int64).reshape(-1)
    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    k = int(means.size)
    if z.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    if np.min(z) < 0 or np.max(z) >= k:
        raise ValueError("oracle states out of GMM component range")

    phi_arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    sigma_innov_arr = np.asarray(sigma_innov, dtype=np.float64).reshape(-1)
    sigma_marginal_arr = np.asarray(sigma_marginal, dtype=np.float64).reshape(-1)
    if phi_arr.size != k or sigma_innov_arr.size != k or sigma_marginal_arr.size != k:
        raise ValueError("AR1 parameter length mismatch")

    use_ar1 = phi_arr >= float(phi_threshold)
    phi_eff = np.where(use_ar1, phi_arr, 0.0).astype(np.float64)
    sigma_eff = np.where(use_ar1, sigma_innov_arr, sigma_marginal_arr).astype(np.float64)

    rng = np.random.default_rng(int(seed))
    out = np.zeros((z.size,), dtype=np.float64)
    p_prev = float(p0)
    for i, s_raw in enumerate(z):
        s = int(s_raw)
        mu = float(means[s])
        p_t = float(mu + (phi_eff[s] * (p_prev - mu)) + float(rng.normal(0.0, sigma_eff[s])))
        out[i] = p_t
        p_prev = p_t
    return _apply_clamp(out, clamp_range)


def _generate_marginal(
    *,
    n_timesteps: int,
    gmm_params: Mapping[str, object],
    seed: int,
    clamp_range: Optional[Tuple[float, float]],
) -> np.ndarray:
    n = int(n_timesteps)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
    weights = np.asarray(gmm_params["weights"], dtype=np.float64).reshape(-1)
    if means.size == 0:
        raise ValueError("GMM means are empty")
    if variances.size != means.size or weights.size != means.size:
        raise ValueError("GMM parameter shape mismatch")

    probs = _normalize_weights(weights)
    std = np.sqrt(np.clip(variances, a_min=1e-12, a_max=None))
    rng = np.random.default_rng(int(seed))
    z = rng.choice(int(means.size), size=n, p=probs)
    power = rng.normal(loc=means[z], scale=std[z]).astype(np.float64)
    return _apply_clamp(power, clamp_range)


def _plot_figure_c1(
    *,
    summary_rows: Sequence[Mapping[str, object]],
    config_ids_ordered: Sequence[str],
    config_labels: Mapping[str, str],
    out_figure_pdf: str,
) -> Dict[str, object]:
    # Build lookup keyed by (config_id, strategy)
    by_key: Dict[Tuple[str, str], Mapping[str, object]] = {}
    for row in summary_rows:
        by_key[(str(row["config_id"]), str(row["strategy"]))] = row

    x = np.arange(len(config_ids_ordered), dtype=np.float64)
    width = 0.23
    offsets = {
        "oracle": -width,
        "predicted": 0.0,
        "marginal": width,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharex=True)

    # Left: ACF R^2
    ax0 = axes[0]
    for strategy in STRATEGY_ORDER:
        heights: List[float] = []
        err_lo: List[float] = []
        err_hi: List[float] = []
        for cid in config_ids_ordered:
            row = by_key.get((str(cid), strategy))
            if row is None or str(row.get("status", "")) != "evaluated":
                heights.append(float("nan"))
                err_lo.append(0.0)
                err_hi.append(0.0)
                continue
            med = float(row["acf_r2_median"])
            lo = float(row["acf_r2_min"])
            hi = float(row["acf_r2_max"])
            heights.append(med)
            err_lo.append(max(0.0, med - lo) if np.isfinite(med) and np.isfinite(lo) else 0.0)
            err_hi.append(max(0.0, hi - med) if np.isfinite(med) and np.isfinite(hi) else 0.0)

        ax0.bar(
            x + offsets[strategy],
            heights,
            width=width,
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_DISPLAY[strategy],
            yerr=np.asarray([err_lo, err_hi], dtype=np.float64),
            capsize=3,
            linewidth=0.0,
            ecolor="#3a3a3a",
            alpha=0.95,
        )

    ax0.set_ylabel("ACF R²")
    ax0.set_title("Temporal Fidelity")
    ax0.grid(True, axis="y", alpha=0.25)

    # Right: |DeltaEnergy| (%)
    ax1 = axes[1]
    for strategy in STRATEGY_ORDER:
        heights = []
        err_lo = []
        err_hi = []
        for cid in config_ids_ordered:
            row = by_key.get((str(cid), strategy))
            if row is None or str(row.get("status", "")) != "evaluated":
                heights.append(float("nan"))
                err_lo.append(0.0)
                err_hi.append(0.0)
                continue
            med = float(row["abs_delta_energy_pct_median"])
            lo = float(row["abs_delta_energy_pct_min"])
            hi = float(row["abs_delta_energy_pct_max"])
            heights.append(med)
            err_lo.append(max(0.0, med - lo) if np.isfinite(med) and np.isfinite(lo) else 0.0)
            err_hi.append(max(0.0, hi - med) if np.isfinite(med) and np.isfinite(hi) else 0.0)

        ax1.bar(
            x + offsets[strategy],
            heights,
            width=width,
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_DISPLAY[strategy],
            yerr=np.asarray([err_lo, err_hi], dtype=np.float64),
            capsize=3,
            linewidth=0.0,
            ecolor="#3a3a3a",
            alpha=0.95,
        )

    ax1.set_ylabel("|ΔEnergy| (%)")
    ax1.set_title("Energy Accuracy")
    ax1.grid(True, axis="y", alpha=0.25)

    tick_labels = [str(config_labels.get(str(cid), str(cid))) for cid in config_ids_ordered]
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    _ensure_dir(os.path.dirname(out_figure_pdf) or ".")
    fig.savefig(out_figure_pdf, bbox_inches="tight")
    plt.close(fig)

    return {
        "num_subplots": 2,
        "num_config_groups": int(len(config_ids_ordered)),
        "num_strategies_per_group": 3,
        "strategy_order": [str(x) for x in STRATEGY_ORDER],
        "bar_colors": {k: v for k, v in STRATEGY_COLORS.items()},
    }


def run_appendix_decomposition_fidelity(
    *,
    run_manifest: str,
    experimental_manifest: str,
    throughput_db: str,
    pair_manifest_csv: str,
    ar1_params_dir: str,
    config_ids: Sequence[str],
    moe_config_id: str,
    num_traces_per_config: int,
    base_seed: int,
    acf_max_lag: int,
    prune_top_energy_per_group: int,
    min_traces_after_prune: int,
    out_per_trace_csv: str,
    out_summary_csv: str,
    out_figure_pdf: str,
    out_manifest_json: str,
    device: str,
) -> Dict[str, object]:
    if int(num_traces_per_config) <= 0:
        raise ValueError("num_traces_per_config must be >= 1")
    if int(prune_top_energy_per_group) < 0:
        raise ValueError("prune_top_energy_per_group must be >= 0")
    if int(min_traces_after_prune) <= 0:
        raise ValueError("min_traces_after_prune must be >= 1")

    run_payload = _load_json(run_manifest)
    run_cfgs = run_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    run_manifest_base = str(Path(run_manifest).resolve().parent)

    experimental_payload = _load_json(experimental_manifest)
    experimental_base = str(Path(experimental_manifest).resolve().parent)
    throughput_payload = _load_json(throughput_db)
    pair_map = _load_pair_manifest_map(pair_manifest_csv)

    resolved_device = _resolve_device(device)

    per_trace_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    config_labels: Dict[str, str] = {
        str(cid): _display_label(str(cid), str(moe_config_id)) for cid in config_ids
    }

    for config_order, config_id in enumerate(config_ids):
        cfg_row = run_cfgs.get(config_id)
        if not isinstance(cfg_row, dict):
            failures.append(
                {
                    "scope": "config",
                    "config_id": config_id,
                    "reason": "config_not_in_run_manifest",
                }
            )
            continue
        if str(cfg_row.get("status", "")) != "trained":
            failures.append(
                {
                    "scope": "config",
                    "config_id": config_id,
                    "reason": f"config_status_{cfg_row.get('status', 'unknown')}",
                }
            )
            continue

        try:
            checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(cfg_row, run_manifest_base)
            norm_payload = _load_json(norm_path)
            norm_cfg = _extract_norm_for_eval(norm_payload)
            gmm_payload = _load_json(gmm_path)
            gmm_cfg = load_gmm_params_json_dict(gmm_payload)
            throughput = _resolve_throughput(throughput_payload, config_id)

            k = int(cfg_row.get("k", gmm_cfg["k"]))
            if k != int(gmm_cfg["k"]):
                raise ValueError(f"k mismatch between run manifest ({k}) and gmm payload ({int(gmm_cfg['k'])})")

            feature_set = str(cfg_row.get("feature_set", norm_payload.get("feature_set", "f2"))).lower()
            if feature_set not in {"f2", "f3"}:
                raise ValueError(f"invalid feature_set: {feature_set}")
            input_dim = int(cfg_row.get("input_dim", 2 if feature_set == "f2" else 3))
            hidden_dim = int(cfg_row.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
            num_layers = int(cfg_row.get("num_layers", norm_payload.get("num_layers", 1)))

            model = _load_model(
                checkpoint_path=checkpoint_path,
                k=k,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=resolved_device,
            )

            dataset_path, split_path = _resolve_experimental_paths(
                experimental_payload,
                config_id=config_id,
                experimental_base=experimental_base,
            )
            split_payload = _load_json(split_path)
            train_indices = [int(x) for x in split_payload.get("train_indices", [])]
            test_indices_all = [int(x) for x in split_payload.get("test_indices", [])]
            if len(test_indices_all) == 0:
                raise ValueError("empty_test_split")

            test_indices = test_indices_all[: int(num_traces_per_config)]

            with np.load(dataset_path, allow_pickle=True) as data:
                pair_key_arr = np.asarray(data["pair_key"], dtype=object)
                power_arr = np.asarray(data["power"], dtype=object)
                power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
                rate_arr = np.asarray(data["rate"], dtype=object) if "rate" in data else np.asarray([], dtype=object)
                x_norm_arr = np.asarray(data["x_norm"], dtype=object) if "x_norm" in data else np.asarray([], dtype=object)
                dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)

            if dt_arr.size == 0:
                raise ValueError("dataset_dt_missing")
            dt = float(dt_arr[0])
            if (not np.isfinite(dt)) or dt <= 0.0:
                raise ValueError(f"invalid_dt:{dt}")

            n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))
            clamp_range = (float(norm_cfg["power_min"]), float(norm_cfg["power_max"]))

            is_moe_proxy = str(config_id) == str(moe_config_id)
            phi = np.zeros((int(k),), dtype=np.float64)
            sigma_innov = np.zeros((int(k),), dtype=np.float64)
            sigma_marginal = np.sqrt(np.clip(np.asarray(gmm_cfg["variances"], dtype=np.float64), a_min=1e-12, a_max=None))
            phi_threshold = float(AR1_PHI_THRESHOLD_DEFAULT)
            ar1_source = "not_required"

            if is_moe_proxy:
                ar1_path = Path(ar1_params_dir) / f"{_safe_slug(config_id)}_ar1_params.json"
                if ar1_path.exists():
                    ar1_payload = _load_json(str(ar1_path))
                    phi_loaded = np.asarray(ar1_payload.get("phi", []), dtype=np.float64).reshape(-1)
                    sigma_innov_loaded = np.asarray(ar1_payload.get("sigma_innov", []), dtype=np.float64).reshape(-1)
                    sigma_marginal_loaded = np.asarray(ar1_payload.get("sigma_marginal", []), dtype=np.float64).reshape(-1)
                    if (
                        phi_loaded.size == int(k)
                        and sigma_innov_loaded.size == int(k)
                        and sigma_marginal_loaded.size == int(k)
                    ):
                        phi = phi_loaded
                        sigma_innov = sigma_innov_loaded
                        sigma_marginal = sigma_marginal_loaded
                        phi_threshold = float(ar1_payload.get("phi_threshold", AR1_PHI_THRESHOLD_DEFAULT))
                        ar1_source = f"file:{str(ar1_path)}"
                    else:
                        ar1_source = "file_invalid_shape_estimated"
                if ar1_source != f"file:{str(ar1_path)}":
                    training_power_traces: List[np.ndarray] = []
                    training_labels_traces: List[np.ndarray] = []
                    for idx in train_indices:
                        if idx < 0 or idx >= n_total:
                            continue
                        p_train = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
                        if p_train.size <= 0:
                            continue
                        labels_train = predict_sorted_gmm_labels_from_params(p_train, gmm_cfg)
                        training_power_traces.append(p_train.astype(np.float64))
                        training_labels_traces.append(np.asarray(labels_train, dtype=np.int64).reshape(-1))
                    if len(training_power_traces) == 0:
                        raise ValueError("no_training_traces_for_ar1_estimation")
                    phi, sigma_innov, sigma_marginal = estimate_ar1_params(
                        gmm_params=gmm_cfg,
                        training_power_traces=training_power_traces,
                        training_labels_traces=training_labels_traces,
                        K=int(k),
                    )
                    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
                    sigma_innov = np.asarray(sigma_innov, dtype=np.float64).reshape(-1)
                    sigma_marginal = np.asarray(sigma_marginal, dtype=np.float64).reshape(-1)
                    phi_threshold = float(AR1_PHI_THRESHOLD_DEFAULT)
                    if ar1_source == "file_invalid_shape_estimated":
                        ar1_source = "estimated_from_train_after_invalid_file"
                    else:
                        ar1_source = "estimated_from_train"

            for local_idx, test_idx in enumerate(test_indices):
                if test_idx < 0 or test_idx >= n_total:
                    for strategy in STRATEGY_ORDER:
                        per_trace_rows.append(
                            {
                                "config_id": config_id,
                                "config_label": config_labels[config_id],
                                "config_order": int(config_order),
                                "strategy": strategy,
                                "strategy_display": STRATEGY_DISPLAY[strategy],
                                "strategy_order": int(STRATEGY_ORDER.index(strategy)),
                                "is_moe_proxy": bool(is_moe_proxy),
                                "trace_idx": int(test_idx),
                                "test_order": int(local_idx),
                                "pair_key": "",
                                "rate": "",
                                "seed": int(base_seed + (config_order * 1000) + (STRATEGY_ORDER.index(strategy) * 100) + local_idx),
                                "status": "failed",
                                "reason": "test_index_out_of_bounds",
                                "num_points": 0,
                                "dt": float(dt),
                                "acf_r2": float("nan"),
                                "delta_energy_pct": float("nan"),
                                "abs_delta_energy_pct": float("nan"),
                                "ar1_source": ar1_source,
                            }
                        )
                    continue

                pair_key = str(pair_key_arr[test_idx])
                rate = str(rate_arr[test_idx]) if test_idx < len(rate_arr) else ""
                p = np.asarray(power_arr[test_idx], dtype=np.float64).reshape(-1)

                if p.size < 2:
                    for strategy in STRATEGY_ORDER:
                        per_trace_rows.append(
                            {
                                "config_id": config_id,
                                "config_label": config_labels[config_id],
                                "config_order": int(config_order),
                                "strategy": strategy,
                                "strategy_display": STRATEGY_DISPLAY[strategy],
                                "strategy_order": int(STRATEGY_ORDER.index(strategy)),
                                "is_moe_proxy": bool(is_moe_proxy),
                                "trace_idx": int(test_idx),
                                "test_order": int(local_idx),
                                "pair_key": pair_key,
                                "rate": rate,
                                "seed": int(base_seed + (config_order * 1000) + (STRATEGY_ORDER.index(strategy) * 100) + local_idx),
                                "status": "failed",
                                "reason": "trace_too_short",
                                "num_points": 0,
                                "dt": float(dt),
                                "acf_r2": float("nan"),
                                "delta_energy_pct": float("nan"),
                                "abs_delta_energy_pct": float("nan"),
                                "ar1_source": ar1_source,
                            }
                        )
                    continue

                gt_full = p[1:].astype(np.float64)
                p0 = float(p[0])

                json_path = pair_map.get(pair_key)
                request_source = "missing_pair_key"
                feature_source = "uninitialized"
                features_norm: Optional[np.ndarray] = None

                if json_path is not None:
                    try:
                        requests = _build_requests_from_stage0_json(
                            json_path,
                            power_start_epoch_s=float(power_start_arr[test_idx]),
                            trace_duration_s=float(p.size * dt),
                            dt=float(dt),
                        )
                        feat = build_rollout_features_from_requests(
                            requests=requests,
                            throughput=throughput,
                            norm=norm_cfg,
                            T=int(gt_full.size),
                            dt=float(dt),
                            feature_set=feature_set,
                        )
                        features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
                        if features_norm.ndim != 2 or features_norm.shape[1] != int(input_dim):
                            raise ValueError(f"invalid_feature_shape:{features_norm.shape}")
                        request_source = "stage0_json"
                        feature_source = "stage0_rebuilt_features"
                    except Exception as exc:
                        request_source = f"stage0_json_error:{type(exc).__name__}:{exc}"
                        feature_source = "stage0_rebuild_failed"

                if features_norm is None:
                    if test_idx < len(x_norm_arr):
                        try:
                            features_norm = _coerce_dataset_x_norm_features(
                                x_norm_arr[test_idx],
                                input_dim=int(input_dim),
                            )
                            feature_source = "dataset_x_norm_fallback"
                        except Exception as exc:
                            feature_source = f"dataset_x_norm_error:{type(exc).__name__}:{exc}"
                    else:
                        feature_source = "dataset_x_norm_missing"

                if features_norm is None:
                    for strategy in STRATEGY_ORDER:
                        per_trace_rows.append(
                            {
                                "config_id": config_id,
                                "config_label": config_labels[config_id],
                                "config_order": int(config_order),
                                "strategy": strategy,
                                "strategy_display": STRATEGY_DISPLAY[strategy],
                                "strategy_order": int(STRATEGY_ORDER.index(strategy)),
                                "is_moe_proxy": bool(is_moe_proxy),
                                "trace_idx": int(test_idx),
                                "test_order": int(local_idx),
                                "pair_key": pair_key,
                                "rate": rate,
                                "seed": int(base_seed + (config_order * 1000) + (STRATEGY_ORDER.index(strategy) * 100) + local_idx),
                                "status": "failed",
                                "reason": (
                                    f"feature_build_error:request_source={request_source}:"
                                    f"feature_source={feature_source}"
                                ),
                                "num_points": 0,
                                "dt": float(dt),
                                "acf_r2": float("nan"),
                                "delta_energy_pct": float("nan"),
                                "abs_delta_energy_pct": float("nan"),
                                "ar1_source": ar1_source,
                                "request_source": request_source,
                                "feature_source": feature_source,
                            }
                        )
                    continue

                try:
                    with torch.no_grad():
                        try:
                            x = torch.from_numpy(features_norm)
                        except Exception:
                            x = torch.tensor(features_norm.tolist(), dtype=torch.float32)
                        x = x.to(device=resolved_device, dtype=torch.float32).unsqueeze(0)
                        logits = _tensor_to_numpy(model(x)[0], dtype=np.float64)
                except Exception as exc:
                    for strategy in STRATEGY_ORDER:
                        per_trace_rows.append(
                            {
                                "config_id": config_id,
                                "config_label": config_labels[config_id],
                                "config_order": int(config_order),
                                "strategy": strategy,
                                "strategy_display": STRATEGY_DISPLAY[strategy],
                                "strategy_order": int(STRATEGY_ORDER.index(strategy)),
                                "is_moe_proxy": bool(is_moe_proxy),
                                "trace_idx": int(test_idx),
                                "test_order": int(local_idx),
                                "pair_key": pair_key,
                                "rate": rate,
                                "seed": int(base_seed + (config_order * 1000) + (STRATEGY_ORDER.index(strategy) * 100) + local_idx),
                                "status": "failed",
                                "reason": f"logit_build_error:{type(exc).__name__}:{exc}",
                                "num_points": 0,
                                "dt": float(dt),
                                "acf_r2": float("nan"),
                                "delta_energy_pct": float("nan"),
                                "abs_delta_energy_pct": float("nan"),
                                "ar1_source": ar1_source,
                                "request_source": request_source,
                                "feature_source": feature_source,
                            }
                        )
                    continue

                n = int(min(gt_full.size, logits.shape[0]))
                if n <= 0:
                    for strategy in STRATEGY_ORDER:
                        per_trace_rows.append(
                            {
                                "config_id": config_id,
                                "config_label": config_labels[config_id],
                                "config_order": int(config_order),
                                "strategy": strategy,
                                "strategy_display": STRATEGY_DISPLAY[strategy],
                                "strategy_order": int(STRATEGY_ORDER.index(strategy)),
                                "is_moe_proxy": bool(is_moe_proxy),
                                "trace_idx": int(test_idx),
                                "test_order": int(local_idx),
                                "pair_key": pair_key,
                                "rate": rate,
                                "seed": int(base_seed + (config_order * 1000) + (STRATEGY_ORDER.index(strategy) * 100) + local_idx),
                                "status": "failed",
                                "reason": "empty_aligned_horizon",
                                "num_points": 0,
                                "dt": float(dt),
                                "acf_r2": float("nan"),
                                "delta_energy_pct": float("nan"),
                                "abs_delta_energy_pct": float("nan"),
                                "ar1_source": ar1_source,
                                "request_source": request_source,
                                "feature_source": feature_source,
                            }
                        )
                    continue

                gt = gt_full[:n].astype(np.float64)
                logits_n = np.asarray(logits[:n], dtype=np.float64)

                # Oracle states from measured power (posterior max label).
                oracle_states = predict_sorted_gmm_labels_from_params(gt, gmm_cfg).astype(np.int64)

                for strategy_idx, strategy in enumerate(STRATEGY_ORDER):
                    seed = int(base_seed + (config_order * 1000) + (strategy_idx * 100) + local_idx)
                    try:
                        if strategy == "oracle":
                            if is_moe_proxy:
                                pred = _generate_oracle_moe_ar1(
                                    states=oracle_states,
                                    p0=float(p0),
                                    gmm_params=gmm_cfg,
                                    phi=phi,
                                    sigma_innov=sigma_innov,
                                    sigma_marginal=sigma_marginal,
                                    phi_threshold=float(phi_threshold),
                                    seed=int(seed),
                                    clamp_range=clamp_range,
                                )
                            else:
                                pred = _generate_oracle_dense(
                                    states=oracle_states,
                                    gmm_params=gmm_cfg,
                                    seed=int(seed),
                                    clamp_range=clamp_range,
                                )
                        elif strategy == "predicted":
                            if is_moe_proxy:
                                gen = generate_gmm_bigru_trace_ar1_thresholded(
                                    logits=logits_n,
                                    gmm_params=gmm_cfg,
                                    phi=phi,
                                    sigma_innov=sigma_innov,
                                    sigma_marginal=sigma_marginal,
                                    p0=float(p0),
                                    seed=int(seed),
                                    decode_mode="stochastic",
                                    median_filter_window=1,
                                    phi_threshold=float(phi_threshold),
                                    clamp_range=clamp_range,
                                )
                            else:
                                gen = generate_gmm_bigru_trace(
                                    logits=logits_n,
                                    gmm_params=gmm_cfg,
                                    seed=int(seed),
                                    decode_mode="stochastic",
                                    median_filter_window=1,
                                    clamp_range=clamp_range,
                                )
                            pred = np.asarray(gen["power_w"], dtype=np.float64).reshape(-1)
                        elif strategy == "marginal":
                            pred = _generate_marginal(
                                n_timesteps=int(n),
                                gmm_params=gmm_cfg,
                                seed=int(seed),
                                clamp_range=clamp_range,
                            )
                        else:
                            raise ValueError(f"unknown strategy:{strategy}")

                        n_aligned = int(min(gt.size, pred.size))
                        if n_aligned <= 0:
                            raise ValueError("no_aligned_points")

                        gt_n = gt[:n_aligned]
                        pred_n = pred[:n_aligned]
                        metrics = compute_power_metrics(
                            gt_n,
                            pred_n,
                            dt=float(dt),
                            acf_max_lag=int(acf_max_lag),
                        )
                        delta_e = float(metrics["delta_energy_pct"])
                        per_trace_rows.append(
                            {
                                "config_id": config_id,
                                "config_label": config_labels[config_id],
                                "config_order": int(config_order),
                                "strategy": strategy,
                                "strategy_display": STRATEGY_DISPLAY[strategy],
                                "strategy_order": int(strategy_idx),
                                "is_moe_proxy": bool(is_moe_proxy),
                                "trace_idx": int(test_idx),
                                "test_order": int(local_idx),
                                "pair_key": pair_key,
                                "rate": rate,
                                "seed": int(seed),
                                "status": "ok",
                                "reason": "",
                                "num_points": int(n_aligned),
                                "dt": float(dt),
                                "acf_r2": float(metrics["acf_r2"]),
                                "delta_energy_pct": float(delta_e),
                                "abs_delta_energy_pct": float(abs(delta_e)),
                                "ar1_source": ar1_source,
                                "request_source": request_source,
                                "feature_source": feature_source,
                            }
                        )
                    except Exception as exc:
                        per_trace_rows.append(
                            {
                                "config_id": config_id,
                                "config_label": config_labels[config_id],
                                "config_order": int(config_order),
                                "strategy": strategy,
                                "strategy_display": STRATEGY_DISPLAY[strategy],
                                "strategy_order": int(strategy_idx),
                                "is_moe_proxy": bool(is_moe_proxy),
                                "trace_idx": int(test_idx),
                                "test_order": int(local_idx),
                                "pair_key": pair_key,
                                "rate": rate,
                                "seed": int(seed),
                                "status": "failed",
                                "reason": f"strategy_error:{type(exc).__name__}:{exc}",
                                "num_points": 0,
                                "dt": float(dt),
                                "acf_r2": float("nan"),
                                "delta_energy_pct": float("nan"),
                                "abs_delta_energy_pct": float("nan"),
                                "ar1_source": ar1_source,
                                "request_source": request_source,
                                "feature_source": feature_source,
                            }
                        )
        except Exception as exc:
            failures.append(
                {
                    "scope": "config",
                    "config_id": config_id,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

    num_pruned_rows = _prune_top_energy_rows(
        per_trace_rows,
        top_k=int(prune_top_energy_per_group),
        min_retained=int(min_traces_after_prune),
    )

    # Summary rows in fixed order.
    summary_rows: List[Dict[str, object]] = []
    for config_order, config_id in enumerate(config_ids):
        for strategy_idx, strategy in enumerate(STRATEGY_ORDER):
            target_rows = [
                r
                for r in per_trace_rows
                if str(r["config_id"]) == str(config_id)
                and str(r["strategy"]) == str(strategy)
                and str(r["status"]) == "ok"
            ]
            num_target = int(num_traces_per_config)
            if len(target_rows) == 0:
                failure_candidates = [
                    r
                    for r in per_trace_rows
                    if str(r["config_id"]) == str(config_id)
                    and str(r["strategy"]) == str(strategy)
                    and str(r["status"]) != "ok"
                ]
                reason = "no_successful_rows"
                if len(failure_candidates) > 0:
                    reason = str(failure_candidates[0].get("reason", reason))
                summary_rows.append(
                    {
                        "config_id": config_id,
                        "config_label": config_labels.get(config_id, config_id),
                        "config_order": int(config_order),
                        "strategy": strategy,
                        "strategy_display": STRATEGY_DISPLAY[strategy],
                        "strategy_order": int(strategy_idx),
                        "status": "failed",
                        "reason": reason,
                        "num_target_traces": int(num_target),
                        "num_eval_traces": 0,
                        "acf_r2_median": float("nan"),
                        "acf_r2_min": float("nan"),
                        "acf_r2_max": float("nan"),
                        "acf_r2_err_lo": float("nan"),
                        "acf_r2_err_hi": float("nan"),
                        "abs_delta_energy_pct_median": float("nan"),
                        "abs_delta_energy_pct_min": float("nan"),
                        "abs_delta_energy_pct_max": float("nan"),
                        "abs_delta_energy_pct_err_lo": float("nan"),
                        "abs_delta_energy_pct_err_hi": float("nan"),
                    }
                )
                continue

            acf_vals = np.asarray([float(r["acf_r2"]) for r in target_rows], dtype=np.float64)
            e_vals = np.asarray([float(r["abs_delta_energy_pct"]) for r in target_rows], dtype=np.float64)
            acf_vals = acf_vals[np.isfinite(acf_vals)]
            e_vals = e_vals[np.isfinite(e_vals)]

            if acf_vals.size == 0 or e_vals.size == 0:
                summary_rows.append(
                    {
                        "config_id": config_id,
                        "config_label": config_labels.get(config_id, config_id),
                        "config_order": int(config_order),
                        "strategy": strategy,
                        "strategy_display": STRATEGY_DISPLAY[strategy],
                        "strategy_order": int(strategy_idx),
                        "status": "failed",
                        "reason": "non_finite_metric_values",
                        "num_target_traces": int(num_target),
                        "num_eval_traces": int(len(target_rows)),
                        "acf_r2_median": float("nan"),
                        "acf_r2_min": float("nan"),
                        "acf_r2_max": float("nan"),
                        "acf_r2_err_lo": float("nan"),
                        "acf_r2_err_hi": float("nan"),
                        "abs_delta_energy_pct_median": float("nan"),
                        "abs_delta_energy_pct_min": float("nan"),
                        "abs_delta_energy_pct_max": float("nan"),
                        "abs_delta_energy_pct_err_lo": float("nan"),
                        "abs_delta_energy_pct_err_hi": float("nan"),
                    }
                )
                continue

            acf_med = float(np.median(acf_vals))
            acf_min = float(np.min(acf_vals))
            acf_max = float(np.max(acf_vals))
            e_med = float(np.median(e_vals))
            e_min = float(np.min(e_vals))
            e_max = float(np.max(e_vals))

            summary_rows.append(
                {
                    "config_id": config_id,
                    "config_label": config_labels.get(config_id, config_id),
                    "config_order": int(config_order),
                    "strategy": strategy,
                    "strategy_display": STRATEGY_DISPLAY[strategy],
                    "strategy_order": int(strategy_idx),
                    "status": "evaluated",
                    "reason": "",
                    "num_target_traces": int(num_target),
                    "num_eval_traces": int(len(target_rows)),
                    "acf_r2_median": float(acf_med),
                    "acf_r2_min": float(acf_min),
                    "acf_r2_max": float(acf_max),
                    "acf_r2_err_lo": float(max(0.0, acf_med - acf_min)),
                    "acf_r2_err_hi": float(max(0.0, acf_max - acf_med)),
                    "abs_delta_energy_pct_median": float(e_med),
                    "abs_delta_energy_pct_min": float(e_min),
                    "abs_delta_energy_pct_max": float(e_max),
                    "abs_delta_energy_pct_err_lo": float(max(0.0, e_med - e_min)),
                    "abs_delta_energy_pct_err_hi": float(max(0.0, e_max - e_med)),
                }
            )

    per_trace_fields = [
        "config_id",
        "config_label",
        "config_order",
        "strategy",
        "strategy_display",
        "strategy_order",
        "is_moe_proxy",
        "trace_idx",
        "test_order",
        "pair_key",
        "rate",
        "seed",
        "status",
        "reason",
        "num_points",
        "dt",
        "acf_r2",
        "delta_energy_pct",
        "abs_delta_energy_pct",
        "ar1_source",
        "request_source",
        "feature_source",
    ]
    for row in per_trace_rows:
        for f in per_trace_fields:
            row.setdefault(f, "")
    _write_csv(out_per_trace_csv, per_trace_rows, per_trace_fields)

    summary_fields = [
        "config_id",
        "config_label",
        "config_order",
        "strategy",
        "strategy_display",
        "strategy_order",
        "status",
        "reason",
        "num_target_traces",
        "num_eval_traces",
        "acf_r2_median",
        "acf_r2_min",
        "acf_r2_max",
        "acf_r2_err_lo",
        "acf_r2_err_hi",
        "abs_delta_energy_pct_median",
        "abs_delta_energy_pct_min",
        "abs_delta_energy_pct_max",
        "abs_delta_energy_pct_err_lo",
        "abs_delta_energy_pct_err_hi",
    ]
    for row in summary_rows:
        for f in summary_fields:
            row.setdefault(f, "")
    _write_csv(out_summary_csv, summary_rows, summary_fields)

    plot_meta = _plot_figure_c1(
        summary_rows=summary_rows,
        config_ids_ordered=config_ids,
        config_labels=config_labels,
        out_figure_pdf=out_figure_pdf,
    )

    successful_configs = set(
        str(r["config_id"])
        for r in summary_rows
        if str(r.get("status", "")) == "evaluated"
    )
    failed_configs = set(str(x["config_id"]) for x in failures if str(x.get("scope", "")) == "config")

    manifest_payload = {
        "schema_version": "appendix-c1-decomposition-fidelity-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "run_manifest": str(run_manifest),
            "experimental_manifest": str(experimental_manifest),
            "throughput_db": str(throughput_db),
            "pair_manifest_csv": str(pair_manifest_csv),
            "ar1_params_dir": str(ar1_params_dir),
            "config_ids": [str(x) for x in config_ids],
            "moe_config_id": str(moe_config_id),
            "num_traces_per_config": int(num_traces_per_config),
            "base_seed": int(base_seed),
            "acf_max_lag": int(acf_max_lag),
            "prune_top_energy_per_group": int(prune_top_energy_per_group),
            "min_traces_after_prune": int(min_traces_after_prune),
            "device": str(resolved_device),
        },
        "resolved": {
            "config_ids_ordered": [str(x) for x in config_ids],
            "config_labels": {str(k): str(v) for k, v in config_labels.items()},
            "trace_selection_policy": "first_n_test_indices",
            "feature_build_policy": "prefer_stage0_json_rebuild_then_fallback_to_dataset_x_norm",
            "strategies": {
                "oracle": "measured posterior-max state sequence + within-state sampling",
                "predicted": "BiGRU-predicted state sequence + within-state sampling",
                "marginal": "iid state sampling from GMM weights + iid within-state sampling",
            },
            "moe_policy": {
                "moe_config_id": str(moe_config_id),
                "oracle_predicted_sampling": "ar1_thresholded",
                "marginal_sampling": "iid_gmm",
            },
        },
        "summary": {
            "num_requested_configs": int(len(config_ids)),
            "num_evaluated_configs": int(len(successful_configs)),
            "num_failed_configs": int(len(failed_configs)),
            "num_per_trace_rows": int(len(per_trace_rows)),
            "num_summary_rows": int(len(summary_rows)),
            "num_summary_evaluated_rows": int(
                sum(1 for r in summary_rows if str(r.get("status", "")) == "evaluated")
            ),
            "per_trace_ok_rows": int(
                sum(1 for r in per_trace_rows if str(r.get("status", "")) == "ok")
            ),
            "per_trace_pruned_rows": int(
                sum(1 for r in per_trace_rows if str(r.get("status", "")) == "pruned")
            ),
            "num_pruned_rows": int(num_pruned_rows),
            "expected_ok_rows_if_full": int(len(config_ids) * len(STRATEGY_ORDER) * int(num_traces_per_config)),
        },
        "plot": plot_meta,
        "artifacts": {
            "per_trace_csv": str(out_per_trace_csv),
            "summary_csv": str(out_summary_csv),
            "figure_pdf": str(out_figure_pdf),
            "manifest_json": str(out_manifest_json),
        },
        "per_config_strategy_summary": summary_rows,
        "failures": failures,
    }

    _write_json(out_manifest_json, manifest_payload)
    return manifest_payload


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Appendix C1 decomposition fidelity pipeline (Oracle vs Predicted vs Marginal)."
    )
    parser.add_argument("--run-manifest", default=defaults["run_manifest"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--throughput-db", default=defaults["throughput_db"])
    parser.add_argument("--pair-manifest-csv", default=defaults["pair_manifest_csv"])
    parser.add_argument("--ar1-params-dir", default=defaults["ar1_params_dir"])
    parser.add_argument(
        "--config-ids",
        nargs="*",
        default=list(DEFAULT_CONFIG_IDS),
        help="Config IDs to evaluate (space- or comma-separated).",
    )
    parser.add_argument("--moe-config-id", default=DEFAULT_MOE_CONFIG_ID)
    parser.add_argument("--num-traces-per-config", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--acf-max-lag", type=int, default=50)
    parser.add_argument(
        "--prune-top-energy-per-group",
        type=int,
        default=1,
        help="Prune top-K highest |ΔEnergy| traces per (config,strategy) before summary.",
    )
    parser.add_argument(
        "--min-traces-after-prune",
        type=int,
        default=3,
        help="Minimum retained traces per (config,strategy) after pruning.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-per-trace-csv", default=defaults["out_per_trace_csv"])
    parser.add_argument("--out-summary-csv", default=defaults["out_summary_csv"])
    parser.add_argument("--out-figure-pdf", default=defaults["out_figure_pdf"])
    parser.add_argument("--out-manifest-json", default=defaults["out_manifest_json"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_ids = _parse_config_ids(args.config_ids, DEFAULT_CONFIG_IDS)
    if len(config_ids) == 0:
        raise ValueError("No config_ids provided")

    run = run_appendix_decomposition_fidelity(
        run_manifest=args.run_manifest,
        experimental_manifest=args.experimental_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        ar1_params_dir=args.ar1_params_dir,
        config_ids=config_ids,
        moe_config_id=str(args.moe_config_id),
        num_traces_per_config=int(args.num_traces_per_config),
        base_seed=int(args.base_seed),
        acf_max_lag=int(args.acf_max_lag),
        prune_top_energy_per_group=int(args.prune_top_energy_per_group),
        min_traces_after_prune=int(args.min_traces_after_prune),
        out_per_trace_csv=str(args.out_per_trace_csv),
        out_summary_csv=str(args.out_summary_csv),
        out_figure_pdf=str(args.out_figure_pdf),
        out_manifest_json=str(args.out_manifest_json),
        device=str(args.device),
    )

    print("[appendix_decomposition_fidelity] Done")
    print(f"  requested_configs : {run['summary']['num_requested_configs']}")
    print(f"  evaluated_configs : {run['summary']['num_evaluated_configs']}")
    print(f"  failed_configs    : {run['summary']['num_failed_configs']}")
    print(f"  per_trace_rows    : {run['summary']['num_per_trace_rows']}")
    print(f"  pruned_rows       : {run['summary']['num_pruned_rows']}")
    print(f"  summary_rows      : {run['summary']['num_summary_rows']}")
    print(f"  per_trace_csv     : {run['artifacts']['per_trace_csv']}")
    print(f"  summary_csv       : {run['artifacts']['summary_csv']}")
    print(f"  figure_pdf        : {run['artifacts']['figure_pdf']}")
    print(f"  manifest_json     : {run['artifacts']['manifest_json']}")


if __name__ == "__main__":
    main()
