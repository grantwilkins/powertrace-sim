#!/usr/bin/env python3
"""
Generate side-by-side CDF plots comparing held-out measured vs sampled power traces.

This script reproduces/updates publication-style CDF figures like:
  - Original (measured) vs Sampled (synthetic)
  - One panel per config_id (e.g., A100 TP=1 and H100 TP=1)

Sampling is driven by trained GMM+BiGRU artifacts and can run in:
  - iid mode (default, uses state-conditional GMM sampling)
  - ar1_thresholded mode (hybrid AR(1) per-state sampling)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.classifiers.continuous_gru import compute_inference_features  # noqa: E402
from model.classifiers.gru import GRUClassifier  # noqa: E402
from model.classifiers.stateless_mean_reverting import (  # noqa: E402
    compute_delta_active_requests,
    normalize_delta_active_requests,
)
from model.scripts.continuous_v1_eval import compute_power_metrics  # noqa: E402

AR1_MIN_RUN_LENGTH = 5
AR1_PHI_THRESHOLD = 0.3
DEFAULT_CONFIG_IDS = (
    "llama-3-8b_A100_tp1",
    "llama-3-8b_H100_tp1",
)

MODEL_NAME_MAP = {
    "llama-3": "Llama-3",
    "deepseek-r1-distill": "DeepSeek-R1-Distill",
    "gpt-oss": "gpt-oss",
}

CONFIG_MODEL_SIZE_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")
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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / np.clip(denom, a_min=EPS, a_max=None)


def _tensor_to_numpy(t: torch.Tensor, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Convert torch tensor -> numpy with fallback for environments where torch.numpy bridge is unavailable.
    """
    cpu_t = t.detach().cpu()
    try:
        return np.asarray(cpu_t.numpy(), dtype=dtype)
    except Exception:
        return np.asarray(cpu_t.tolist(), dtype=dtype)


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
    *,
    requests: Sequence[Mapping[str, float]],
    throughput: Mapping[str, float],
    norm: Mapping[str, float],
    T: Optional[int],
    dt: float,
    feature_set: str = "f2",
) -> Dict[str, np.ndarray]:
    feat = str(feature_set).strip().lower()
    if feat not in {"f2", "f3"}:
        raise ValueError(f"feature_set must be one of {{'f2','f3'}}; got {feature_set}")

    lambda_prefill = _extract_norm_value(throughput, "lambda_prefill", "prefill_rate_median_toks_per_s")
    lambda_decode = _extract_norm_value(throughput, "lambda_decode", "decode_rate_median_toks_per_s")
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
    logits: np.ndarray | torch.Tensor,
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

    if clamp_range is not None:
        lo, hi = float(clamp_range[0]), float(clamp_range[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            margin = 0.05 * (hi - lo)
            power = np.clip(power, lo - margin, hi + margin)

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

    weights = np.clip(weights, a_min=1e-12, a_max=None)
    weights = weights / np.sum(weights)

    x = y.reshape(-1, 1)
    log_norm = -0.5 * (
        np.log(2.0 * np.pi * variances).reshape(1, -1) + ((x - means.reshape(1, -1)) ** 2) / variances.reshape(1, -1)
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

    for k in range(int(K)):
        runs = state_run_residuals[k]
        if len(runs) == 0:
            phi[k] = 0.0
            sigma_innov[k] = sigma_marginal[k]
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
            phi[k] = float(np.clip(phi_raw, 0.0, 0.99))
        else:
            phi[k] = 0.0

        sigma_innov[k] = float(sigma_marginal[k] * np.sqrt(max(1e-12, 1.0 - (phi[k] ** 2))))
    return phi, sigma_innov, sigma_marginal


def generate_gmm_bigru_trace_ar1_thresholded(
    *,
    logits: np.ndarray | torch.Tensor,
    gmm_params: Dict[str, object],
    phi: np.ndarray,
    sigma_innov: np.ndarray,
    sigma_marginal: np.ndarray,
    p0: float,
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    phi_threshold: float = AR1_PHI_THRESHOLD,
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
    if k <= 0:
        raise ValueError("GMM means are empty")
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

    t = int(z.shape[0])
    power = np.zeros((t,), dtype=np.float64)
    p_prev = float(p0)
    for i in range(t):
        s = int(states[i])
        mu = float(means[s])
        p_t = float(mu + (phi_gen[s] * (p_prev - mu)) + float(rng.normal(0.0, sigma_gen[s])))
        if clamp_range is not None:
            lo, hi = float(clamp_range[0]), float(clamp_range[1])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                margin = 0.05 * (hi - lo)
                p_t = float(np.clip(p_t, lo - margin, hi + margin))
        power[i] = p_t
        p_prev = p_t

    return {
        "power_w": power.astype(np.float64),
        "states": states.astype(np.int64),
        "states_raw": states_raw.astype(np.int64),
        "probs": probs.astype(np.float64),
        "use_ar1": use_ar1.astype(bool),
    }


def _ensure_dir_for_file(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_dir_for_file(path)
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


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _parse_config_ids(config_ids: Optional[Sequence[str]]) -> List[str]:
    if not config_ids:
        return []
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


def _finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _synthesize_request_timestamps(payload: Dict[str, object], n: int) -> Optional[List[float]]:
    if n <= 0:
        return []

    duration = _finite_float(payload.get("duration"))
    if duration is not None and duration > 0:
        step = float(duration) / float(max(n, 1))
        if step > 0:
            values = (np.arange(n, dtype=np.float64) + 0.5) * step + 1.0
            return [float(x) for x in values]

    request_rate = _finite_float(payload.get("request_rate"))
    poisson_rate = _finite_float(payload.get("poisson_rate"))
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
        float(np.min(arrivals)) < -float(dt) or float(np.max(arrivals)) > float(trace_duration_s) + float(dt)
    ):
        arrivals = arrivals - float(np.min(arrivals))

    requests: List[Dict[str, float]] = []
    for i in range(n):
        a = float(arrivals[i])
        nin = float(input_lens[i])
        nout = float(output_lens[i])
        if not (np.isfinite(a) and np.isfinite(nin) and np.isfinite(nout)):
            continue
        requests.append(
            {
                "arrival_time": float(a),
                "input_tokens": float(max(0.0, nin)),
                "output_tokens": float(max(0.0, nout)),
            }
        )
    if len(requests) == 0:
        raise ValueError("no valid requests after filtering")
    return requests


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
            json_path = _resolve_existing_path(json_path_raw, base_dir)
            if json_path is not None:
                out[key] = json_path
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


def _resolve_throughput(throughput_db: Dict[str, object], config_id: str) -> Dict[str, float]:
    cfgs = throughput_db.get("configs", {})
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


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    ys = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if xs.size == 0 or ys.size == 0:
        return float("nan")
    values = np.concatenate([xs, ys])
    values.sort()
    cdf_x = np.searchsorted(xs, values, side="right") / float(xs.size)
    cdf_y = np.searchsorted(ys, values, side="right") / float(ys.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = int(x.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    y = np.arange(1, n + 1, dtype=np.float64) / float(n)
    return x, y


def _display_label(config_id: str) -> str:
    m = CONFIG_MODEL_SIZE_RE.match(str(config_id).strip())
    if m is None:
        return str(config_id)
    model_family, model_size, hardware, tp = m.groups()
    model_name = MODEL_NAME_MAP.get(model_family, model_family)
    return f"{model_name}-{int(model_size)}B {hardware} TP={int(tp)}"


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "run_manifest": str(repo_root / "results" / "continuous_v1_gmm_bigru" / "k10_f2" / "run_manifest.json"),
        "experimental_manifest": str(repo_root / "results" / "experimental_continuous_v1" / "manifest.json"),
        "throughput_db": str(repo_root / "model" / "config" / "throughput_database.json"),
        "pair_manifest_csv": str(repo_root / "results" / "stage0" / "pair_manifest.csv"),
        "out_plot_dir": str(repo_root / "figures" / "trace_power_cdf_comparison"),
        "out_cdf_csv": str(repo_root / "results" / "eval_paper" / "trace_power_cdf_comparison_points.csv"),
        "out_summary_csv": str(repo_root / "results" / "eval_paper" / "trace_power_cdf_comparison.csv"),
        "out_json": str(repo_root / "results" / "eval_paper" / "trace_power_cdf_comparison.json"),
    }


def _collect_config_cdf(
    *,
    config_id: str,
    run_cfg_row: Dict[str, object],
    run_manifest_base: str,
    experimental_payload: Dict[str, object],
    experimental_base: str,
    throughput_payload: Dict[str, object],
    pair_map: Mapping[str, str],
    seeds: Sequence[int],
    generation_mode: str,
    decode_mode: str,
    median_filter_window: int,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(run_cfg_row, run_manifest_base)
    norm_payload = _load_json(norm_path)
    norm_cfg = _extract_norm_for_eval(norm_payload)
    gmm_payload = _load_json(gmm_path)
    gmm_cfg = load_gmm_params_json_dict(gmm_payload)
    throughput = _resolve_throughput(throughput_payload, config_id)

    k = int(run_cfg_row.get("k", gmm_cfg["k"]))
    feature_set = str(run_cfg_row.get("feature_set", norm_payload.get("feature_set", "f2"))).lower()
    if feature_set not in {"f2", "f3"}:
        raise ValueError(f"invalid feature_set for '{config_id}': {feature_set}")
    input_dim = int(run_cfg_row.get("input_dim", 2 if feature_set == "f2" else 3))
    hidden_dim = int(run_cfg_row.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
    num_layers = int(run_cfg_row.get("num_layers", norm_payload.get("num_layers", 1)))
    if k != int(gmm_cfg["k"]):
        raise ValueError(f"k mismatch between run manifest ({k}) and gmm payload ({int(gmm_cfg['k'])})")

    model = _load_model(
        checkpoint_path=checkpoint_path,
        k=k,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )

    dataset_path, split_path = _resolve_experimental_paths(
        experimental_payload,
        config_id=config_id,
        experimental_base=experimental_base,
    )
    split_payload = _load_json(split_path)
    test_indices = [int(x) for x in split_payload.get("test_indices", [])]
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]
    if len(test_indices) == 0:
        raise ValueError(f"empty test split for {config_id}")

    with np.load(dataset_path, allow_pickle=True) as data:
        pair_key_arr = np.asarray(data["pair_key"], dtype=object)
        power_arr = np.asarray(data["power"], dtype=object)
        power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
        dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)

    if dt_arr.size == 0:
        raise ValueError("dataset dt missing")
    dt = float(dt_arr[0])
    if (not np.isfinite(dt)) or dt <= 0.0:
        raise ValueError(f"invalid dt in dataset: {dt}")

    n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))

    phi = np.zeros((int(k),), dtype=np.float64)
    sigma_innov = np.zeros((int(k),), dtype=np.float64)
    sigma_marginal = np.zeros((int(k),), dtype=np.float64)
    if generation_mode == "ar1_thresholded":
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
            training_labels_traces.append(labels_train.astype(np.int64))
        if len(training_power_traces) == 0:
            raise ValueError(f"no valid training traces for AR1 estimation: {config_id}")
        phi, sigma_innov, sigma_marginal = estimate_ar1_params(
            gmm_params=gmm_cfg,
            training_power_traces=training_power_traces,
            training_labels_traces=training_labels_traces,
            K=int(k),
            min_run_length=AR1_MIN_RUN_LENGTH,
        )

    original_chunks: List[np.ndarray] = []
    sampled_chunks: List[np.ndarray] = []
    chosen_seed_rows: List[Dict[str, object]] = []
    skipped = 0

    for trace_idx in test_indices:
        if trace_idx < 0 or trace_idx >= n_total:
            skipped += 1
            continue
        power = np.asarray(power_arr[trace_idx], dtype=np.float64).reshape(-1)
        if power.size < 2:
            skipped += 1
            continue
        pair_key = str(pair_key_arr[trace_idx])
        json_path = pair_map.get(pair_key)
        if json_path is None:
            skipped += 1
            continue

        try:
            p0 = float(power[0])
            gt = power[1:].astype(np.float64)
            requests = _build_requests_from_stage0_json(
                json_path,
                power_start_epoch_s=float(power_start_arr[trace_idx]),
                trace_duration_s=float(power.size * dt),
                dt=float(dt),
            )
            feat = build_rollout_features_from_requests(
                requests=requests,
                throughput=throughput,
                norm=norm_cfg,
                T=int(gt.size),
                dt=float(dt),
                feature_set=feature_set,
            )
            features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
            if features_norm.ndim != 2 or features_norm.shape[1] != input_dim:
                skipped += 1
                continue
            with torch.no_grad():
                x = torch.tensor(features_norm.tolist(), dtype=torch.float32, device=device).unsqueeze(0)
                logits = _tensor_to_numpy(model(x)[0], dtype=np.float64)

            per_seed_rows: List[Dict[str, object]] = []
            preds_by_seed: Dict[int, np.ndarray] = {}
            for seed in seeds:
                if generation_mode == "iid":
                    gen = generate_gmm_bigru_trace(
                        logits=logits,
                        gmm_params=gmm_cfg,
                        seed=int(seed),
                        decode_mode=decode_mode,
                        median_filter_window=int(median_filter_window),
                        clamp_range=(norm_cfg["power_min"], norm_cfg["power_max"]),
                    )
                else:
                    gen = generate_gmm_bigru_trace_ar1_thresholded(
                        logits=logits,
                        gmm_params=gmm_cfg,
                        phi=phi,
                        sigma_innov=sigma_innov,
                        sigma_marginal=sigma_marginal,
                        p0=float(p0),
                        seed=int(seed),
                        decode_mode=decode_mode,
                        median_filter_window=int(median_filter_window),
                        phi_threshold=float(AR1_PHI_THRESHOLD),
                        clamp_range=(norm_cfg["power_min"], norm_cfg["power_max"]),
                    )
                pred = np.asarray(gen["power_w"], dtype=np.float64).reshape(-1)
                n = int(min(gt.size, pred.size))
                if n <= 0:
                    continue
                gt_n = gt[:n]
                pred_n = pred[:n]
                metrics = compute_power_metrics(gt_n, pred_n, dt=dt, acf_max_lag=50)
                row = {
                    "seed": int(seed),
                    "n": int(n),
                    "nrmse": float(metrics["nrmse"]),
                }
                per_seed_rows.append(row)
                preds_by_seed[int(seed)] = pred_n

            if len(per_seed_rows) == 0:
                skipped += 1
                continue

            nrmse_arr = np.asarray([float(r["nrmse"]) for r in per_seed_rows], dtype=np.float64)
            median_nrmse = float(np.median(nrmse_arr))
            best_idx = int(np.argmin(np.abs(nrmse_arr - median_nrmse)))
            chosen_seed = int(per_seed_rows[best_idx]["seed"])
            pred_sel = np.asarray(preds_by_seed[chosen_seed], dtype=np.float64).reshape(-1)
            n_sel = int(min(gt.size, pred_sel.size))
            if n_sel <= 0:
                skipped += 1
                continue

            original_chunks.append(gt[:n_sel].astype(np.float64))
            sampled_chunks.append(pred_sel[:n_sel].astype(np.float64))
            chosen_seed_rows.append(
                {
                    "config_id": config_id,
                    "trace_idx": int(trace_idx),
                    "pair_key": pair_key,
                    "chosen_seed": int(chosen_seed),
                    "chosen_nrmse": float(per_seed_rows[best_idx]["nrmse"]),
                    "median_nrmse": float(median_nrmse),
                    "num_seed_candidates": int(len(per_seed_rows)),
                    "num_points": int(n_sel),
                }
            )
        except Exception:
            skipped += 1
            continue

    if len(original_chunks) == 0 or len(sampled_chunks) == 0:
        raise ValueError(f"no valid traces evaluated for {config_id}")

    original_all = np.concatenate(original_chunks, axis=0).astype(np.float64)
    sampled_all = np.concatenate(sampled_chunks, axis=0).astype(np.float64)
    n_all = int(min(original_all.size, sampled_all.size))
    if n_all <= 0:
        raise ValueError(f"no aligned points after aggregation for {config_id}")
    original_all = original_all[:n_all]
    sampled_all = sampled_all[:n_all]

    metrics = compute_power_metrics(original_all, sampled_all, dt=dt, acf_max_lag=50)
    sorted_orig, cdf_orig = _ecdf(original_all)
    sorted_samp, cdf_samp = _ecdf(sampled_all)
    ks_stat = _ks_statistic(original_all, sampled_all)

    return {
        "config_id": config_id,
        "display_label": _display_label(config_id),
        "dt": float(dt),
        "num_test_traces": int(len(test_indices)),
        "num_eval_traces": int(len(chosen_seed_rows)),
        "num_skipped_traces": int(skipped),
        "num_points": int(n_all),
        "generation_mode": str(generation_mode),
        "decode_mode": str(decode_mode),
        "median_filter_window": int(median_filter_window),
        "summary": {
            "ks_stat": float(ks_stat),
            "nrmse": float(metrics["nrmse"]),
            "acf_r2": float(metrics["acf_r2"]),
            "p95_error_pct": float(metrics["p95_error_pct"]),
            "p99_error_pct": float(metrics["p99_error_pct"]),
            "delta_energy_pct": float(metrics["delta_energy_pct"]),
            "original_mean_w": float(np.mean(original_all)),
            "sampled_mean_w": float(np.mean(sampled_all)),
        },
        "original_sorted": sorted_orig,
        "original_cdf": cdf_orig,
        "sampled_sorted": sorted_samp,
        "sampled_cdf": cdf_samp,
        "chosen_seed_rows": chosen_seed_rows,
    }


def _plot_cdfs(
    *,
    results: Sequence[Dict[str, object]],
    out_plot_dir: str,
) -> List[Dict[str, str]]:
    if int(len(results)) <= 0:
        raise ValueError("No results to plot")

    Path(out_plot_dir).mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plot_files: List[Dict[str, str]] = []
    for row in results:
        fig, ax = plt.subplots(figsize=(6, 6))
        x_o = np.asarray(row["original_sorted"], dtype=np.float64)
        y_o = np.asarray(row["original_cdf"], dtype=np.float64)
        x_s = np.asarray(row["sampled_sorted"], dtype=np.float64)
        y_s = np.asarray(row["sampled_cdf"], dtype=np.float64)

        ax.plot(x_o, y_o, label="Original", color="#1f77b4", linewidth=2.2)
        ax.plot(x_s, y_s, label="Sampled", color="#d99000", linewidth=2.2)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Active GPU Power (W)")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best")

        slug = _safe_slug(str(row["config_id"]))
        out_pdf = str(Path(out_plot_dir) / f"{slug}_power_cdf.pdf")
        out_png = str(Path(out_plot_dir) / f"{slug}_power_cdf.png")
        fig.tight_layout()
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(
            {
                "config_id": str(row["config_id"]),
                "plot_pdf": out_pdf,
                "plot_png": out_png,
            }
        )
    return plot_files


def generate_power_cdf_comparison(
    *,
    run_manifest: str,
    experimental_manifest: str,
    throughput_db: str,
    pair_manifest_csv: str,
    config_ids: Sequence[str],
    generation_mode: str,
    num_seeds: int,
    base_seed: int,
    decode_mode: str,
    median_filter_window: int,
    device: str,
    out_plot_dir: str,
    out_cdf_csv: str,
    out_summary_csv: str,
    out_json: str,
) -> Dict[str, object]:
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    if generation_mode not in {"iid", "ar1_thresholded"}:
        raise ValueError("generation_mode must be one of {'iid', 'ar1_thresholded'}")
    if decode_mode not in {"stochastic", "argmax"}:
        raise ValueError("decode_mode must be one of {'stochastic','argmax'}")

    run_payload = _load_json(run_manifest)
    run_cfgs = run_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    run_manifest_base = str(Path(run_manifest).resolve().parent)

    experimental_payload = _load_json(experimental_manifest)
    experimental_base = str(Path(experimental_manifest).resolve().parent)
    throughput_payload = _load_json(throughput_db)
    pair_map = _load_pair_manifest_map(pair_manifest_csv)

    seeds = [int(base_seed) + i for i in range(int(num_seeds))]
    resolved_device = _resolve_device(device)

    results: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    for config_id in config_ids:
        row = run_cfgs.get(config_id)
        if not isinstance(row, dict):
            failures.append({"config_id": config_id, "reason": "config_not_in_run_manifest"})
            continue
        if str(row.get("status", "")) != "trained":
            failures.append({"config_id": config_id, "reason": f"config_status_{row.get('status', 'unknown')}"})
            continue
        try:
            cfg_result = _collect_config_cdf(
                config_id=config_id,
                run_cfg_row=row,
                run_manifest_base=run_manifest_base,
                experimental_payload=experimental_payload,
                experimental_base=experimental_base,
                throughput_payload=throughput_payload,
                pair_map=pair_map,
                seeds=seeds,
                generation_mode=generation_mode,
                decode_mode=decode_mode,
                median_filter_window=int(median_filter_window),
                device=resolved_device,
            )
            results.append(cfg_result)
        except Exception as exc:
            failures.append({"config_id": config_id, "reason": f"{type(exc).__name__}:{exc}"})

    if len(results) == 0:
        raise ValueError(f"No configs successfully evaluated. Failures: {failures}")

    plot_files = _plot_cdfs(
        results=results,
        out_plot_dir=out_plot_dir,
    )

    cdf_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    chosen_seed_rows: List[Dict[str, object]] = []

    for row in results:
        config_id = str(row["config_id"])
        for x, y in zip(np.asarray(row["original_sorted"]), np.asarray(row["original_cdf"])):
            cdf_rows.append(
                {
                    "config_id": config_id,
                    "display_label": str(row["display_label"]),
                    "series": "original",
                    "power_w": float(x),
                    "cdf": float(y),
                }
            )
        for x, y in zip(np.asarray(row["sampled_sorted"]), np.asarray(row["sampled_cdf"])):
            cdf_rows.append(
                {
                    "config_id": config_id,
                    "display_label": str(row["display_label"]),
                    "series": "sampled",
                    "power_w": float(x),
                    "cdf": float(y),
                }
            )

        s = dict(row["summary"])
        summary_rows.append(
            {
                "config_id": config_id,
                "display_label": str(row["display_label"]),
                "generation_mode": str(row["generation_mode"]),
                "decode_mode": str(row["decode_mode"]),
                "median_filter_window": int(row["median_filter_window"]),
                "dt": float(row["dt"]),
                "num_test_traces": int(row["num_test_traces"]),
                "num_eval_traces": int(row["num_eval_traces"]),
                "num_skipped_traces": int(row["num_skipped_traces"]),
                "num_points": int(row["num_points"]),
                "ks_stat": float(s["ks_stat"]),
                "nrmse": float(s["nrmse"]),
                "acf_r2": float(s["acf_r2"]),
                "p95_error_pct": float(s["p95_error_pct"]),
                "p99_error_pct": float(s["p99_error_pct"]),
                "delta_energy_pct": float(s["delta_energy_pct"]),
                "original_mean_w": float(s["original_mean_w"]),
                "sampled_mean_w": float(s["sampled_mean_w"]),
            }
        )
        chosen_seed_rows.extend([dict(x) for x in row["chosen_seed_rows"]])

    _write_csv(
        out_cdf_csv,
        cdf_rows,
        fieldnames=["config_id", "display_label", "series", "power_w", "cdf"],
    )
    _write_csv(
        out_summary_csv,
        summary_rows,
        fieldnames=[
            "config_id",
            "display_label",
            "generation_mode",
            "decode_mode",
            "median_filter_window",
            "dt",
            "num_test_traces",
            "num_eval_traces",
            "num_skipped_traces",
            "num_points",
            "ks_stat",
            "nrmse",
            "acf_r2",
            "p95_error_pct",
            "p99_error_pct",
            "delta_energy_pct",
            "original_mean_w",
            "sampled_mean_w",
        ],
    )

    payload = {
        "status": "ok",
        "inputs": {
            "run_manifest": str(run_manifest),
            "experimental_manifest": str(experimental_manifest),
            "throughput_db": str(throughput_db),
            "pair_manifest_csv": str(pair_manifest_csv),
            "config_ids": [str(x) for x in config_ids],
            "generation_mode": str(generation_mode),
            "num_seeds": int(num_seeds),
            "base_seed": int(base_seed),
            "decode_mode": str(decode_mode),
            "median_filter_window": int(median_filter_window),
            "device": str(resolved_device),
        },
        "artifacts": {
            "plot_dir": str(out_plot_dir),
            "plot_files": plot_files,
            "cdf_points_csv": str(out_cdf_csv),
            "summary_csv": str(out_summary_csv),
        },
        "summary": {
            "num_requested_configs": int(len(config_ids)),
            "num_successful_configs": int(len(results)),
            "num_failed_configs": int(len(failures)),
        },
        "config_summaries": summary_rows,
        "chosen_seed_rows": chosen_seed_rows,
        "failures": failures,
    }
    _write_json(out_json, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Generate publication-style CDF comparison plots for measured vs sampled held-out power traces."
    )
    parser.add_argument("--run-manifest", default=defaults["run_manifest"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--throughput-db", default=defaults["throughput_db"])
    parser.add_argument("--pair-manifest-csv", default=defaults["pair_manifest_csv"])
    parser.add_argument(
        "--config-ids",
        nargs="*",
        default=list(DEFAULT_CONFIG_IDS),
        help="Config IDs to plot (space- or comma-separated).",
    )
    parser.add_argument("--generation-mode", choices=["iid", "ar1_thresholded"], default="iid")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--decode-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-plot-dir", default=defaults["out_plot_dir"])
    parser.add_argument("--out-cdf-csv", default=defaults["out_cdf_csv"])
    parser.add_argument("--out-summary-csv", default=defaults["out_summary_csv"])
    parser.add_argument("--out-json", default=defaults["out_json"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_ids = _parse_config_ids(args.config_ids)
    if len(config_ids) == 0:
        raise ValueError("No config_ids provided")

    run = generate_power_cdf_comparison(
        run_manifest=args.run_manifest,
        experimental_manifest=args.experimental_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        config_ids=config_ids,
        generation_mode=args.generation_mode,
        num_seeds=int(args.num_seeds),
        base_seed=int(args.base_seed),
        decode_mode=args.decode_mode,
        median_filter_window=int(args.median_filter_window),
        device=args.device,
        out_plot_dir=args.out_plot_dir,
        out_cdf_csv=args.out_cdf_csv,
        out_summary_csv=args.out_summary_csv,
        out_json=args.out_json,
    )

    print("[generate_power_cdf_comparison] Done")
    print(f"  requested_configs: {run['summary']['num_requested_configs']}")
    print(f"  successful_configs: {run['summary']['num_successful_configs']}")
    print(f"  failed_configs: {run['summary']['num_failed_configs']}")
    print(f"  plot_dir: {run['artifacts']['plot_dir']}")
    print(f"  summary_csv: {run['artifacts']['summary_csv']}")
    print(f"  cdf_points_csv: {run['artifacts']['cdf_points_csv']}")


if __name__ == "__main__":
    main()
