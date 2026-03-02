#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from model.classifiers.continuous_gru import compute_inference_features
from model.classifiers.stateless_mean_reverting import (
    compute_delta_active_requests,
    normalize_delta_active_requests,
)

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

    A_norm = np.asarray(base[:, 0], dtype=np.float32)
    t_arrive_norm = np.asarray(base[:, 1], dtype=np.float32)
    A_raw = (A_norm.astype(np.float64) * float(active_std)) + float(active_mean)
    delta_raw = compute_delta_active_requests(A_raw).astype(np.float64)
    delta_norm = normalize_delta_active_requests(delta_raw, mean=dA_mean, std=dA_std).astype(np.float32)

    cols = [A_norm, delta_norm]
    if feat == "f3":
        cols.append(t_arrive_norm)
    features = np.stack(cols, axis=-1).astype(np.float32)

    return {
        "features_norm": features,
        "A_raw": A_raw.astype(np.float64),
        "A_norm": A_norm.astype(np.float32),
        "delta_A_raw": delta_raw.astype(np.float64),
        "delta_A_norm": delta_norm.astype(np.float32),
        "t_arrive_norm": t_arrive_norm.astype(np.float32),
    }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / np.clip(denom, a_min=EPS, a_max=None)


def _median_filter_states(states: np.ndarray, window: int) -> np.ndarray:
    z = np.asarray(states, dtype=np.int64).reshape(-1)
    n = int(z.size)
    if n == 0:
        return z
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


def generate_gmm_bigru_trace(
    logits: Union[np.ndarray, torch.Tensor],
    gmm_params: Mapping[str, object],
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    if isinstance(logits, torch.Tensor):
        try:
            z = logits.detach().cpu().numpy()
        except Exception:
            z = np.asarray(logits.detach().cpu().tolist(), dtype=np.float64)
    else:
        z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 2:
        raise ValueError(f"logits must have shape (T,K) or (1,T,K); got {z.shape}")

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
    K = int(means.size)
    if z.shape[1] != K:
        raise ValueError(f"logits K mismatch: got {z.shape[1]} but GMM has {K}")

    probs = _softmax_np(z)
    mode = str(decode_mode).strip().lower()
    if mode not in {"stochastic", "argmax"}:
        raise ValueError(f"decode_mode must be 'stochastic' or 'argmax'; got {decode_mode}")

    rng = np.random.default_rng(seed)
    if mode == "argmax":
        states_raw = np.argmax(probs, axis=-1).astype(np.int64)
    else:
        states_raw = np.array([rng.choice(K, p=probs_t) for probs_t in probs], dtype=np.int64)

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
    min_run_length: int = 5,
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
    logits: Union[np.ndarray, torch.Tensor],
    gmm_params: Dict[str, object],
    phi: np.ndarray,
    sigma_innov: np.ndarray,
    sigma_marginal: np.ndarray,
    p0: float,
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    phi_threshold: float = 0.3,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    if isinstance(logits, torch.Tensor):
        try:
            z = logits.detach().cpu().numpy()
        except Exception:
            z = np.asarray(logits.detach().cpu().tolist(), dtype=np.float64)
    else:
        z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 2:
        raise ValueError(f"logits must have shape (T,K) or (1,T,K); got {z.shape}")

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    K = int(means.size)
    if K <= 0:
        raise ValueError("GMM means are empty")
    if z.shape[1] != K:
        raise ValueError(f"logits K mismatch: got {z.shape[1]} but GMM has {K}")

    phi_arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    sigma_innov_arr = np.asarray(sigma_innov, dtype=np.float64).reshape(-1)
    sigma_marginal_arr = np.asarray(sigma_marginal, dtype=np.float64).reshape(-1)
    if phi_arr.size != K or sigma_innov_arr.size != K or sigma_marginal_arr.size != K:
        raise ValueError(f"phi/sigma arrays size mismatch for K={K}")

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
        states_raw = np.array([rng.choice(K, p=probs_t) for probs_t in probs], dtype=np.int64)
    states = _median_filter_states(states_raw, int(median_filter_window))

    T = int(z.shape[0])
    power = np.zeros((T,), dtype=np.float64)
    p_prev = float(p0)
    for t in range(T):
        s = int(states[t])
        mu = float(means[s])
        p_t = float(mu + (phi_gen[s] * (p_prev - mu)) + float(rng.normal(0.0, sigma_gen[s])))
        if clamp_range is not None:
            lo, hi = float(clamp_range[0]), float(clamp_range[1])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                margin = 0.05 * (hi - lo)
                p_t = float(np.clip(p_t, lo - margin, hi + margin))
        power[t] = p_t
        p_prev = p_t

    return {
        "power_w": power.astype(np.float64),
        "states": states.astype(np.int64),
        "states_raw": states_raw.astype(np.int64),
        "probs": probs.astype(np.float64),
        "use_ar1": use_ar1.astype(bool),
        "phi_gen": phi_gen.astype(np.float64),
        "sigma_gen": sigma_gen.astype(np.float64),
    }
