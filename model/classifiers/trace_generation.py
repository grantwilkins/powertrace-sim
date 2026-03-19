from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

EPS = 1e-12
AR1_MIN_RUN_LENGTH = 5
AR1_PHI_THRESHOLD = 0.3


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


def _logits_to_numpy(logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(logits, torch.Tensor):
        try:
            z = np.asarray(logits.detach().cpu().numpy(), dtype=np.float64)
        except Exception:
            z = np.asarray(logits.detach().cpu().tolist(), dtype=np.float64)
    else:
        z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 2:
        raise ValueError(f"logits must have shape (T,K) or (1,T,K); got {z.shape}")
    return z


def generate_gmm_bigru_trace(
    logits: Union[np.ndarray, torch.Tensor],
    gmm_params: Mapping[str, object],
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Generate a power trace from classifier logits and state-conditional Gaussian params."""
    z = _logits_to_numpy(logits)

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
    K = int(means.size)
    if z.shape[1] != K:
        raise ValueError(f"logits K mismatch: got {z.shape[1]} but GMM has {K}")

    probs = _softmax_np(z)
    mode = str(decode_mode).strip().lower()
    if mode not in {"stochastic", "argmax"}:
        raise ValueError(
            f"decode_mode must be 'stochastic' or 'argmax'; got {decode_mode}"
        )

    rng = np.random.default_rng(seed)
    if mode == "argmax":
        states_raw = np.argmax(probs, axis=-1).astype(np.int64)
    else:
        states_raw = np.array(
            [rng.choice(K, p=probs_t) for probs_t in probs],
            dtype=np.int64,
        )

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


def estimate_ar1_params(
    gmm_params: Mapping[str, object],
    training_power_traces: Sequence[np.ndarray],
    training_labels_traces: Sequence[np.ndarray],
    K: int,
    min_run_length: int = AR1_MIN_RUN_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.clip(
        np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1),
        a_min=1e-12,
        a_max=None,
    )
    if means.size != int(K) or variances.size != int(K):
        raise ValueError(f"GMM parameter size mismatch for K={K}")

    state_run_residuals: Dict[int, list[np.ndarray]] = {k: [] for k in range(int(K))}
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

        sigma_innov[k] = float(
            sigma_marginal[k] * np.sqrt(max(1e-12, 1.0 - (phi[k] ** 2)))
        )
    return phi, sigma_innov, sigma_marginal


def generate_gmm_bigru_trace_ar1_thresholded(
    *,
    logits: np.ndarray | torch.Tensor,
    gmm_params: Mapping[str, object],
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
    z = _logits_to_numpy(logits)

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
    sigma_gen = np.where(use_ar1, sigma_innov_arr, sigma_marginal_arr).astype(
        np.float64
    )

    probs = _softmax_np(z)
    mode = str(decode_mode).strip().lower()
    if mode not in {"stochastic", "argmax"}:
        raise ValueError(
            f"decode_mode must be 'stochastic' or 'argmax'; got {decode_mode}"
        )

    rng = np.random.default_rng(seed)
    if mode == "argmax":
        states_raw = np.argmax(probs, axis=-1).astype(np.int64)
    else:
        states_raw = np.asarray(
            [rng.choice(k, p=probs_t) for probs_t in probs], dtype=np.int64
        )
    states = _median_filter_states(states_raw, int(median_filter_window))

    t = int(z.shape[0])
    power = np.zeros((t,), dtype=np.float64)
    p_prev = float(p0)
    for i in range(t):
        s = int(states[i])
        mu = float(means[s])
        p_t = float(
            mu + (phi_gen[s] * (p_prev - mu)) + float(rng.normal(0.0, sigma_gen[s]))
        )
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
        "phi_gen": phi_gen.astype(np.float64),
        "sigma_gen": sigma_gen.astype(np.float64),
    }
