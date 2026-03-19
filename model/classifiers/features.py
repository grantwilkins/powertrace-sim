from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


def compute_delta_active_requests(active_requests: np.ndarray) -> np.ndarray:
    """Compute the first-difference of active request counts.

    Returns array of same length with delta[0]=0 and delta[t]=active[t]-active[t-1].
    """
    arr = np.asarray(active_requests, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    delta = np.zeros_like(arr, dtype=np.float64)
    if arr.size > 1:
        delta[1:] = arr[1:] - arr[:-1]
    return delta


def normalize_delta_active_requests(delta_active: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score normalize delta active request values."""
    arr = np.asarray(delta_active, dtype=np.float64).reshape(-1)
    denom = max(float(std), 1e-6)
    return ((arr - float(mean)) / denom).astype(np.float32)


def compute_inference_features(
    requests: Sequence[Dict[str, object]],
    config: Dict[str, float],
    T: Optional[int] = None,
    dt: float = 0.25,
) -> np.ndarray:
    """Build normalized inference features from a request schedule.

    Returns (T, 2) array with columns [A_norm, T_arrive_norm].
    """
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}")

    lambda_prefill = float(config["lambda_prefill"])
    lambda_decode = float(config["lambda_decode"])
    if (not np.isfinite(lambda_prefill)) or lambda_prefill <= 0.0:
        raise ValueError(f"lambda_prefill must be positive; got {lambda_prefill}")
    if (not np.isfinite(lambda_decode)) or lambda_decode <= 0.0:
        raise ValueError(f"lambda_decode must be positive; got {lambda_decode}")

    A_mean = float(config["A_mean"])
    A_std = max(float(config["A_std"]), EPS)
    t_arrive_mean = float(config["T_arrive_log_mean"])
    t_arrive_std = max(float(config["T_arrive_log_std"]), EPS)

    parsed: List[Tuple[float, float, float]] = []
    max_est_completion = 0.0
    for i, req in enumerate(requests):
        if not isinstance(req, dict):
            raise ValueError(f"request[{i}] must be a dict")
        try:
            arrival_time = float(req["arrival_time"])
            input_tokens = float(req["input_tokens"])
            output_tokens = float(req["output_tokens"])
        except Exception as exc:
            raise ValueError(f"request[{i}] missing/invalid required fields") from exc
        if not (np.isfinite(arrival_time) and np.isfinite(input_tokens) and np.isfinite(output_tokens)):
            raise ValueError(f"request[{i}] contains non-finite values")

        input_tokens = max(0.0, input_tokens)
        output_tokens = max(0.0, output_tokens)
        prefill_time = input_tokens / lambda_prefill
        decode_time = output_tokens / lambda_decode
        est_completion = arrival_time + prefill_time + decode_time
        parsed.append((arrival_time, input_tokens, est_completion))
        if est_completion > max_est_completion:
            max_est_completion = float(est_completion)

    if T is None:
        if len(parsed) == 0:
            raise ValueError("Empty request schedule requires explicit T.")
        T = int(max(0, math.ceil((max_est_completion + dt) / dt)))
    else:
        T = int(T)
        if T < 0:
            raise ValueError(f"T must be >= 0; got {T}")

    t_arrive = np.zeros((T,), dtype=np.float64)
    active_diff = np.zeros((T + 1,), dtype=np.float64)

    for arrival_time, input_tokens, est_completion in parsed:
        arrive_bin = int(math.floor(arrival_time / dt))
        if 0 <= arrive_bin < T:
            t_arrive[arrive_bin] += input_tokens

        start_idx = int(math.ceil(arrival_time / dt))
        end_idx = int(math.ceil(est_completion / dt))
        if end_idx <= 0 or start_idx >= T:
            continue
        left = max(0, start_idx)
        right = min(T, end_idx)
        if right <= left:
            continue
        active_diff[left] += 1.0
        active_diff[right] -= 1.0

    A = np.clip(np.cumsum(active_diff[:-1]), a_min=0.0, a_max=None)
    t_arrive_log = np.log1p(np.clip(t_arrive, a_min=0.0, a_max=None))

    A_norm = (A - A_mean) / A_std
    t_arrive_norm = (t_arrive_log - t_arrive_mean) / t_arrive_std
    return np.stack([A_norm, t_arrive_norm], axis=-1).astype(np.float32)


def _extract_norm_value(norm: Mapping[str, float], *keys: str) -> float:
    for key in keys:
        if key in norm:
            return float(norm[key])
    raise KeyError(f"Missing required norm key; expected one of {keys}")


def _safe_std(value: float) -> float:
    out = float(value)
    if (not np.isfinite(out)) or out <= 0.0:
        return 1e-6
    return max(out, 1e-6)


def extract_norm_params(norm_payload: Mapping[str, object]) -> Dict[str, float]:
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


def _validate_feature_set(feature_set: str) -> str:
    value = str(feature_set).strip().lower()
    if value == "f3":
        raise ValueError("feature_set='f3' is no longer supported; use 'f2'.")
    if value != "f2":
        raise ValueError(f"feature_set must be 'f2'; got {feature_set}")
    return value


def _as_1d_finite(values, *, allow_empty: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        if allow_empty:
            return arr
        raise ValueError("Expected non-empty array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Array contains non-finite values.")
    return arr


def build_features_from_active(
    active_requests,
    t_arrive_log,
    norm: Mapping[str, float],
    feature_set: str = "f2",
    max_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Build normalized training features from active request counts."""
    _validate_feature_set(feature_set)
    del t_arrive_log
    active = _as_1d_finite(active_requests, allow_empty=True)
    if active.size == 0:
        return {
            "features_norm": np.zeros((0, 2), dtype=np.float32),
            "A_raw": np.zeros((0,), dtype=np.float64),
            "A_norm": np.zeros((0,), dtype=np.float32),
            "delta_A_raw": np.zeros((0,), dtype=np.float64),
            "delta_A_norm": np.zeros((0,), dtype=np.float32),
            "t_arrive_norm": np.zeros((0,), dtype=np.float32),
        }

    L = int(max(0, active.size - 1))
    if max_length is not None:
        L = int(min(L, int(max(0, max_length))))

    if L <= 0:
        return {
            "features_norm": np.zeros((0, 2), dtype=np.float32),
            "A_raw": np.zeros((0,), dtype=np.float64),
            "A_norm": np.zeros((0,), dtype=np.float32),
            "delta_A_raw": np.zeros((0,), dtype=np.float64),
            "delta_A_norm": np.zeros((0,), dtype=np.float32),
            "t_arrive_norm": np.zeros((0,), dtype=np.float32),
        }

    A_raw = active[1 : L + 1]
    delta_raw = A_raw - active[:L]

    active_mean = _extract_norm_value(norm, "active_mean", "A_mean")
    active_std = _safe_std(_extract_norm_value(norm, "active_std", "A_std"))
    dA_mean = _extract_norm_value(norm, "delta_A_mean")
    dA_std = _safe_std(_extract_norm_value(norm, "delta_A_std"))

    A_norm = ((A_raw - active_mean) / active_std).astype(np.float32)
    delta_norm = normalize_delta_active_requests(
        delta_raw, mean=dA_mean, std=dA_std
    ).astype(np.float32)

    t_norm = np.zeros((L,), dtype=np.float32)
    features = np.stack([A_norm, delta_norm], axis=-1).astype(np.float32)
    return {
        "features_norm": features,
        "A_raw": A_raw.astype(np.float64),
        "A_norm": A_norm,
        "delta_A_raw": delta_raw.astype(np.float64),
        "delta_A_norm": delta_norm,
        "t_arrive_norm": t_norm,
    }


def build_rollout_features_from_requests(
    requests: Sequence[Dict[str, object]],
    throughput: Mapping[str, float],
    norm: Mapping[str, float],
    T: Optional[int] = None,
    dt: float = 0.25,
    feature_set: str = "f2",
) -> Dict[str, np.ndarray]:
    """Build rollout-time normalized features from request logs and throughput rates."""
    _validate_feature_set(feature_set)
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

    A_norm = np.asarray(base[:, 0], dtype=np.float32)
    t_arrive_norm = np.asarray(base[:, 1], dtype=np.float32)
    A_raw = (A_norm.astype(np.float64) * float(active_std)) + float(active_mean)
    delta_raw = compute_delta_active_requests(A_raw).astype(np.float64)
    delta_norm = normalize_delta_active_requests(
        delta_raw, mean=dA_mean, std=dA_std
    ).astype(np.float32)

    features = np.stack([A_norm, delta_norm], axis=-1).astype(np.float32)

    return {
        "features_norm": features,
        "A_raw": A_raw.astype(np.float64),
        "A_norm": A_norm.astype(np.float32),
        "delta_A_raw": delta_raw.astype(np.float64),
        "delta_A_norm": delta_norm.astype(np.float32),
        "t_arrive_norm": t_arrive_norm.astype(np.float32),
    }
