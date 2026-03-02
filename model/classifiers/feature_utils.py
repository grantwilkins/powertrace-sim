"""
Feature computation utilities for power trace modeling.

This module contains functions extracted from legacy classifiers to provide
shared feature computation for GMM-BiGRU and evaluation pipelines.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-12


def compute_delta_active_requests(active_requests: np.ndarray) -> np.ndarray:
    """
    Compute the first-difference of active request counts.

    Args:
        active_requests: Array of active request counts per time step.

    Returns:
        Array of same length with delta[0]=0 and delta[t]=active[t]-active[t-1].
    """
    arr = np.asarray(active_requests, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    delta = np.zeros_like(arr, dtype=np.float64)
    if arr.size > 1:
        delta[1:] = arr[1:] - arr[:-1]
    return delta


def normalize_delta_active_requests(delta_active: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Z-score normalize delta active request values.

    Args:
        delta_active: Raw delta values.
        mean: Mean for normalization.
        std: Standard deviation for normalization.

    Returns:
        Normalized delta values as float32.
    """
    arr = np.asarray(delta_active, dtype=np.float64).reshape(-1)
    denom = max(float(std), 1e-6)
    return ((arr - float(mean)) / denom).astype(np.float32)


def compute_inference_features(
    requests: Sequence[Dict[str, object]],
    config: Dict[str, float],
    T: Optional[int] = None,
    dt: float = 0.25,
) -> np.ndarray:
    """
    Build normalized inference features from a request schedule.

    Args:
        requests: List of request dicts with keys:
            - arrival_time: Time when request arrives
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens
        config: Configuration dict with keys:
            - lambda_prefill: Prefill throughput (tokens/sec)
            - lambda_decode: Decode throughput (tokens/sec)
            - A_mean: Mean of active request count for normalization
            - A_std: Std of active request count for normalization
            - T_arrive_log_mean: Mean of log(1+arrival_tokens) for normalization
            - T_arrive_log_std: Std of log(1+arrival_tokens) for normalization
        T: Number of time steps. If None, computed from request schedule.
        dt: Time step duration in seconds.

    Returns:
        (T, 2) array with columns [A_norm, T_arrive_norm]
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
