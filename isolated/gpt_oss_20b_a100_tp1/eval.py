#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.classifiers.gmm_bigru import build_rollout_features_from_requests
from model.classifiers.metrics import compute_power_metrics

DEFAULT_CONFIG_ID = "gpt-oss-20b_A100_tp1"


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / np.clip(denom, a_min=1e-12, a_max=None)


def _build_request_list(
    *,
    request_arrival_time_s: np.ndarray,
    input_tokens: np.ndarray,
    output_tokens: np.ndarray,
    power_start_epoch_s: float,
    dt: float,
    trace_duration_s: float,
) -> List[Dict[str, float]]:
    arrivals_abs = np.asarray(request_arrival_time_s, dtype=np.float64).reshape(-1)
    nin = np.asarray(input_tokens, dtype=np.float64).reshape(-1)
    nout = np.asarray(output_tokens, dtype=np.float64).reshape(-1)
    n = int(min(arrivals_abs.size, nin.size, nout.size))
    if n <= 0:
        raise ValueError("Heldout request cache entry is empty")

    arrivals = arrivals_abs[:n] - float(power_start_epoch_s)
    if arrivals.size > 0:
        min_arrival = float(np.min(arrivals))
        max_arrival = float(np.max(arrivals))
        if min_arrival < -float(dt) or max_arrival > float(trace_duration_s) + float(dt):
            arrivals = arrivals - min_arrival

    requests: List[Dict[str, float]] = []
    for idx in range(n):
        a = float(arrivals[idx])
        in_tok = float(max(0.0, nin[idx]))
        out_tok = float(max(0.0, nout[idx]))
        if not (np.isfinite(a) and np.isfinite(in_tok) and np.isfinite(out_tok)):
            continue
        requests.append(
            {
                "arrival_time": a,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        )
    if len(requests) == 0:
        raise ValueError("Heldout request cache contained no valid requests")
    return requests


def _expected_power_from_logits(logits: np.ndarray, gmm_means: np.ndarray) -> np.ndarray:
    # Keep evaluation deterministic: use the classifier posterior over states and
    # map it directly to the GMM state means instead of sampling a rollout.
    probs = _softmax_np(logits)
    return np.asarray(probs @ gmm_means.reshape(-1), dtype=np.float64).reshape(-1)


def _median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def evaluate_trained_model(
    *,
    model: torch.nn.Module,
    gmm_params: Mapping[str, object],
    norm_payload: Mapping[str, float],
    prepared_dir: str | Path,
    feature_set: str,
    input_dim: int,
    device: Optional[torch.device | str] = None,
    acf_max_lag: int = 50,
) -> Dict[str, float]:
    prepared_root = Path(prepared_dir).resolve()
    dataset_path = prepared_root / "dataset.npz"
    split_path = prepared_root / "split.json"
    throughput_path = prepared_root / "throughput_database.json"
    heldout_requests_path = prepared_root / "heldout_requests.npz"

    split_payload = _load_json(split_path)
    throughput_payload = _load_json(throughput_path)
    throughput_cfgs = throughput_payload.get("configs", {})
    if not isinstance(throughput_cfgs, dict):
        raise ValueError(f"Invalid throughput database format: {throughput_path}")
    throughput_row = throughput_cfgs.get(DEFAULT_CONFIG_ID)
    if not isinstance(throughput_row, dict):
        raise ValueError(f"Missing throughput config '{DEFAULT_CONFIG_ID}' in {throughput_path}")

    throughput = {
        "lambda_prefill": float(throughput_row["prefill_rate_median_toks_per_s"]),
        "lambda_decode": float(throughput_row["decode_rate_median_toks_per_s"]),
    }
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    with np.load(dataset_path, allow_pickle=True) as data:
        pair_key_arr = np.asarray(data["pair_key"], dtype=object)
        power_arr = np.asarray(data["power"], dtype=object)
        power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64).reshape(-1)
        dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)

    with np.load(heldout_requests_path, allow_pickle=True) as heldout_data:
        trace_idx_arr = np.asarray(heldout_data["trace_idx"], dtype=np.int64).reshape(-1)
        heldout_map = {
            int(trace_idx_arr[i]): {
                "pair_key": str(np.asarray(heldout_data["pair_key"], dtype=object).reshape(-1)[i]),
                "rate": str(np.asarray(heldout_data["rate"], dtype=object).reshape(-1)[i]),
                "request_arrival_time_s": np.asarray(
                    np.asarray(heldout_data["request_arrival_time_s"], dtype=object).reshape(-1)[i],
                    dtype=np.float64,
                ),
                "input_tokens": np.asarray(
                    np.asarray(heldout_data["input_tokens"], dtype=object).reshape(-1)[i],
                    dtype=np.float64,
                ),
                "output_tokens": np.asarray(
                    np.asarray(heldout_data["output_tokens"], dtype=object).reshape(-1)[i],
                    dtype=np.float64,
                ),
            }
            for i in range(trace_idx_arr.size)
        }

    if dt_arr.size == 0:
        raise ValueError("Prepared dataset is missing dt")
    dt = float(dt_arr[0])
    test_indices_raw = split_payload.get("test_indices", [])
    if not isinstance(test_indices_raw, list) or len(test_indices_raw) == 0:
        raise ValueError(f"Prepared split has no test indices: {split_path}")
    test_indices = [int(idx) for idx in test_indices_raw]

    gmm_means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    if gmm_means.size != int(gmm_params["k"]):
        raise ValueError("GMM means size does not match k")

    per_trace_metrics: List[Dict[str, float]] = []
    for trace_idx in test_indices:
        cached = heldout_map.get(int(trace_idx))
        if cached is None:
            raise KeyError(f"Missing heldout request cache for trace index {trace_idx}")
        if trace_idx < 0 or trace_idx >= len(power_arr):
            raise IndexError(f"Trace index out of bounds for dataset: {trace_idx}")

        power = np.asarray(power_arr[trace_idx], dtype=np.float64).reshape(-1)
        if power.size < 2:
            raise ValueError(f"Heldout trace {trace_idx} has length < 2")
        ground_truth = power[1:].astype(np.float64)
        T = int(ground_truth.size)
        if T <= 0:
            raise ValueError(f"Heldout trace {trace_idx} has no forecast horizon")

        requests = _build_request_list(
            request_arrival_time_s=cached["request_arrival_time_s"],
            input_tokens=cached["input_tokens"],
            output_tokens=cached["output_tokens"],
            power_start_epoch_s=float(power_start_arr[trace_idx]),
            dt=dt,
            trace_duration_s=float((T + 1) * dt),
        )
        features = build_rollout_features_from_requests(
            requests=requests,
            throughput=throughput,
            norm=norm_payload,
            T=T,
            dt=dt,
            feature_set=feature_set,
        )
        features_norm = np.asarray(features["features_norm"], dtype=np.float32)
        if features_norm.ndim != 2 or features_norm.shape[1] != int(input_dim):
            raise ValueError(
                f"Feature shape mismatch for trace {trace_idx}: {features_norm.shape} vs input_dim={input_dim}"
            )

        with torch.no_grad():
            x = torch.from_numpy(features_norm).to(device=resolved_device, dtype=torch.float32)
            logits = model(x.unsqueeze(0))[0].detach().cpu().numpy()

        predicted = _expected_power_from_logits(logits, gmm_means)
        metrics = compute_power_metrics(
            ground_truth,
            predicted,
            dt=dt,
            acf_max_lag=int(acf_max_lag),
        )
        per_trace_metrics.append(
            {
                "trace_idx": float(trace_idx),
                "acf_r2": float(metrics["acf_r2"]),
                "ks_stat": float(metrics["ks_stat"]),
                "delta_energy_pct": float(metrics["delta_energy_pct"]),
            }
        )

    return {
        "num_heldout_traces": float(len(per_trace_metrics)),
        "acf_r2": _median(row["acf_r2"] for row in per_trace_metrics),
        "ks_stat": _median(row["ks_stat"] for row in per_trace_metrics),
        "delta_energy_pct": _median(row["delta_energy_pct"] for row in per_trace_metrics),
    }
