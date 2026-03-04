#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.classifiers.gmm_bigru import (
    build_rollout_features_from_requests,
    load_gmm_params_json_dict,
)
from model.classifiers.gru import GRUClassifier
from model.classifiers.metrics import compute_power_metrics

AR1_MIN_RUN_LENGTH = 5
AR1_PHI_THRESHOLD = 0.3


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
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
    path_text = str(path_str).strip()
    if path_text == "":
        return None
    repo_root = Path(__file__).resolve().parents[2]
    repo_name = repo_root.name
    raw = Path(path_text)
    if raw.is_absolute():
        if raw.exists():
            return str(raw)
        # Handle manifests produced on a different machine where absolute
        # paths still contain ".../<repo_name>/...".
        parts = raw.parts
        if repo_name in parts:
            i = parts.index(repo_name)
            suffix = Path(*parts[i + 1 :]) if (i + 1) < len(parts) else Path()
            remapped = repo_root / suffix
            if remapped.exists():
                return str(remapped)
        return None
    local = Path(path_text)
    if local.exists():
        return str(local)
    from_base = Path(base_dir) / raw
    if from_base.exists():
        return str(from_base)
    # Pair manifests often store paths relative to repo root (e.g. "data/...").
    from_repo_root = repo_root / raw
    if from_repo_root.exists():
        return str(from_repo_root)
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


def _nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


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


def _resolve_checkpoint_norm_gmm_paths(
    config_entry: Dict[str, object],
    base_dir: str,
) -> Tuple[str, str, str]:
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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / np.clip(denom, a_min=1e-12, a_max=None)


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
    log_norm = -0.5 * (np.log(2.0 * np.pi * variances).reshape(1, -1) + ((x - means.reshape(1, -1)) ** 2) / variances.reshape(1, -1))
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
        z = logits.detach().cpu().numpy()
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


def generate_gmm_bigru_trace_ar1(
    *,
    logits: np.ndarray | torch.Tensor,
    gmm_params: Dict[str, object],
    phi: np.ndarray,
    sigma_innov: np.ndarray,
    p0: float,
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    sigma_marginal = np.asarray(sigma_innov, dtype=np.float64).reshape(-1)
    return generate_gmm_bigru_trace_ar1_thresholded(
        logits=logits,
        gmm_params=gmm_params,
        phi=phi,
        sigma_innov=sigma_innov,
        sigma_marginal=sigma_marginal,
        p0=p0,
        seed=seed,
        decode_mode=decode_mode,
        median_filter_window=median_filter_window,
        phi_threshold=0.0,
        clamp_range=clamp_range,
    )


def _plot_overlay(path: str, *, dt: float, gt: np.ndarray, pred: np.ndarray, title: str) -> None:
    n = int(min(len(gt), len(pred)))
    t = np.arange(n, dtype=np.float64) * float(dt)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, gt[:n], label="Measured", linewidth=1.5)
    ax.plot(t, pred[:n], label="Generated", linewidth=1.2, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_ar1_params(
    path: str,
    *,
    gmm_means: np.ndarray,
    phi: np.ndarray,
    sigma_marginal: np.ndarray,
    sigma_innov: np.ndarray,
    title: str,
    phi_threshold: float = AR1_PHI_THRESHOLD,
) -> None:
    means = np.asarray(gmm_means, dtype=np.float64).reshape(-1)
    phi_arr = np.asarray(phi, dtype=np.float64).reshape(-1)
    sigma_m = np.asarray(sigma_marginal, dtype=np.float64).reshape(-1)
    sigma_i = np.asarray(sigma_innov, dtype=np.float64).reshape(-1)
    K = int(means.size)
    if phi_arr.size != K or sigma_m.size != K or sigma_i.size != K:
        raise ValueError("AR(1) plot parameter size mismatch")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(K, dtype=np.int64)

    threshold = float(phi_threshold)
    colors = ["#d62728" if p >= threshold else "#2ca02c" for p in phi_arr]
    ax1.bar(x, phi_arr, color=colors, alpha=0.8)
    ax1.set_xlabel("GMM State (sorted by mean power)")
    ax1.set_ylabel("phi (AR(1) persistence)")
    ax1.set_title("Within-state persistence")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(m)}W" for m in means], rotation=45, fontsize=8)
    ax1.axhline(y=threshold, color="gray", linestyle="--", alpha=0.5, label=f"phi={threshold:.1f} threshold")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0.0, 1.0)

    ax2.bar(x - 0.15, sigma_m, width=0.3, label="sigma_marginal", alpha=0.8)
    ax2.bar(x + 0.15, sigma_i, width=0.3, label="sigma_innovation", alpha=0.8)
    ax2.set_xlabel("GMM State")
    ax2.set_ylabel("Std Dev (W)")
    ax2.set_title("Marginal vs Innovation Noise")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(m)}W" for m in means], rotation=45, fontsize=8)
    ax2.legend(fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _build_trace_record(
    *,
    trace_idx: int,
    pair_key: str,
    rate: str,
    power_start_epoch_s: float,
    power: np.ndarray,
    dt: float,
) -> Dict[str, Any]:
    p = np.asarray(power, dtype=np.float64).reshape(-1)
    if p.size < 2:
        raise ValueError(f"Trace {trace_idx} has length < 2")
    L = int(len(p) - 1)
    if L <= 0:
        raise ValueError(f"Trace {trace_idx} has no aligned points")
    return {
        "trace_idx": int(trace_idx),
        "pair_key": str(pair_key),
        "rate": str(rate),
        "power_start_epoch_s": float(power_start_epoch_s),
        "power": p[: L + 1],
        "ground_truth": p[1 : L + 1],
        "p0": float(p[0]),
        "dt": float(dt),
        "num_points": int(L),
    }


def evaluate_from_artifacts(
    *,
    run_manifest: str = "results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1/manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    out_dir: str = "results/continuous_v1_gmm_bigru/k10_f2/eval_metrics",
    config_ids: Optional[Sequence[str]] = None,
    num_seeds: int = 5,
    base_seed: int = 42,
    device: str = "auto",
    acf_max_lag: int = 50,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    plots: bool = True,
) -> Dict[str, object]:
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    if decode_mode not in {"stochastic", "argmax"}:
        raise ValueError(f"decode_mode must be one of {{'stochastic','argmax'}}; got {decode_mode}")

    run_manifest_payload = _load_json(run_manifest)
    run_cfgs = run_manifest_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    run_manifest_base = str(Path(run_manifest).resolve().parent)

    experimental_payload = _load_json(experimental_manifest)
    experimental_base = str(Path(experimental_manifest).resolve().parent)

    throughput_payload = _load_json(throughput_db)
    pair_map = _load_pair_manifest_map(pair_manifest_csv)

    requested = _parse_config_ids(config_ids)
    if requested:
        targets = requested
    else:
        targets = sorted([cid for cid, row in run_cfgs.items() if isinstance(row, dict) and row.get("status") == "trained"])

    resolved_device = _resolve_device(device)
    _ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plots_dir)
    ar1_params_dir = os.path.join(str(Path(out_dir).parent), "ar1_params")
    _ensure_dir(ar1_params_dir)

    per_seed_rows: List[Dict[str, object]] = []
    per_trace_rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []
    config_results: Dict[str, Dict[str, object]] = {}
    seeds = [int(base_seed) + i for i in range(int(num_seeds))]

    for config_id in targets:
        row = run_cfgs.get(config_id)
        if not isinstance(row, dict):
            cfg_row = {"config_id": config_id, "status": "skipped", "reason": "config_not_in_run_manifest"}
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
            continue
        if row.get("status") != "trained":
            cfg_row = {
                "config_id": config_id,
                "status": "skipped",
                "reason": f"config_status_{row.get('status', 'unknown')}",
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
            continue

        try:
            checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(row, run_manifest_base)
            norm_payload = _load_json(norm_path)
            norm_cfg = _extract_norm_for_eval(norm_payload)
            gmm_payload = _load_json(gmm_path)
            gmm_cfg = load_gmm_params_json_dict(gmm_payload)

            k = int(row.get("k", gmm_cfg["k"]))
            feature_set = str(row.get("feature_set", norm_payload.get("feature_set", "f2"))).lower()
            if feature_set not in {"f2", "f3"}:
                raise ValueError(f"invalid feature_set for '{config_id}': {feature_set}")
            input_dim = int(row.get("input_dim", 2 if feature_set == "f2" else 3))
            hidden_dim = int(row.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
            num_layers = int(row.get("num_layers", norm_payload.get("num_layers", 1)))
            if k != int(gmm_cfg["k"]):
                raise ValueError(f"k mismatch between run manifest ({k}) and gmm payload ({int(gmm_cfg['k'])})")

            model = _load_model(
                checkpoint_path=checkpoint_path,
                k=k,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                device=resolved_device,
            )
            throughput = _resolve_throughput(throughput_payload, config_id)
            dataset_path, split_path = _resolve_experimental_paths(
                experimental_payload,
                config_id=config_id,
                experimental_base=experimental_base,
            )
            split_payload = _load_json(split_path)
            test_indices = [int(x) for x in split_payload.get("test_indices", [])]
            train_indices = [int(x) for x in split_payload.get("train_indices", [])]
            if len(test_indices) == 0:
                raise ValueError("empty test split")

            with np.load(dataset_path, allow_pickle=True) as data:
                pair_key_arr = np.asarray(data["pair_key"], dtype=object)
                power_arr = np.asarray(data["power"], dtype=object)
                power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
                rate_arr = np.asarray(data["rate"], dtype=object) if "rate" in data else np.asarray([], dtype=object)
                dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)
            if dt_arr.size == 0:
                raise ValueError("dataset dt missing")
            dt = float(dt_arr[0])
            if (not np.isfinite(dt)) or dt <= 0.0:
                raise ValueError(f"invalid dt in dataset: {dt}")
            n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))

            training_power_traces: List[np.ndarray] = []
            training_labels_traces: List[np.ndarray] = []
            for idx in train_indices:
                if idx < 0 or idx >= n_total:
                    continue
                p_train = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
                if p_train.size == 0:
                    continue
                labels_train = predict_sorted_gmm_labels_from_params(p_train, gmm_cfg)
                training_power_traces.append(p_train.astype(np.float64))
                training_labels_traces.append(labels_train.astype(np.int64))

            phi, sigma_innov, sigma_marginal = estimate_ar1_params(
                gmm_params=gmm_cfg,
                training_power_traces=training_power_traces,
                training_labels_traces=training_labels_traces,
                K=int(k),
                min_run_length=AR1_MIN_RUN_LENGTH,
            )
            phi_above_threshold = phi >= float(AR1_PHI_THRESHOLD)
            slug = _safe_slug(config_id)
            ar1_params_path = os.path.join(ar1_params_dir, f"{slug}_ar1_params.json")
            _write_json(
                ar1_params_path,
                {
                    "config_id": config_id,
                    "phi": phi.tolist(),
                    "phi_threshold": float(AR1_PHI_THRESHOLD),
                    "phi_above_threshold": [bool(v) for v in phi_above_threshold.tolist()],
                    "num_ar1_states": int(np.sum(phi_above_threshold)),
                    "num_iid_states": int(np.sum(~phi_above_threshold)),
                    "sigma_innov": sigma_innov.tolist(),
                    "sigma_marginal": sigma_marginal.tolist(),
                    "gmm_means": np.asarray(gmm_cfg["means"], dtype=np.float64).reshape(-1).tolist(),
                    "min_run_length": int(AR1_MIN_RUN_LENGTH),
                },
            )

            trace_records: List[Dict[str, Any]] = []
            for idx in test_indices:
                if idx < 0 or idx >= n_total:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": int(idx),
                            "pair_key": "",
                            "status": "skipped",
                            "reason": "test_index_out_of_bounds",
                        }
                    )
                    continue
                try:
                    tr = _build_trace_record(
                        trace_idx=int(idx),
                        pair_key=str(pair_key_arr[idx]),
                        rate=str(rate_arr[idx]) if idx < len(rate_arr) else "",
                        power_start_epoch_s=float(power_start_arr[idx]),
                        power=np.asarray(power_arr[idx], dtype=np.float64),
                        dt=dt,
                    )
                    trace_records.append(tr)
                except Exception as exc:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": int(idx),
                            "pair_key": str(pair_key_arr[idx]) if idx < len(pair_key_arr) else "",
                            "status": "skipped",
                            "reason": f"trace_load_error:{type(exc).__name__}:{exc}",
                        }
                    )

            if len(trace_records) == 0:
                raise ValueError("no valid test traces to evaluate")

            eval_trace_rows: List[Dict[str, object]] = []
            representative_trace_idx = int(trace_records[0]["trace_idx"])
            representative_seed = int(seeds[0])
            representative_gt: Optional[np.ndarray] = None
            representative_pred: Optional[np.ndarray] = None

            for tr in trace_records:
                trace_idx = int(tr["trace_idx"])
                pair_key = str(tr["pair_key"])
                json_path = pair_map.get(pair_key)
                if json_path is None:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": trace_idx,
                            "pair_key": pair_key,
                            "status": "skipped",
                            "reason": "pair_key_not_found_in_pair_manifest",
                        }
                    )
                    continue

                try:
                    requests = _build_requests_from_stage0_json(
                        json_path,
                        power_start_epoch_s=float(tr["power_start_epoch_s"]),
                        trace_duration_s=float((int(tr["num_points"]) + 1) * dt),
                        dt=dt,
                    )
                    feat = build_rollout_features_from_requests(
                        requests=requests,
                        throughput=throughput,
                        norm=norm_cfg,
                        T=int(tr["num_points"]),
                        dt=dt,
                        feature_set=feature_set,
                    )
                    features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
                    if features_norm.ndim != 2 or features_norm.shape[1] != input_dim:
                        raise ValueError(f"rollout feature shape mismatch: {features_norm.shape} vs input_dim={input_dim}")

                    with torch.no_grad():
                        x = torch.from_numpy(features_norm).to(device=resolved_device, dtype=torch.float32).unsqueeze(0)
                        logits = model(x)[0].detach().cpu().numpy()

                    gt = np.asarray(tr["ground_truth"], dtype=np.float64).reshape(-1)
                    if gt.size == 0:
                        raise ValueError("empty ground truth trace")

                    seed_rows: List[Dict[str, object]] = []
                    pred_by_seed: Dict[int, np.ndarray] = {}
                    for seed_value in seeds:
                        gen = generate_gmm_bigru_trace_ar1_thresholded(
                            logits=logits,
                            gmm_params=gmm_cfg,
                            phi=phi,
                            sigma_innov=sigma_innov,
                            sigma_marginal=sigma_marginal,
                            p0=float(tr["p0"]),
                            seed=int(seed_value),
                            decode_mode=decode_mode,
                            median_filter_window=int(median_filter_window),
                            phi_threshold=float(AR1_PHI_THRESHOLD),
                            clamp_range=(norm_cfg["power_min"], norm_cfg["power_max"]),
                        )
                        pred = np.asarray(gen["power_w"], dtype=np.float64).reshape(-1)
                        n = int(min(len(gt), len(pred)))
                        if n <= 0:
                            raise ValueError("no aligned points after generation")
                        gt_n = gt[:n]
                        pred_n = pred[:n]
                        metrics = compute_power_metrics(
                            gt_n,
                            pred_n,
                            dt=dt,
                            acf_max_lag=int(acf_max_lag),
                        )
                        seed_row = {
                            "config_id": config_id,
                            "trace_idx": trace_idx,
                            "pair_key": pair_key,
                            "seed": int(seed_value),
                            "num_points": int(n),
                            "status": "ok",
                            "reason": "",
                            **metrics,
                        }
                        per_seed_rows.append(seed_row)
                        seed_rows.append(seed_row)
                        pred_by_seed[int(seed_value)] = pred_n

                    trace_row = {
                        "config_id": config_id,
                        "trace_idx": trace_idx,
                        "pair_key": pair_key,
                        "rate": str(tr["rate"]),
                        "status": "evaluated",
                        "reason": "",
                        "num_requests": int(len(requests)),
                        "num_points": int(seed_rows[0]["num_points"]) if seed_rows else int(tr["num_points"]),
                        "dt": float(dt),
                        "num_seeds": int(len(seed_rows)),
                        "seeds": ";".join(str(x) for x in seeds),
                        "ks_stat_median": _nanmedian(r["ks_stat"] for r in seed_rows),
                        "acf_r2_median": _nanmedian(r["acf_r2"] for r in seed_rows),
                        "nrmse_median": _nanmedian(r["nrmse"] for r in seed_rows),
                        "p95_error_pct_median": _nanmedian(r["p95_error_pct"] for r in seed_rows),
                        "p99_error_pct_median": _nanmedian(r["p99_error_pct"] for r in seed_rows),
                        "delta_energy_pct_median": _nanmedian(r["delta_energy_pct"] for r in seed_rows),
                    }
                    per_trace_rows.append(trace_row)
                    eval_trace_rows.append(trace_row)

                    if trace_idx == representative_trace_idx:
                        nrmse_vals = np.asarray([float(r["nrmse"]) for r in seed_rows], dtype=np.float64)
                        med_nrmse = float(np.median(nrmse_vals))
                        best_i = int(np.argmin(np.abs(nrmse_vals - med_nrmse)))
                        representative_seed = int(seed_rows[best_i]["seed"])
                        representative_gt = gt[: len(pred_by_seed[representative_seed])]
                        representative_pred = pred_by_seed[representative_seed]
                except Exception as exc:
                    per_trace_rows.append(
                        {
                            "config_id": config_id,
                            "trace_idx": trace_idx,
                            "pair_key": pair_key,
                            "status": "failed",
                            "reason": f"{type(exc).__name__}:{exc}",
                        }
                    )

            if len(eval_trace_rows) == 0:
                raise ValueError("all test traces failed or were skipped")

            plot_paths: Dict[str, str] = {}
            ar1_params_plot_path = ""
            if plots and representative_gt is not None and representative_pred is not None:
                stem = f"{slug}_trace{representative_trace_idx}"
                overlay_path = os.path.join(plots_dir, f"{stem}_overlay.png")
                _plot_overlay(
                    overlay_path,
                    dt=dt,
                    gt=representative_gt,
                    pred=representative_pred,
                    title=f"{config_id} trace={representative_trace_idx} generated vs measured",
                )
                plot_paths = {
                    "overlay_plot": overlay_path,
                }
            if plots:
                ar1_params_plot_path = os.path.join(plots_dir, f"{slug}_ar1_params.png")
                _plot_ar1_params(
                    ar1_params_plot_path,
                    gmm_means=np.asarray(gmm_cfg["means"], dtype=np.float64).reshape(-1),
                    phi=phi,
                    sigma_marginal=sigma_marginal,
                    sigma_innov=sigma_innov,
                    title=f"{config_id} AR(1) parameter diagnostics",
                    phi_threshold=float(AR1_PHI_THRESHOLD),
                )

            cfg_row = {
                "config_id": config_id,
                "status": "evaluated",
                "reason": "",
                "generation_mode": "ar1_thresholded",
                "k": int(k),
                "feature_set": feature_set,
                "decode_mode": decode_mode,
                "median_filter_window": int(median_filter_window),
                "gmm_covariance_type": str(gmm_cfg.get("covariance_type", "full")),
                "num_test_traces": int(len(test_indices)),
                "num_eval_traces": int(len(eval_trace_rows)),
                "num_skipped_or_failed_traces": int(len(test_indices) - len(eval_trace_rows)),
                "num_seeds": int(num_seeds),
                "ks_stat_median": _nanmedian(r["ks_stat_median"] for r in eval_trace_rows),
                "acf_r2_median": _nanmedian(r["acf_r2_median"] for r in eval_trace_rows),
                "nrmse_median": _nanmedian(r["nrmse_median"] for r in eval_trace_rows),
                "p95_error_pct_median": _nanmedian(r["p95_error_pct_median"] for r in eval_trace_rows),
                "p99_error_pct_median": _nanmedian(r["p99_error_pct_median"] for r in eval_trace_rows),
                "delta_energy_pct_median": _nanmedian(r["delta_energy_pct_median"] for r in eval_trace_rows),
                "phi_median": _nanmedian(phi),
                "representative_trace_idx": int(representative_trace_idx),
                "representative_seed": int(representative_seed),
                "ar1_params_json": ar1_params_path,
                "ar1_params_plot": ar1_params_plot_path,
                **plot_paths,
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)
        except Exception as exc:
            cfg_row = {
                "config_id": config_id,
                "status": "failed",
                "reason": f"{type(exc).__name__}:{exc}",
            }
            config_rows.append(cfg_row)
            config_results[config_id] = dict(cfg_row)

    per_seed_fields = [
        "config_id",
        "trace_idx",
        "pair_key",
        "seed",
        "status",
        "reason",
        "num_points",
        "ks_stat",
        "acf_r2",
        "nrmse",
        "p95_error_pct",
        "p99_error_pct",
        "delta_energy_pct",
    ]
    for r in per_seed_rows:
        for f in per_seed_fields:
            r.setdefault(f, "")
    per_seed_csv = os.path.join(out_dir, "per_seed_metrics.csv")
    _write_csv(per_seed_csv, per_seed_rows, per_seed_fields)

    per_trace_fields = [
        "config_id",
        "trace_idx",
        "pair_key",
        "rate",
        "status",
        "reason",
        "num_requests",
        "num_points",
        "dt",
        "num_seeds",
        "seeds",
        "ks_stat_median",
        "acf_r2_median",
        "nrmse_median",
        "p95_error_pct_median",
        "p99_error_pct_median",
        "delta_energy_pct_median",
    ]
    for r in per_trace_rows:
        for f in per_trace_fields:
            r.setdefault(f, "")
    per_trace_csv = os.path.join(out_dir, "per_trace_metrics.csv")
    _write_csv(per_trace_csv, per_trace_rows, per_trace_fields)

    config_fields = [
        "config_id",
        "status",
        "reason",
        "generation_mode",
        "k",
        "feature_set",
        "decode_mode",
        "median_filter_window",
        "gmm_covariance_type",
        "num_test_traces",
        "num_eval_traces",
        "num_skipped_or_failed_traces",
        "num_seeds",
        "ks_stat_median",
        "acf_r2_median",
        "nrmse_median",
        "p95_error_pct_median",
        "p99_error_pct_median",
        "delta_energy_pct_median",
        "phi_median",
        "representative_trace_idx",
        "representative_seed",
        "ar1_params_json",
        "ar1_params_plot",
        "overlay_plot",
    ]
    for r in config_rows:
        for f in config_fields:
            r.setdefault(f, "")
    config_csv = os.path.join(out_dir, "config_summary.csv")
    _write_csv(config_csv, config_rows, config_fields)

    run_manifest_payload = {
        "schema_version": "continuous-v1-gmm-bigru-eval-run-v2",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "run_manifest": run_manifest,
            "experimental_manifest": experimental_manifest,
            "throughput_db": throughput_db,
            "pair_manifest_csv": pair_manifest_csv,
        },
        "defaults": {
            "out_dir": out_dir,
            "num_seeds": int(num_seeds),
            "base_seed": int(base_seed),
            "acf_max_lag": int(acf_max_lag),
            "decode_mode": str(decode_mode),
            "median_filter_window": int(median_filter_window),
            "device": str(resolved_device),
            "plots": bool(plots),
            "generation_mode": "ar1_thresholded",
            "phi_threshold": float(AR1_PHI_THRESHOLD),
            "min_run_length": int(AR1_MIN_RUN_LENGTH),
        },
        "summary": {
            "num_target_configs": int(len(targets)),
            "num_evaluated_configs": int(sum(1 for r in config_rows if r.get("status") == "evaluated")),
            "num_failed_configs": int(sum(1 for r in config_rows if r.get("status") == "failed")),
            "num_skipped_configs": int(sum(1 for r in config_rows if r.get("status") == "skipped")),
        },
        "artifacts": {
            "per_seed_metrics_csv": per_seed_csv,
            "per_trace_metrics_csv": per_trace_csv,
            "config_summary_csv": config_csv,
            "plots_dir": plots_dir,
            "ar1_params_dir": ar1_params_dir,
        },
        "configs": config_results,
    }
    run_manifest_out = os.path.join(out_dir, "run_manifest.json")
    _write_json(run_manifest_out, run_manifest_payload)
    return run_manifest_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate continuous v1 GMM+BiGRU models on test traces.")
    parser.add_argument("--run-manifest", default="results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json")
    parser.add_argument("--experimental-manifest", default="results/experimental_continuous_v1/manifest.json")
    parser.add_argument("--throughput-db", default="model/config/throughput_database.json")
    parser.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    parser.add_argument("--out-dir", default="results/continuous_v1_gmm_bigru/k10_f2/eval_metrics")
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--acf-max-lag", type=int, default=50)
    parser.add_argument("--decode-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run = evaluate_from_artifacts(
        run_manifest=args.run_manifest,
        experimental_manifest=args.experimental_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        out_dir=args.out_dir,
        config_ids=args.config_id,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
        device=args.device,
        acf_max_lag=args.acf_max_lag,
        decode_mode=args.decode_mode,
        median_filter_window=args.median_filter_window,
        plots=(not args.no_plots),
    )

    print("[eval_gmm_bigru] Done")
    for k, v in run.get("summary", {}).items():
        print(f"  {k}: {v}")
    artifacts = run.get("artifacts", {})
    print(f"  per_seed_metrics : {artifacts.get('per_seed_metrics_csv', '')}")
    print(f"  per_trace_metrics: {artifacts.get('per_trace_metrics_csv', '')}")
    print(f"  config_summary   : {artifacts.get('config_summary_csv', '')}")
    print(f"  run_manifest     : {os.path.join(args.out_dir, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
