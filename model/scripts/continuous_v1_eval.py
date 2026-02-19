#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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

from model.classifiers.continuous_gru import (
    MeanRevertingGRU,
    compute_inference_features,
    generate_mean_reverting_trace,
)

EPS = 1e-12


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


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    ys = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if xs.size == 0 or ys.size == 0:
        return float("nan")
    values = np.concatenate([xs, ys])
    values.sort()
    cdf_x = np.searchsorted(xs, values, side="right") / float(xs.size)
    cdf_y = np.searchsorted(ys, values, side="right") / float(ys.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n == 0:
        return np.zeros((1,), dtype=np.float64)
    lag = int(max(0, min(max_lag, n - 1)))
    centered = arr - float(np.mean(arr))
    denom = float(np.dot(centered, centered))
    out = np.zeros((lag + 1,), dtype=np.float64)
    out[0] = 1.0
    if lag == 0 or denom <= EPS:
        return out
    for k in range(1, lag + 1):
        out[k] = float(np.dot(centered[:-k], centered[k:]) / denom)
    return out


def _integrate_trapezoid(values: np.ndarray, *, dx: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(arr, dx=float(dx)))
    return float(np.trapz(arr, dx=float(dx)))


def autocorrelation_r2(real: np.ndarray, synthetic: np.ndarray, max_lag: int = 50) -> float:
    r = np.asarray(real, dtype=np.float64).reshape(-1)
    s = np.asarray(synthetic, dtype=np.float64).reshape(-1)
    if r.size < 3 or s.size < 3:
        return float("nan")
    lag = int(min(max_lag, r.size - 1, s.size - 1))
    if lag < 1:
        return float("nan")
    r_acf = _acf(r, lag)[1:]
    s_acf = _acf(s, lag)[1:]
    if r_acf.size == 0:
        return float("nan")
    sse = float(np.sum((r_acf - s_acf) ** 2))
    tss = float(np.sum((r_acf - float(np.mean(r_acf))) ** 2))
    if tss <= EPS:
        return float(1.0 if sse <= 1e-10 else 0.0)
    return float(1.0 - (sse / tss))


def compute_power_metrics(
    ground_truth_w: np.ndarray,
    generated_w: np.ndarray,
    *,
    dt: float,
    acf_max_lag: int = 50,
) -> Dict[str, float]:
    gt = np.asarray(ground_truth_w, dtype=np.float64).reshape(-1)
    pred = np.asarray(generated_w, dtype=np.float64).reshape(-1)
    n = int(min(len(gt), len(pred)))
    if n <= 0:
        raise ValueError("No aligned points for metric computation.")
    gt = gt[:n]
    pred = pred[:n]

    err = pred - gt
    rmse = float(np.sqrt(np.mean(err**2)))
    scale = float(np.max(gt) - np.min(gt))
    nrmse = float(rmse / (scale + EPS))

    p95_gt = float(np.percentile(gt, 95))
    p95_pred = float(np.percentile(pred, 95))
    p99_gt = float(np.percentile(gt, 99))
    p99_pred = float(np.percentile(pred, 99))

    p95_error_pct = float(100.0 * abs(p95_pred - p95_gt) / (abs(p95_gt) + EPS))
    p99_error_pct = float(100.0 * abs(p99_pred - p99_gt) / (abs(p99_gt) + EPS))

    energy_gt = _integrate_trapezoid(gt, dx=float(dt))
    energy_pred = _integrate_trapezoid(pred, dx=float(dt))
    delta_energy_pct = float(100.0 * (energy_pred - energy_gt) / (abs(energy_gt) + EPS))

    return {
        "ks_stat": ks_statistic(gt, pred),
        "acf_r2": autocorrelation_r2(gt, pred, max_lag=int(acf_max_lag)),
        "nrmse": nrmse,
        "p95_error_pct": p95_error_pct,
        "p99_error_pct": p99_error_pct,
        "delta_energy_pct": delta_energy_pct,
    }


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
    )
    missing = [k for k in required if k not in norm_payload]
    if missing:
        raise ValueError(f"Norm params missing keys: {missing}")

    out = {
        "A_mean": float(norm_payload["active_mean"]),
        "A_std": float(norm_payload["active_std"]),
        "T_arrive_log_mean": float(norm_payload["t_arrive_log_mean"]),
        "T_arrive_log_std": float(norm_payload["t_arrive_log_std"]),
        "power_mean": float(norm_payload["power_mean"]),
        "power_std": float(norm_payload["power_std"]),
        "power_min": float(norm_payload["power_min"]),
        "power_max": float(norm_payload["power_max"]),
    }
    if out["A_std"] <= 0.0:
        raise ValueError("A_std must be positive")
    if out["T_arrive_log_std"] <= 0.0:
        raise ValueError("T_arrive_log_std must be positive")
    if out["power_std"] <= 0.0:
        raise ValueError("power_std must be positive")
    return out


def _resolve_checkpoint_and_norm_paths(
    config_entry: Dict[str, object],
    base_dir: str,
) -> Tuple[str, str]:
    checkpoint_raw = str(config_entry.get("checkpoint_path", ""))
    norm_raw = str(config_entry.get("norm_params_path", ""))
    checkpoint_path = _resolve_existing_path(checkpoint_raw, base_dir)
    norm_path = _resolve_existing_path(norm_raw, base_dir)
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint path not found: {checkpoint_raw}")
    if norm_path is None:
        raise ValueError(f"Norm params path not found: {norm_raw}")
    return checkpoint_path, norm_path


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
    norm_payload: Dict[str, object],
    n_mix: int,
    device: torch.device,
) -> MeanRevertingGRU:
    hidden_dim = int(norm_payload.get("hidden_dim", 64))
    num_layers = int(norm_payload.get("num_layers", 1))
    model = MeanRevertingGRU(
        input_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_mix=int(n_mix),
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
    z = z - float(np.max(z, axis=-1, keepdims=True))
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / np.clip(denom, a_min=EPS, a_max=None)


def _teacher_forced_diagnostics(
    *,
    model: MeanRevertingGRU,
    p_prev_norm: np.ndarray,
    features_norm: np.ndarray,
    p_target_norm: np.ndarray,
    norm_cfg: Dict[str, float],
    n_mix: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    L = int(min(len(p_prev_norm), len(features_norm), len(p_target_norm)))
    if L <= 0:
        raise ValueError("empty trace for diagnostics")
    p_prev = np.asarray(p_prev_norm[:L], dtype=np.float32).reshape(-1, 1)
    feats = np.asarray(features_norm[:L], dtype=np.float32).reshape(-1, 2)
    p_tgt = np.asarray(p_target_norm[:L], dtype=np.float32).reshape(-1, 1)

    x = np.concatenate([p_prev, feats], axis=-1)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        params, _ = model(x_t)

    alpha = params["alpha"][0, :, 0].detach().cpu().numpy().astype(np.float64)
    p_prev_vec = p_prev[:, 0].astype(np.float64)
    p_target_vec = p_tgt[:, 0].astype(np.float64)

    if int(n_mix) == 1:
        mu_norm = params["mu"][0, :, 0].detach().cpu().numpy().astype(np.float64)
        sigma_norm = torch.exp(params["log_sigma"][0, :, 0]).detach().cpu().numpy().astype(np.float64)
        pi = np.ones((L, 1), dtype=np.float64)
        n_active = np.ones((L,), dtype=np.float64)
    else:
        logits = params["logit_pi"][0].detach().cpu().numpy().astype(np.float64)
        pi = _softmax_np(logits)
        mu_components = params["mu"][0].detach().cpu().numpy().astype(np.float64)
        sigma_components = torch.exp(params["log_sigma"][0]).detach().cpu().numpy().astype(np.float64)
        mu_norm = np.sum(pi * mu_components, axis=-1)
        sigma_norm = np.sum(pi * sigma_components, axis=-1)
        entropy = -np.sum(pi * np.log(np.clip(pi, a_min=EPS, a_max=None)), axis=-1)
        n_active = np.exp(entropy)

    pred_mean_norm = ((1.0 - alpha) * p_prev_vec) + (alpha * mu_norm)

    pm = float(norm_cfg["power_mean"])
    ps = float(norm_cfg["power_std"])
    mu_raw = (mu_norm * ps) + pm
    pred_mean_raw = (pred_mean_norm * ps) + pm
    target_raw = (p_target_vec * ps) + pm
    sigma_raw = sigma_norm * ps

    return {
        "mu_norm": mu_norm,
        "mu_raw": mu_raw,
        "pred_mean_raw": pred_mean_raw,
        "target_raw": target_raw,
        "alpha": alpha,
        "sigma_norm": sigma_norm,
        "sigma_raw": sigma_raw,
        "pi": pi,
        "n_active": n_active,
    }


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


def _plot_mu_vs_actual(
    path: str,
    *,
    dt: float,
    actual: np.ndarray,
    mu_raw: np.ndarray,
    pred_mean_raw: np.ndarray,
    title: str,
) -> None:
    n = int(min(len(actual), len(mu_raw), len(pred_mean_raw)))
    t = np.arange(n, dtype=np.float64) * float(dt)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, actual[:n], label="Measured", linewidth=1.5)
    ax.plot(t, mu_raw[:n], label="Learned μ_t", linewidth=1.2)
    ax.plot(t, pred_mean_raw[:n], label="Conditional Mean", linewidth=1.2, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_scalar_series(path: str, *, dt: float, values: np.ndarray, ylabel: str, title: str) -> None:
    n = int(len(values))
    t = np.arange(n, dtype=np.float64) * float(dt)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, values, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_mixture_weights(path: str, *, dt: float, pi: np.ndarray, title: str) -> None:
    n = int(pi.shape[0])
    t = np.arange(n, dtype=np.float64) * float(dt)
    fig, ax = plt.subplots(figsize=(12, 4))
    for k in range(int(pi.shape[1])):
        ax.plot(t, pi[:, k], linewidth=1.1, label=f"pi_{k}")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mixture Weight")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", ncol=min(3, int(pi.shape[1])))
    ax.grid(True, alpha=0.25)
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
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    dt: float,
) -> Dict[str, Any]:
    p = np.asarray(power, dtype=np.float64).reshape(-1)
    x = np.asarray(x_norm, dtype=np.float32)
    y = np.asarray(y_norm, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError(f"Trace {trace_idx} has invalid x_norm shape: {x.shape}")
    if p.size < 2:
        raise ValueError(f"Trace {trace_idx} has length < 2")

    L = int(min(len(x), len(y), len(p) - 1))
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
        "x_norm": x[:L, :3].astype(np.float32),
        "y_norm": y[:L].astype(np.float32),
        "p_prev_norm": x[:L, 0].astype(np.float32),
        "features_norm_train": x[:L, 1:3].astype(np.float32),
        "p_target_norm": y[:L].astype(np.float32),
        "dt": float(dt),
        "num_points": int(L),
    }


def evaluate_from_artifacts(
    *,
    run_manifest: str = "results/continuous_v1/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1/manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    out_dir: str = "results/continuous_v1/eval_metrics",
    config_ids: Optional[Sequence[str]] = None,
    num_seeds: int = 5,
    base_seed: int = 42,
    device: str = "auto",
    acf_max_lag: int = 50,
    plots: bool = True,
) -> Dict[str, object]:
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")

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

    per_seed_rows: List[Dict[str, object]] = []
    per_trace_rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []
    config_results: Dict[str, Dict[str, object]] = {}

    seeds = [int(base_seed) + i for i in range(int(num_seeds))]

    for config_id in targets:
        row = run_cfgs.get(config_id)
        if not isinstance(row, dict):
            config_row = {
                "config_id": config_id,
                "status": "skipped",
                "reason": "config_not_in_run_manifest",
            }
            config_rows.append(config_row)
            config_results[config_id] = dict(config_row)
            continue
        if row.get("status") != "trained":
            config_row = {
                "config_id": config_id,
                "status": "skipped",
                "reason": f"config_status_{row.get('status', 'unknown')}",
            }
            config_rows.append(config_row)
            config_results[config_id] = dict(config_row)
            continue

        try:
            checkpoint_path, norm_path = _resolve_checkpoint_and_norm_paths(row, run_manifest_base)
            norm_payload = _load_json(norm_path)
            norm_cfg = _extract_norm_for_eval(norm_payload)
            n_mix = int(row.get("n_mix", norm_payload.get("n_mix", 1)))
            if n_mix < 1:
                raise ValueError(f"Invalid n_mix for '{config_id}': {n_mix}")
            model = _load_model(
                checkpoint_path=checkpoint_path,
                norm_payload=norm_payload,
                n_mix=n_mix,
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
            if len(test_indices) == 0:
                raise ValueError("empty test split")

            with np.load(dataset_path, allow_pickle=True) as data:
                pair_key_arr = np.asarray(data["pair_key"], dtype=object)
                power_arr = np.asarray(data["power"], dtype=object)
                power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
                rate_arr = np.asarray(data["rate"], dtype=object) if "rate" in data else np.asarray([], dtype=object)
                x_norm_arr = np.asarray(data["x_norm"], dtype=object)
                y_norm_arr = np.asarray(data["y_norm"], dtype=object)
                dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)
            if dt_arr.size == 0:
                raise ValueError("dataset dt missing")
            dt = float(dt_arr[0])
            if (not np.isfinite(dt)) or dt <= 0.0:
                raise ValueError(f"invalid dt in dataset: {dt}")

            trace_records: List[Dict[str, Any]] = []
            n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr), len(x_norm_arr), len(y_norm_arr)))
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
                        x_norm=np.asarray(x_norm_arr[idx], dtype=np.float32),
                        y_norm=np.asarray(y_norm_arr[idx], dtype=np.float32),
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

            sigma_sum = 0.0
            alpha_sum = 0.0
            active_sum = 0.0
            diag_points = 0
            eval_trace_rows: List[Dict[str, object]] = []

            representative_trace_idx = int(trace_records[0]["trace_idx"])
            representative_seed = int(seeds[0])
            representative_gt: Optional[np.ndarray] = None
            representative_pred: Optional[np.ndarray] = None
            representative_diag: Optional[Dict[str, np.ndarray]] = None

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
                    features_norm = compute_inference_features(
                        requests=requests,
                        config={
                            "lambda_prefill": throughput["lambda_prefill"],
                            "lambda_decode": throughput["lambda_decode"],
                            "A_mean": norm_cfg["A_mean"],
                            "A_std": norm_cfg["A_std"],
                            "T_arrive_log_mean": norm_cfg["T_arrive_log_mean"],
                            "T_arrive_log_std": norm_cfg["T_arrive_log_std"],
                        },
                        T=int(tr["num_points"]),
                        dt=dt,
                    )
                    gt = np.asarray(tr["ground_truth"], dtype=np.float64).reshape(-1)
                    if len(features_norm) <= 0 or len(gt) <= 0:
                        raise ValueError("empty aligned sequence")

                    seed_rows: List[Dict[str, object]] = []
                    pred_by_seed: Dict[int, np.ndarray] = {}
                    for seed_value in seeds:
                        pred = generate_mean_reverting_trace(
                            model=model,
                            features_norm=features_norm,
                            P_0=float(tr["p0"]),
                            config_norm={
                                "power_mean": norm_cfg["power_mean"],
                                "power_std": norm_cfg["power_std"],
                                "power_min": norm_cfg["power_min"],
                                "power_max": norm_cfg["power_max"],
                            },
                            n_mix=n_mix,
                            seed=int(seed_value),
                        )
                        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
                        n = int(min(len(gt), len(pred)))
                        if n <= 0:
                            raise ValueError("no aligned points after generation")
                        pred = pred[:n]
                        gt_n = gt[:n]
                        metrics = compute_power_metrics(
                            gt_n,
                            pred,
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
                        pred_by_seed[int(seed_value)] = pred

                    diag = _teacher_forced_diagnostics(
                        model=model,
                        p_prev_norm=np.asarray(tr["p_prev_norm"], dtype=np.float32),
                        features_norm=np.asarray(tr["features_norm_train"], dtype=np.float32),
                        p_target_norm=np.asarray(tr["p_target_norm"], dtype=np.float32),
                        norm_cfg=norm_cfg,
                        n_mix=n_mix,
                        device=resolved_device,
                    )
                    sigma_mean = float(np.mean(diag["sigma_raw"]))
                    alpha_mean = float(np.mean(diag["alpha"]))
                    active_mean = float(np.mean(diag["n_active"]))
                    sigma_sum += float(np.sum(diag["sigma_raw"]))
                    alpha_sum += float(np.sum(diag["alpha"]))
                    active_sum += float(np.sum(diag["n_active"]))
                    diag_points += int(len(diag["alpha"]))

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
                        "mean_sigma_w": sigma_mean,
                        "mean_alpha": alpha_mean,
                        "mean_active_components": active_mean,
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
                        representative_diag = diag
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
            if plots and representative_gt is not None and representative_pred is not None and representative_diag is not None:
                slug = _safe_slug(config_id)
                stem = f"{slug}_trace{representative_trace_idx}"
                overlay_path = os.path.join(plots_dir, f"{stem}_overlay.png")
                mu_path = os.path.join(plots_dir, f"{stem}_mu_vs_actual.png")
                alpha_path = os.path.join(plots_dir, f"{stem}_alpha.png")
                sigma_path = os.path.join(plots_dir, f"{stem}_sigma.png")
                _plot_overlay(
                    overlay_path,
                    dt=dt,
                    gt=representative_gt,
                    pred=representative_pred,
                    title=f"{config_id} trace={representative_trace_idx} generated vs measured",
                )
                _plot_mu_vs_actual(
                    mu_path,
                    dt=dt,
                    actual=np.asarray(representative_diag["target_raw"], dtype=np.float64),
                    mu_raw=np.asarray(representative_diag["mu_raw"], dtype=np.float64),
                    pred_mean_raw=np.asarray(representative_diag["pred_mean_raw"], dtype=np.float64),
                    title=f"{config_id} trace={representative_trace_idx} learned μ_t vs measured",
                )
                _plot_scalar_series(
                    alpha_path,
                    dt=dt,
                    values=np.asarray(representative_diag["alpha"], dtype=np.float64),
                    ylabel="alpha_t",
                    title=f"{config_id} trace={representative_trace_idx} learned alpha_t",
                )
                _plot_scalar_series(
                    sigma_path,
                    dt=dt,
                    values=np.asarray(representative_diag["sigma_raw"], dtype=np.float64),
                    ylabel="sigma_t (W)",
                    title=f"{config_id} trace={representative_trace_idx} learned sigma_t",
                )
                plot_paths = {
                    "overlay_plot": overlay_path,
                    "mu_vs_actual_plot": mu_path,
                    "alpha_plot": alpha_path,
                    "sigma_plot": sigma_path,
                }
                pi = np.asarray(representative_diag["pi"], dtype=np.float64)
                if pi.ndim == 2 and pi.shape[1] > 1:
                    mix_path = os.path.join(plots_dir, f"{stem}_mixture_weights.png")
                    _plot_mixture_weights(
                        mix_path,
                        dt=dt,
                        pi=pi,
                        title=f"{config_id} trace={representative_trace_idx} mixture weights",
                    )
                    plot_paths["mixture_weights_plot"] = mix_path

            cfg_row = {
                "config_id": config_id,
                "status": "evaluated",
                "reason": "",
                "n_mix": int(n_mix),
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
                "mean_sigma_w_test_timesteps": float(sigma_sum / max(diag_points, 1)),
                "mean_alpha_test_timesteps": float(alpha_sum / max(diag_points, 1)),
                "mean_active_components_test_timesteps": float(active_sum / max(diag_points, 1)),
                "representative_trace_idx": int(representative_trace_idx),
                "representative_seed": int(representative_seed),
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
        "mean_sigma_w",
        "mean_alpha",
        "mean_active_components",
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
        "n_mix",
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
        "mean_sigma_w_test_timesteps",
        "mean_alpha_test_timesteps",
        "mean_active_components_test_timesteps",
        "representative_trace_idx",
        "representative_seed",
        "overlay_plot",
        "mu_vs_actual_plot",
        "alpha_plot",
        "sigma_plot",
        "mixture_weights_plot",
    ]
    for r in config_rows:
        for f in config_fields:
            r.setdefault(f, "")
    config_csv = os.path.join(out_dir, "config_summary.csv")
    _write_csv(config_csv, config_rows, config_fields)

    run_manifest_payload = {
        "schema_version": "continuous-v1-eval-run-v1",
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
            "device": str(resolved_device),
            "plots": bool(plots),
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
        },
        "configs": config_results,
    }
    run_manifest_out = os.path.join(out_dir, "run_manifest.json")
    _write_json(run_manifest_out, run_manifest_payload)
    return run_manifest_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage v1 mean-reverting models on test traces.")
    parser.add_argument("--run-manifest", default="results/continuous_v1/run_manifest.json")
    parser.add_argument("--experimental-manifest", default="results/experimental_continuous_v1/manifest.json")
    parser.add_argument("--throughput-db", default="model/config/throughput_database.json")
    parser.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    parser.add_argument("--out-dir", default="results/continuous_v1/eval_metrics")
    parser.add_argument("--config-id", action="append", default=[], help="Optional config filter, repeatable/comma-separated.")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--acf-max-lag", type=int, default=50)
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
        plots=(not args.no_plots),
    )

    print("[continuous_v1_eval] Done")
    for k, v in run.get("summary", {}).items():
        print(f"  {k}: {v}")
    artifacts = run.get("artifacts", {})
    print(f"  per_seed_metrics : {artifacts.get('per_seed_metrics_csv', '')}")
    print(f"  per_trace_metrics: {artifacts.get('per_trace_metrics_csv', '')}")
    print(f"  config_summary   : {artifacts.get('config_summary_csv', '')}")
    print(f"  run_manifest     : {os.path.join(args.out_dir, 'run_manifest.json')}")


if __name__ == "__main__":
    main()
