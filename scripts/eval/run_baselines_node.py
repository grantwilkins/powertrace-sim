#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import torch

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.classifiers.gru import GRUClassifier
from model.scripts.continuous_v1_eval import compute_power_metrics
from scripts.eval.pipeline_utils import (
    build_rollout_features_from_requests,
    load_gmm_params_json_dict,
    estimate_ar1_params,
    predict_sorted_gmm_labels_from_params,
)
from scripts.eval.baselines import (
    build_splitwise_lut_params,
    generate_mean,
    generate_ours,
    generate_splitwise_lut,
    generate_tdp,
)

METHODS = ("tdp", "mean", "splitwise_lut", "ours")
STOCHASTIC_METHODS = {"ours"}
CONSTANT_METHODS = {"tdp", "mean"}
METRIC_KEYS = (
    "ks_stat",
    "acf_r2",
    "nrmse",
    "p95_error_pct",
    "p99_error_pct",
    "delta_energy_pct",
)

DEFAULT_CONFIG_IDS = [
    "deepseek-r1-distill-70b_A100_tp4",
    "deepseek-r1-distill-70b_H100_tp4",
    "llama-3-70b_A100_tp4",
    "llama-3-70b_H100_tp4",
]
CONFIG_70B_TP4_RE = re.compile(r"^.+-70b_(A100|H100)_tp4$")
CONFIG_MODEL_SIZE_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _resolve_device(device: Optional[Union[torch.device, str]]) -> torch.device:
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


def _is_70b_tp4_config(config_id: str) -> bool:
    return CONFIG_70B_TP4_RE.match(str(config_id).strip()) is not None


def _is_moe_config(config_id: str) -> bool:
    """
    Return True when config_id is treated as MoE in eval scripts.

    Current policy:
      - DeepSeek-R1-Distill is dense
      - GPT-OSS 20B+ is MoE
    """
    match = CONFIG_MODEL_SIZE_RE.match(str(config_id).strip())
    if match is None:
        return False
    model_family = str(match.group(1)).lower()
    model_size = int(match.group(2))

    if "deepseek-r1-distill" in model_family:
        return False
    if "gpt-oss" in model_family and model_size >= 20:
        return True
    return False


def _nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _suppress_short_true_runs(mask: np.ndarray, min_run_len: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size == 0 or int(min_run_len) <= 1:
        return arr
    out = arr.copy()
    i = 0
    n = int(arr.size)
    min_len = int(min_run_len)
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        if (j - i) < min_len:
            out[i:j] = False
        i = j
    return out


def _splitwise_phase_masks(
    *,
    a_raw: np.ndarray,
    delta_a_raw: np.ndarray,
    phase_eps: float = 1e-6,
    prefill_delta_threshold: float = 0.05,
    prefill_min_steps: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a_raw, dtype=np.float64).reshape(-1)
    da = np.asarray(delta_a_raw, dtype=np.float64).reshape(-1)
    n = int(min(a.size, da.size))
    if n <= 0:
        z = np.zeros((0,), dtype=bool)
        return z, z, z
    a_n = a[:n]
    da_n = da[:n]
    active = a_n > float(phase_eps)
    prefill = active & (da_n > max(float(phase_eps), float(prefill_delta_threshold)))
    prefill = _suppress_short_true_runs(prefill, int(prefill_min_steps))
    decode = active & (~prefill)
    idle = ~active
    return idle, decode, prefill


def _estimate_splitwise_phase_targets_from_indices(
    *,
    indices: Sequence[int],
    pair_key_arr: np.ndarray,
    power_arr: np.ndarray,
    power_start_arr: np.ndarray,
    pair_map: Mapping[str, str],
    throughput: Mapping[str, float],
    norm_cfg: Mapping[str, float],
    feature_set: str,
    dt: float,
    non_gpu_overhead_w: float,
    phase_eps: float = 1e-6,
    prefill_delta_threshold: float = 0.05,
    prefill_min_steps: int = 2,
) -> Dict[str, float]:
    idle_vals: List[np.ndarray] = []
    decode_vals: List[np.ndarray] = []
    prefill_vals: List[np.ndarray] = []

    n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))
    for idx_raw in indices:
        idx = int(idx_raw)
        if idx < 0 or idx >= n_total:
            continue
        power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if power.size < 2:
            continue
        pair_key = str(pair_key_arr[idx])
        json_path = pair_map.get(pair_key)
        if json_path is None:
            continue
        try:
            requests = _build_requests_from_stage0_json(
                json_path,
                power_start_epoch_s=float(power_start_arr[idx]),
                trace_duration_s=float(power.size * dt),
                dt=float(dt),
            )
            gt_node = power[1:].astype(np.float64)
            feat = build_rollout_features_from_requests(
                requests=requests,
                throughput=throughput,
                norm=norm_cfg,
                T=int(gt_node.size),
                dt=float(dt),
                feature_set=str(feature_set),
            )
        except Exception:
            continue

        a_raw = np.asarray(feat.get("A_raw", []), dtype=np.float64).reshape(-1)
        delta_a_raw = np.asarray(feat.get("delta_A_raw", []), dtype=np.float64).reshape(-1)
        n_eval = int(min(gt_node.size, a_raw.size, delta_a_raw.size))
        if n_eval <= 0:
            continue
        gpu_power = np.clip(gt_node[:n_eval] - float(non_gpu_overhead_w), a_min=0.0, a_max=None)
        idle_mask, decode_mask, prefill_mask = _splitwise_phase_masks(
            a_raw=a_raw[:n_eval],
            delta_a_raw=delta_a_raw[:n_eval],
            phase_eps=float(phase_eps),
            prefill_delta_threshold=float(prefill_delta_threshold),
            prefill_min_steps=int(prefill_min_steps),
        )
        if np.any(idle_mask):
            idle_vals.append(gpu_power[idle_mask].astype(np.float64))
        if np.any(decode_mask):
            decode_vals.append(gpu_power[decode_mask].astype(np.float64))
        if np.any(prefill_mask):
            prefill_vals.append(gpu_power[prefill_mask].astype(np.float64))

    out: Dict[str, float] = {}
    if len(idle_vals) > 0:
        out["target_idle_node_gpu_w"] = float(np.mean(np.concatenate(idle_vals)))
    if len(decode_vals) > 0:
        out["target_decode_node_gpu_w"] = float(np.mean(np.concatenate(decode_vals)))
    if len(prefill_vals) > 0:
        out["target_prefill_node_gpu_w"] = float(np.mean(np.concatenate(prefill_vals)))
    out["splitwise_phase_samples_idle"] = float(sum(int(x.size) for x in idle_vals))
    out["splitwise_phase_samples_decode"] = float(sum(int(x.size) for x in decode_vals))
    out["splitwise_phase_samples_prefill"] = float(sum(int(x.size) for x in prefill_vals))
    return out


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


def _load_or_estimate_ar1_params(
    *,
    config_id: str,
    gmm_params: Dict[str, object],
    train_power_traces: Sequence[np.ndarray],
    ar1_params_dir: str,
) -> Dict[str, np.ndarray]:
    ar1_path = Path(ar1_params_dir) / f"{config_id}_ar1_params.json"
    k = int(gmm_params["k"])
    if ar1_path.exists():
        payload = _load_json(str(ar1_path))
        phi = np.asarray(payload.get("phi", []), dtype=np.float64).reshape(-1)
        sigma_innov = np.asarray(payload.get("sigma_innov", []), dtype=np.float64).reshape(-1)
        sigma_marginal = np.asarray(payload.get("sigma_marginal", []), dtype=np.float64).reshape(-1)
        if phi.size == k and sigma_innov.size == k and sigma_marginal.size == k:
            return {
                "phi": phi,
                "sigma_innov": sigma_innov,
                "sigma_marginal": sigma_marginal,
                "phi_threshold": float(payload.get("phi_threshold", 0.3)),
            }

    train_labels = [
        predict_sorted_gmm_labels_from_params(trace, gmm_params).astype(np.int64)
        for trace in train_power_traces
    ]
    phi, sigma_innov, sigma_marginal = estimate_ar1_params(
        gmm_params=gmm_params,
        training_power_traces=train_power_traces,
        training_labels_traces=train_labels,
        K=k,
    )
    return {
        "phi": np.asarray(phi, dtype=np.float64).reshape(-1),
        "sigma_innov": np.asarray(sigma_innov, dtype=np.float64).reshape(-1),
        "sigma_marginal": np.asarray(sigma_marginal, dtype=np.float64).reshape(-1),
        "phi_threshold": 0.3,
    }


def _make_failed_row(
    *,
    config_id: str,
    method: str,
    reason: str,
    dt: float = float("nan"),
    feature_set: str = "",
    k: int = -1,
    decode_mode: str = "",
    median_filter_window: int = 0,
    ours_std_scale: float = 1.0,
    ours_logit_temperature: float = 1.0,
    splitwise_source_model: str = "",
    splitwise_source_hardware: str = "",
    splitwise_source_tp: int = 4,
    splitwise_calibration_mode: str = "train_phase_matched_v1",
    splitwise_phase_detection_note: str = "",
    splitwise_decode_occupancy_note: str = "",
) -> Dict[str, object]:
    return {
        "config_id": config_id,
        "method": method,
        "status": "failed",
        "reason": reason,
        "num_test_traces": 0,
        "num_eval_traces": 0,
        "num_failed_traces": 0,
        "num_seeds": 0,
        "ks_stat": float("nan"),
        "acf_r2": float("nan"),
        "nrmse": float("nan"),
        "p95_error_pct": float("nan"),
        "p99_error_pct": float("nan"),
        "delta_energy_pct": float("nan"),
        "acf_note": "N/A_constant_trace" if method in CONSTANT_METHODS else "",
        "dt": dt,
        "feature_set": feature_set,
        "k": k,
        "decode_mode": decode_mode,
        "median_filter_window": median_filter_window,
        "ours_std_scale": float(ours_std_scale),
        "ours_logit_temperature": float(ours_logit_temperature),
        "splitwise_source_model": str(splitwise_source_model),
        "splitwise_source_hardware": str(splitwise_source_hardware),
        "splitwise_source_tp": int(splitwise_source_tp),
        "splitwise_calibration_mode": str(splitwise_calibration_mode),
        "splitwise_phase_detection_note": str(splitwise_phase_detection_note),
        "splitwise_decode_occupancy_note": str(splitwise_decode_occupancy_note),
    }


def run_baselines_node(
    *,
    run_manifest: str = "results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1/manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    ar1_params_dir: str = "results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/ar1_params",
    out_csv: str = "results/eval_paper/baselines_node_level.csv",
    config_ids: Optional[Sequence[str]] = None,
    num_seeds: int = 5,
    base_seed: int = 42,
    device: str = "auto",
    acf_max_lag: int = 50,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    ours_std_scale: float = 1.0,
    ours_logit_temperature: float = 1.0,
    splitwise_perf_model_csv: str = "data/perf_model.csv",
    splitwise_source_model: str = "llama2-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: int = 4,
    splitwise_calibration_mode: str = "train_phase_matched_v1",
) -> Dict[str, object]:
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    if float(ours_std_scale) <= 0:
        raise ValueError("ours_std_scale must be > 0")
    if float(ours_logit_temperature) <= 0:
        raise ValueError("ours_logit_temperature must be > 0")
    if int(splitwise_source_tp) != 4:
        raise ValueError("splitwise_source_tp must be 4 for 70B TP4-only comparison.")

    run_manifest_payload = _load_json(run_manifest)
    run_cfgs = run_manifest_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    run_manifest_base = str(Path(run_manifest).resolve().parent)

    experimental_payload = _load_json(experimental_manifest)
    experimental_base = str(Path(experimental_manifest).resolve().parent)
    throughput_payload = _load_json(throughput_db)
    pair_map = _load_pair_manifest_map(pair_manifest_csv)
    parsed_targets = _parse_config_ids(config_ids) or list(DEFAULT_CONFIG_IDS)
    targets = [cid for cid in parsed_targets if _is_70b_tp4_config(cid)]
    if len(targets) == 0:
        raise ValueError("No valid 70B TP4 configs selected for Splitwise baseline comparison.")
    resolved_device = _resolve_device(device)

    rows: List[Dict[str, object]] = []
    for config_id in targets:
        method_trace_metrics: Dict[str, List[Dict[str, float]]] = {method: [] for method in METHODS}
        method_errors: Dict[str, List[str]] = {method: [] for method in METHODS}
        per_method_num_eval_seeds: Dict[str, int] = {
            method: (int(num_seeds) if method in STOCHASTIC_METHODS else 1) for method in METHODS
        }
        total_test_traces = 0
        dt = float("nan")
        feature_set = ""
        k = -1
        splitwise_phase_detection_note = ""
        splitwise_decode_occupancy_note = ""

        try:
            cfg_entry = run_cfgs.get(config_id)
            if not isinstance(cfg_entry, dict):
                raise ValueError("config_not_in_run_manifest")
            if str(cfg_entry.get("status", "")) != "trained":
                raise ValueError(f"config_status_{cfg_entry.get('status', 'unknown')}")

            checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(cfg_entry, run_manifest_base)
            norm_payload = _load_json(norm_path)
            norm_cfg = _extract_norm_for_eval(norm_payload)
            gmm_payload = _load_json(gmm_path)
            gmm_cfg = load_gmm_params_json_dict(gmm_payload)
            throughput = _resolve_throughput(throughput_payload, config_id)

            feature_set = str(cfg_entry.get("feature_set", norm_payload.get("feature_set", "f2"))).lower()
            if feature_set not in {"f2", "f3"}:
                raise ValueError(f"invalid feature_set: {feature_set}")
            k = int(cfg_entry.get("k", gmm_cfg["k"]))
            if k != int(gmm_cfg["k"]):
                raise ValueError(f"k mismatch: manifest={k}, gmm={int(gmm_cfg['k'])}")

            input_dim = int(cfg_entry.get("input_dim", 2 if feature_set == "f2" else 3))
            hidden_dim = int(cfg_entry.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
            num_layers = int(cfg_entry.get("num_layers", norm_payload.get("num_layers", 1)))
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
            test_indices = [int(x) for x in split_payload.get("test_indices", [])]
            if len(test_indices) == 0:
                raise ValueError("empty_test_split")
            total_test_traces = int(len(test_indices))

            with np.load(dataset_path, allow_pickle=True) as data:
                pair_key_arr = np.asarray(data["pair_key"], dtype=object)
                power_arr = np.asarray(data["power"], dtype=object)
                power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
                rate_arr = np.asarray(data["rate"], dtype=object) if "rate" in data else np.asarray([], dtype=object)
                dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)
            if dt_arr.size == 0:
                raise ValueError("dataset_dt_missing")
            dt = float(dt_arr[0])
            if (not np.isfinite(dt)) or dt <= 0.0:
                raise ValueError(f"invalid_dt:{dt}")
            n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))

            train_power_traces: List[np.ndarray] = []
            train_power_pool: List[np.ndarray] = []
            for idx in train_indices:
                if idx < 0 or idx >= n_total:
                    continue
                p = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
                if p.size == 0:
                    continue
                train_power_traces.append(p.astype(np.float64))
                train_power_pool.append(p.astype(np.float64))
            if len(train_power_pool) == 0:
                for i in range(n_total):
                    p = np.asarray(power_arr[i], dtype=np.float64).reshape(-1)
                    if p.size == 0:
                        continue
                    train_power_traces.append(p.astype(np.float64))
                    train_power_pool.append(p.astype(np.float64))
                    if len(train_power_pool) >= 3:
                        break
            if len(train_power_pool) == 0:
                raise ValueError("empty_training_pool")
            train_power_flat = np.concatenate(train_power_pool, axis=0).astype(np.float64)
            train_power_flat_gpu = np.clip(train_power_flat - 1000.0, a_min=0.0, a_max=None)
            phase_targets = _estimate_splitwise_phase_targets_from_indices(
                indices=train_indices,
                pair_key_arr=pair_key_arr,
                power_arr=power_arr,
                power_start_arr=power_start_arr,
                pair_map=pair_map,
                throughput=throughput,
                norm_cfg=norm_cfg,
                feature_set=feature_set,
                dt=float(dt),
                non_gpu_overhead_w=1000.0,
            )
            splitwise_lut_params = build_splitwise_lut_params(
                config_id=config_id,
                perf_model_csv=splitwise_perf_model_csv,
                train_power_flat=train_power_flat_gpu,
                splitwise_source_model=splitwise_source_model,
                splitwise_source_hardware=splitwise_source_hardware,
                splitwise_source_tp=int(splitwise_source_tp),
                splitwise_calibration_mode=splitwise_calibration_mode,
                n_gpus_per_node=8,
                target_idle_node_gpu_w=phase_targets.get("target_idle_node_gpu_w"),
                target_decode_node_gpu_w=phase_targets.get("target_decode_node_gpu_w"),
                target_prefill_node_gpu_w=phase_targets.get("target_prefill_node_gpu_w"),
            )
            splitwise_phase_detection_note = str(splitwise_lut_params.get("phase_detection_note", ""))
            splitwise_decode_occupancy_note = str(splitwise_lut_params.get("decode_occupancy_note", ""))

            ar1_params = None
            if _is_moe_config(config_id):
                ar1_params = _load_or_estimate_ar1_params(
                    config_id=config_id,
                    gmm_params=gmm_cfg,
                    train_power_traces=train_power_traces
                    if len(train_power_traces) > 0
                    else train_power_pool,
                    ar1_params_dir=ar1_params_dir,
                )

            for test_idx in test_indices:
                if test_idx < 0 or test_idx >= n_total:
                    for method in METHODS:
                        method_errors[method].append(f"trace_idx={test_idx}:out_of_bounds")
                    continue

                power = np.asarray(power_arr[test_idx], dtype=np.float64).reshape(-1)
                if power.size < 2:
                    for method in METHODS:
                        method_errors[method].append(f"trace_idx={test_idx}:trace_too_short")
                    continue
                pair_key = str(pair_key_arr[test_idx])
                json_path = pair_map.get(pair_key)
                if json_path is None:
                    for method in METHODS:
                        method_errors[method].append(f"trace_idx={test_idx}:missing_pair_key")
                    continue

                p0 = float(power[0])
                gt = power[1:].astype(np.float64)
                requests = _build_requests_from_stage0_json(
                    json_path,
                    power_start_epoch_s=float(power_start_arr[test_idx]),
                    trace_duration_s=float(power.size * dt),
                    dt=dt,
                )
                feat = build_rollout_features_from_requests(
                    requests=requests,
                    throughput=throughput,
                    norm=norm_cfg,
                    T=int(gt.size),
                    dt=dt,
                    feature_set=feature_set,
                )
                features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
                if features_norm.ndim != 2:
                    for method in METHODS:
                        method_errors[method].append(f"trace_idx={test_idx}:invalid_feature_shape")
                    continue

                n_eval = int(min(gt.size, features_norm.shape[0]))
                if n_eval <= 0:
                    for method in METHODS:
                        method_errors[method].append(f"trace_idx={test_idx}:empty_aligned_horizon")
                    continue

                gt_eval = gt[:n_eval]
                feat_eval = features_norm[:n_eval]
                a_raw_eval = np.asarray(feat["A_raw"], dtype=np.float64).reshape(-1)[:n_eval]
                delta_a_raw_eval = np.asarray(feat["delta_A_raw"], dtype=np.float64).reshape(-1)[:n_eval]
                tdp_cfg = {"config_id": config_id}
                ours_cfg = {
                    "device": str(resolved_device),
                    "p0": p0,
                    "decode_mode": decode_mode,
                    "median_filter_window": int(median_filter_window),
                    "std_scale": float(ours_std_scale),
                    "logit_temperature": float(ours_logit_temperature),
                    "clamp_range": (norm_cfg["power_min"], norm_cfg["power_max"]),
                    "ar1_params": ar1_params,
                }

                for method in METHODS:
                    seeds = [int(base_seed)] if method not in STOCHASTIC_METHODS else [
                        int(base_seed) + i for i in range(int(num_seeds))
                    ]
                    seed_metric_rows: List[Dict[str, float]] = []
                    for seed in seeds:
                        try:
                            if method == "tdp":
                                pred = generate_tdp(n_eval, tdp_cfg)
                            elif method == "mean":
                                pred = generate_mean(n_eval, {}, train_power_flat)
                            elif method == "splitwise_lut":
                                pred = generate_splitwise_lut(
                                    a_raw_eval,
                                    delta_a_raw_eval,
                                    {
                                        "config_id": config_id,
                                        "tp": 4,
                                        "n_gpus_per_node": 8,
                                        "non_gpu_power_w": 1000.0,
                                    },
                                    splitwise_lut_params,
                                )
                            elif method == "ours":
                                pred = generate_ours(
                                    feat_eval,
                                    ours_cfg,
                                    model,
                                    gmm_cfg,
                                    rng=np.random.default_rng(seed),
                                )
                            else:
                                raise ValueError(f"Unknown method: {method}")

                            metrics = compute_power_metrics(
                                gt_eval,
                                np.asarray(pred, dtype=np.float64).reshape(-1),
                                dt=dt,
                                acf_max_lag=int(acf_max_lag),
                            )
                            if method in CONSTANT_METHODS:
                                metrics["acf_r2"] = float("nan")
                            seed_metric_rows.append({k: float(metrics[k]) for k in METRIC_KEYS})
                        except Exception as exc:
                            method_errors[method].append(f"trace_idx={test_idx}:seed={seed}:{type(exc).__name__}:{exc}")

                    if len(seed_metric_rows) == 0:
                        continue
                    trace_summary = {
                        key: _nanmedian(x[key] for x in seed_metric_rows)
                        for key in METRIC_KEYS
                    }
                    if method in CONSTANT_METHODS:
                        trace_summary["acf_r2"] = float("nan")
                    method_trace_metrics[method].append(trace_summary)
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            for method in METHODS:
                rows.append(
                    _make_failed_row(
                        config_id=config_id,
                        method=method,
                        reason=reason,
                        dt=dt,
                        feature_set=feature_set,
                        k=k,
                        decode_mode=decode_mode,
                        median_filter_window=int(median_filter_window),
                        ours_std_scale=float(ours_std_scale),
                        ours_logit_temperature=float(ours_logit_temperature),
                        splitwise_source_model=splitwise_source_model,
                        splitwise_source_hardware=splitwise_source_hardware,
                        splitwise_source_tp=int(splitwise_source_tp),
                        splitwise_calibration_mode=splitwise_calibration_mode,
                        splitwise_phase_detection_note=splitwise_phase_detection_note,
                        splitwise_decode_occupancy_note=splitwise_decode_occupancy_note,
                    )
                )
            continue

        for method in METHODS:
            metric_rows = method_trace_metrics[method]
            if len(metric_rows) == 0:
                reason = method_errors[method][0] if method_errors[method] else "no_valid_trace_metrics"
                rows.append(
                    _make_failed_row(
                        config_id=config_id,
                        method=method,
                        reason=reason,
                        dt=dt,
                        feature_set=feature_set,
                        k=k,
                        decode_mode=decode_mode if method == "ours" else "",
                        median_filter_window=int(median_filter_window) if method == "ours" else 0,
                        ours_std_scale=float(ours_std_scale) if method == "ours" else 1.0,
                        ours_logit_temperature=float(ours_logit_temperature) if method == "ours" else 1.0,
                        splitwise_source_model=splitwise_source_model,
                        splitwise_source_hardware=splitwise_source_hardware,
                        splitwise_source_tp=int(splitwise_source_tp),
                        splitwise_calibration_mode=splitwise_calibration_mode,
                        splitwise_phase_detection_note=splitwise_phase_detection_note,
                        splitwise_decode_occupancy_note=splitwise_decode_occupancy_note,
                    )
                )
                continue

            aggregated = {key: _nanmedian(r[key] for r in metric_rows) for key in METRIC_KEYS}
            if method in CONSTANT_METHODS:
                aggregated["acf_r2"] = float("nan")
            rows.append(
                {
                    "config_id": config_id,
                    "method": method,
                    "status": "evaluated",
                    "reason": "",
                    "num_test_traces": int(total_test_traces),
                    "num_eval_traces": int(len(metric_rows)),
                    "num_failed_traces": int(max(0, total_test_traces - len(metric_rows))),
                    "num_seeds": int(per_method_num_eval_seeds[method]),
                    "ks_stat": float(aggregated["ks_stat"]),
                    "acf_r2": float(aggregated["acf_r2"]),
                    "nrmse": float(aggregated["nrmse"]),
                    "p95_error_pct": float(aggregated["p95_error_pct"]),
                    "p99_error_pct": float(aggregated["p99_error_pct"]),
                    "delta_energy_pct": float(aggregated["delta_energy_pct"]),
                    "acf_note": "N/A_constant_trace" if method in CONSTANT_METHODS else "",
                    "dt": float(dt),
                    "feature_set": feature_set,
                    "k": int(k),
                    "decode_mode": decode_mode if method == "ours" else "",
                    "median_filter_window": int(median_filter_window) if method == "ours" else 0,
                    "ours_std_scale": float(ours_std_scale) if method == "ours" else 1.0,
                    "ours_logit_temperature": float(ours_logit_temperature) if method == "ours" else 1.0,
                    "splitwise_source_model": splitwise_source_model,
                    "splitwise_source_hardware": splitwise_source_hardware,
                    "splitwise_source_tp": int(splitwise_source_tp),
                    "splitwise_calibration_mode": splitwise_calibration_mode,
                    "splitwise_phase_detection_note": splitwise_phase_detection_note,
                    "splitwise_decode_occupancy_note": splitwise_decode_occupancy_note,
                }
            )

    fieldnames = [
        "config_id",
        "method",
        "status",
        "reason",
        "num_test_traces",
        "num_eval_traces",
        "num_failed_traces",
        "num_seeds",
        "ks_stat",
        "acf_r2",
        "nrmse",
        "p95_error_pct",
        "p99_error_pct",
        "delta_energy_pct",
        "acf_note",
        "dt",
        "feature_set",
        "k",
        "decode_mode",
        "median_filter_window",
        "ours_std_scale",
        "ours_logit_temperature",
        "splitwise_source_model",
        "splitwise_source_hardware",
        "splitwise_source_tp",
        "splitwise_calibration_mode",
        "splitwise_phase_detection_note",
        "splitwise_decode_occupancy_note",
    ]
    _write_csv(out_csv, rows, fieldnames)
    return {
        "rows": rows,
        "out_csv": out_csv,
        "num_configs": len(set(str(r["config_id"]) for r in rows)),
        "num_rows": len(rows),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Node-level baseline comparison (TDP / Mean / Splitwise LUT / Ours).")
    parser.add_argument("--run-manifest", default="results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json")
    parser.add_argument("--experimental-manifest", default="results/experimental_continuous_v1/manifest.json")
    parser.add_argument("--throughput-db", default="model/config/throughput_database.json")
    parser.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    parser.add_argument(
        "--ar1-params-dir",
        default="results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/ar1_params",
        help="Directory containing AR(1) params JSON files (used only for MoE configs).",
    )
    parser.add_argument("--out-csv", default="results/eval_paper/baselines_node_level.csv")
    parser.add_argument("--config-ids", nargs="*", default=None, help="Optional list or comma-separated list of config IDs")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--acf-max-lag", type=int, default=50)
    parser.add_argument("--decode-mode", default="stochastic", choices=["stochastic", "argmax"])
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--ours-std-scale", type=float, default=1.0)
    parser.add_argument("--ours-logit-temperature", type=float, default=1.0)
    parser.add_argument("--splitwise-perf-model-csv", default="data/perf_model.csv")
    parser.add_argument("--splitwise-source-model", default="llama2-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=4)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_baselines_node(
        run_manifest=args.run_manifest,
        experimental_manifest=args.experimental_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        ar1_params_dir=args.ar1_params_dir,
        out_csv=args.out_csv,
        config_ids=args.config_ids,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
        device=args.device,
        acf_max_lag=args.acf_max_lag,
        decode_mode=args.decode_mode,
        median_filter_window=args.median_filter_window,
        ours_std_scale=args.ours_std_scale,
        ours_logit_temperature=args.ours_logit_temperature,
        splitwise_perf_model_csv=args.splitwise_perf_model_csv,
        splitwise_source_model=args.splitwise_source_model,
        splitwise_source_hardware=args.splitwise_source_hardware,
        splitwise_source_tp=args.splitwise_source_tp,
    )
    print("[run_baselines_node] Done")
    print(f"  rows     : {result['num_rows']}")
    print(f"  configs  : {result['num_configs']}")
    print(f"  out_csv  : {result['out_csv']}")


if __name__ == "__main__":
    main()
