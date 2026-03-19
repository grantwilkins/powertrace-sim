#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from model.classifiers.metrics import compute_aggregate_power_metrics
from model.utils.io import ensure_dir, load_json
from scripts.eval.pipeline_utils import (
    build_rollout_features_from_requests,
    estimate_ar1_params,
    extract_norm_params,
    load_gmm_params_json_dict,
    load_gru_classifier,
    predict_sorted_gmm_labels_from_params,
    resolve_checkpoint_norm_gmm_paths as _shared_resolve_checkpoint_norm_gmm_paths,
    resolve_experimental_paths as _shared_resolve_experimental_paths,
    resolve_throughput as _shared_resolve_throughput,
)
from scripts.eval.baselines import (
    SPLITWISE_STYLE_LUT_V1,
    build_splitwise_style_lut_params,
    generate_mean,
    generate_ours,
    generate_splitwise_style_lut_trace,
    generate_tdp,
    normalize_splitwise_style_lut_mode,
)

METHODS = ("tdp", "mean", "splitwise_strict", "ours")
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
    "llama-3-70b_A100_tp8",
    "llama-3-70b_H100_tp4",
]
DEFAULT_PER_GPU_CHIP_TDP_W = {
    "A100": 400.0,
    "H100": 700.0,
}
CONFIG_70B_TP4_RE = re.compile(r"^.+-70b_(A100|H100)_tp4$")
CONFIG_MODEL_SIZE_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")

# Backward-compatible alias used by run_baselines_node_groundtruth imports.
_ensure_dir = ensure_dir


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
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


def _parse_tp_from_config_id(config_id: str) -> int:
    match = CONFIG_MODEL_SIZE_RE.match(str(config_id).strip())
    if match is None:
        raise ValueError(f"Unable to parse TP from config_id: {config_id}")
    tp = int(match.group(4))
    if tp <= 0:
        raise ValueError(f"Invalid TP in config_id: {config_id}")
    return tp


def _parse_hardware_from_config_id(config_id: str) -> str:
    match = CONFIG_MODEL_SIZE_RE.match(str(config_id).strip())
    if match is None:
        raise ValueError(f"Unable to parse hardware from config_id: {config_id}")
    hardware = str(match.group(3)).upper()
    if hardware not in DEFAULT_PER_GPU_CHIP_TDP_W:
        raise ValueError(f"Unsupported hardware in config_id: {config_id}")
    return hardware


def _resolve_per_gpu_chip_tdp_w(
    config_id: str, gpu_tdp_w: Optional[float]
) -> float:
    if gpu_tdp_w is not None:
        resolved = float(gpu_tdp_w)
    else:
        resolved = float(
            DEFAULT_PER_GPU_CHIP_TDP_W[_parse_hardware_from_config_id(config_id)]
        )
    if (not np.isfinite(resolved)) or resolved <= 0.0:
        raise ValueError("gpu_tdp_w must be > 0")
    return resolved


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


def _aggregate_heldout_metrics_by_seed(
    *,
    traces_by_seed: Mapping[int, Sequence[Tuple[int, np.ndarray, np.ndarray]]],
    dt: float,
    acf_max_lag: int,
    force_nan_acf: bool = False,
) -> Tuple[Optional[Dict[str, float]], int, int]:
    seed_metric_rows: List[Dict[str, float]] = []
    trace_ids_used: set[int] = set()

    for seed in sorted(int(x) for x in traces_by_seed.keys()):
        gt_traces: List[np.ndarray] = []
        pred_traces: List[np.ndarray] = []
        for trace_idx, gt_trace, pred_trace in traces_by_seed.get(seed, []):
            gt = np.asarray(gt_trace, dtype=np.float64).reshape(-1)
            pred = np.asarray(pred_trace, dtype=np.float64).reshape(-1)
            n = int(min(gt.size, pred.size))
            if n <= 0:
                continue
            gt_traces.append(gt[:n].astype(np.float64))
            pred_traces.append(pred[:n].astype(np.float64))
            trace_ids_used.add(int(trace_idx))

        if len(gt_traces) == 0:
            continue

        metrics = compute_aggregate_power_metrics(
            gt_traces,
            pred_traces,
            dt=float(dt),
            acf_max_lag=int(acf_max_lag),
        )
        if bool(force_nan_acf):
            metrics["acf_r2"] = float("nan")
        seed_metric_rows.append({k: float(metrics[k]) for k in METRIC_KEYS})

    if len(seed_metric_rows) == 0:
        return None, 0, 0

    aggregated = {
        key: _nanmedian(row[key] for row in seed_metric_rows) for key in METRIC_KEYS
    }
    if bool(force_nan_acf):
        aggregated["acf_r2"] = float("nan")
    return aggregated, int(len(trace_ids_used)), int(len(seed_metric_rows))


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
    require_recorded_timestamps: bool = True,
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
                require_recorded_timestamps=bool(require_recorded_timestamps),
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
        gpu_power = gt_node[:n_eval].astype(np.float64)
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
    require_recorded_timestamps: bool = True,
) -> List[Dict[str, float]]:
    payload = load_json(request_json_path)
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
        if bool(require_recorded_timestamps):
            raise ValueError(
                "request json missing arrays: ['request_timestamps'] "
                "(synthetic fallback disabled)"
            )
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


_extract_norm_for_eval = extract_norm_params


_resolve_checkpoint_norm_gmm_paths = _shared_resolve_checkpoint_norm_gmm_paths
_resolve_throughput = _shared_resolve_throughput
_resolve_experimental_paths = _shared_resolve_experimental_paths


_load_model = load_gru_classifier


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
        payload = load_json(str(ar1_path))
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
    splitwise_style_lut_mode: str = SPLITWISE_STYLE_LUT_V1,
    splitwise_phase_detection_note: str = "",
    splitwise_decode_occupancy_note: str = "",
    splitwise_source_resolved_model: str = "",
    splitwise_source_resolved_hardware: str = "",
    splitwise_source_resolved_tp: int = 0,
    splitwise_source_match_status: str = "",
    splitwise_power_quality_flag: str = "",
    splitwise_power_support_status: str = "",
    splitwise_scheduler_policy: str = "",
    splitwise_extrapolation_events: int = 0,
    splitwise_power_clamp_events: int = 0,
    splitwise_max_batch_tokens_seen: float = float("nan"),
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
        "splitwise_style_lut_mode": str(splitwise_style_lut_mode),
        "splitwise_phase_detection_note": str(splitwise_phase_detection_note),
        "splitwise_decode_occupancy_note": str(splitwise_decode_occupancy_note),
        "splitwise_source_resolved_model": str(splitwise_source_resolved_model),
        "splitwise_source_resolved_hardware": str(splitwise_source_resolved_hardware),
        "splitwise_source_resolved_tp": int(splitwise_source_resolved_tp),
        "splitwise_source_match_status": str(splitwise_source_match_status),
        "splitwise_power_quality_flag": str(splitwise_power_quality_flag),
        "splitwise_power_support_status": str(splitwise_power_support_status),
        "splitwise_scheduler_policy": str(splitwise_scheduler_policy),
        "splitwise_extrapolation_events": int(splitwise_extrapolation_events),
        "splitwise_power_clamp_events": int(splitwise_power_clamp_events),
        "splitwise_max_batch_tokens_seen": float(splitwise_max_batch_tokens_seen),
        "metric_aggregation_scope": "heldout_all_traces_per_seed",
        "seed_aggregation_mode": "median_over_seed_heldout_metrics",
        "delta_energy_definition": "absolute_total_trace_energy_pct",
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
    splitwise_source_model: str = "llama-3-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: Optional[int] = None,
    splitwise_style_lut_mode: str = SPLITWISE_STYLE_LUT_V1,
    allow_synthetic_request_timestamps: bool = False,
) -> Dict[str, object]:
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    if float(ours_std_scale) <= 0:
        raise ValueError("ours_std_scale must be > 0")
    if float(ours_logit_temperature) <= 0:
        raise ValueError("ours_logit_temperature must be > 0")
    splitwise_style_lut_mode = normalize_splitwise_style_lut_mode(
        splitwise_style_lut_mode
    )
    run_manifest_payload = load_json(run_manifest)
    run_cfgs = run_manifest_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    run_manifest_base = str(Path(run_manifest).resolve().parent)

    experimental_payload = load_json(experimental_manifest)
    experimental_base = str(Path(experimental_manifest).resolve().parent)
    throughput_payload = load_json(throughput_db)
    pair_map = _load_pair_manifest_map(pair_manifest_csv)
    parsed_targets = _parse_config_ids(config_ids) or list(DEFAULT_CONFIG_IDS)
    targets = list(dict.fromkeys(parsed_targets))
    if len(targets) == 0:
        raise ValueError("No configs selected for node-level baseline comparison.")
    resolved_device = _resolve_device(device)

    rows: List[Dict[str, object]] = []
    for config_id in targets:
        method_seed_trace_pairs: Dict[
            str, Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]]
        ] = {method: {} for method in METHODS}
        method_errors: Dict[str, List[str]] = {method: [] for method in METHODS}
        splitwise_method_meta: Dict[str, Dict[str, object]] = {}
        splitwise_generation_meta: Dict[str, Dict[str, object]] = {}
        total_test_traces = 0
        dt = float("nan")
        feature_set = ""
        k = -1
        target_tp = 4
        requested_splitwise_tp = int(splitwise_source_tp) if splitwise_source_tp is not None else int(
            max(1, _parse_tp_from_config_id(config_id))
        )

        def _default_splitwise_meta() -> Dict[str, object]:
            return {
                "splitwise_source_model": str(splitwise_source_model),
                "splitwise_source_hardware": str(splitwise_source_hardware),
                "splitwise_source_tp": int(requested_splitwise_tp),
                "splitwise_style_lut_mode": "",
                "splitwise_phase_detection_note": "",
                "splitwise_decode_occupancy_note": "",
                "splitwise_source_resolved_model": "",
                "splitwise_source_resolved_hardware": "",
                "splitwise_source_resolved_tp": 0,
                "splitwise_source_match_status": "",
                "splitwise_power_quality_flag": "",
                "splitwise_power_support_status": "",
                "splitwise_scheduler_policy": "",
                "splitwise_extrapolation_events": 0,
                "splitwise_power_clamp_events": 0,
                "splitwise_max_batch_tokens_seen": float("nan"),
            }

        def _splitwise_meta_for_method(method: str) -> Dict[str, object]:
            if str(method) != "splitwise_strict":
                return _default_splitwise_meta()
            base = _default_splitwise_meta()
            base.update(splitwise_method_meta.get(str(method), {}))
            gen_meta = splitwise_generation_meta.get(str(method), {})
            if gen_meta:
                base["splitwise_extrapolation_events"] = int(
                    gen_meta.get("splitwise_extrapolation_events", base["splitwise_extrapolation_events"])
                )
                base["splitwise_power_clamp_events"] = int(
                    gen_meta.get("splitwise_power_clamp_events", base["splitwise_power_clamp_events"])
                )
                base["splitwise_max_batch_tokens_seen"] = float(
                    gen_meta.get(
                        "splitwise_max_batch_tokens_seen",
                        base["splitwise_max_batch_tokens_seen"],
                    )
                )
                power_support_status = str(
                    gen_meta.get("splitwise_power_support_status", base["splitwise_power_support_status"])
                )
                if power_support_status:
                    base["splitwise_power_support_status"] = power_support_status
            return base

        try:
            cfg_entry = run_cfgs.get(config_id)
            if not isinstance(cfg_entry, dict):
                raise ValueError("config_not_in_run_manifest")
            if str(cfg_entry.get("status", "")) != "trained":
                raise ValueError(f"config_status_{cfg_entry.get('status', 'unknown')}")

            checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(cfg_entry, run_manifest_base)
            norm_payload = load_json(norm_path)
            norm_cfg = _extract_norm_for_eval(norm_payload)
            gmm_payload = load_json(gmm_path)
            gmm_cfg = load_gmm_params_json_dict(gmm_payload)
            throughput = _resolve_throughput(throughput_payload, config_id)

            feature_set = str(cfg_entry.get("feature_set", norm_payload.get("feature_set", "f2"))).lower()
            if feature_set == "f3":
                raise ValueError("feature_set='f3' is no longer supported; use 'f2'.")
            if feature_set != "f2":
                raise ValueError(f"invalid feature_set: {feature_set}")
            k = int(cfg_entry.get("k", gmm_cfg["k"]))
            if k != int(gmm_cfg["k"]):
                raise ValueError(f"k mismatch: manifest={k}, gmm={int(gmm_cfg['k'])}")
            target_tp = int(max(1, _parse_tp_from_config_id(config_id)))

            input_dim = int(cfg_entry.get("input_dim", 2))
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
            split_payload = load_json(split_path)
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
            train_power_flat = np.concatenate(train_power_pool, axis=0).astype(
                np.float64
            )
            train_power_flat_gpu = train_power_flat.copy()
            splitwise_strict_lut_params = build_splitwise_style_lut_params(
                config_id=config_id,
                perf_model_csv=splitwise_perf_model_csv,
                train_power_flat=train_power_flat_gpu,
                splitwise_source_model=splitwise_source_model,
                splitwise_source_hardware=splitwise_source_hardware,
                splitwise_source_tp=int(requested_splitwise_tp),
                splitwise_style_lut_mode=splitwise_style_lut_mode,
                n_gpus_per_node=8,
            )
            splitwise_method_meta = {
                "splitwise_strict": {
                    "splitwise_source_model": str(splitwise_source_model),
                    "splitwise_source_hardware": str(splitwise_source_hardware),
                    "splitwise_source_tp": int(requested_splitwise_tp),
                    "splitwise_style_lut_mode": str(
                        splitwise_strict_lut_params.get(
                            "splitwise_style_lut_mode", SPLITWISE_STYLE_LUT_V1
                        )
                    ),
                    "splitwise_phase_detection_note": str(
                        splitwise_strict_lut_params.get("phase_detection_note", "")
                    ),
                    "splitwise_decode_occupancy_note": str(
                        splitwise_strict_lut_params.get("decode_occupancy_note", "")
                    ),
                    "splitwise_source_resolved_model": str(
                        splitwise_strict_lut_params.get("splitwise_source_resolved_model", "")
                    ),
                    "splitwise_source_resolved_hardware": str(
                        splitwise_strict_lut_params.get("splitwise_source_resolved_hardware", "")
                    ),
                    "splitwise_source_resolved_tp": int(
                        splitwise_strict_lut_params.get("splitwise_source_resolved_tp", 0)
                    ),
                    "splitwise_source_match_status": str(
                        splitwise_strict_lut_params.get("splitwise_source_match_status", "")
                    ),
                    "splitwise_power_quality_flag": str(
                        splitwise_strict_lut_params.get("splitwise_power_quality_flag", "")
                    ),
                    "splitwise_power_support_status": str(
                        splitwise_strict_lut_params.get("splitwise_power_support_status", "")
                    ),
                    "splitwise_scheduler_policy": str(
                        splitwise_strict_lut_params.get("splitwise_scheduler_policy", "")
                    ),
                },
            }
            splitwise_generation_meta["splitwise_strict"] = {
                "splitwise_extrapolation_events": 0,
                "splitwise_power_clamp_events": 0,
                "splitwise_max_batch_tokens_seen": 0.0,
                "splitwise_power_support_status": str(
                    splitwise_strict_lut_params.get("splitwise_power_support_status", "")
                ),
            }

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
                    require_recorded_timestamps=not bool(
                        allow_synthetic_request_timestamps
                    ),
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
                tdp_cfg = {"config_id": config_id, "non_gpu_power_w": 0.0}
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
                    for seed in seeds:
                        try:
                            if method == "tdp":
                                pred = generate_tdp(n_eval, tdp_cfg)
                            elif method == "mean":
                                pred = generate_mean(n_eval, {}, train_power_flat)
                            elif method == "splitwise_strict":
                                pred, strict_meta = generate_splitwise_style_lut_trace(
                                    requests=requests,
                                    T=n_eval,
                                    dt=dt,
                                    config={
                                        "config_id": config_id,
                                        "tp": int(target_tp),
                                        "n_gpus_per_node": 8,
                                        "non_gpu_power_w": 0.0,
                                    },
                                    lut_params=splitwise_strict_lut_params,
                                )
                                splitwise_generation_meta["splitwise_strict"][
                                    "splitwise_extrapolation_events"
                                ] = int(
                                    splitwise_generation_meta["splitwise_strict"].get(
                                        "splitwise_extrapolation_events", 0
                                    )
                                ) + int(strict_meta.get("splitwise_extrapolation_events", 0))
                                splitwise_generation_meta["splitwise_strict"][
                                    "splitwise_power_clamp_events"
                                ] = int(
                                    splitwise_generation_meta["splitwise_strict"].get(
                                        "splitwise_power_clamp_events", 0
                                    )
                                ) + int(strict_meta.get("splitwise_power_clamp_events", 0))
                                splitwise_generation_meta["splitwise_strict"][
                                    "splitwise_max_batch_tokens_seen"
                                ] = float(
                                    max(
                                        float(
                                            splitwise_generation_meta["splitwise_strict"].get(
                                                "splitwise_max_batch_tokens_seen", 0.0
                                            )
                                        ),
                                        float(
                                            strict_meta.get(
                                                "splitwise_max_batch_tokens_seen", 0.0
                                            )
                                        ),
                                    )
                                )
                                splitwise_generation_meta["splitwise_strict"][
                                    "splitwise_power_support_status"
                                ] = str(
                                    strict_meta.get(
                                        "splitwise_power_support_status",
                                        splitwise_generation_meta["splitwise_strict"].get(
                                            "splitwise_power_support_status", ""
                                        ),
                                    )
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

                            pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
                            method_seed_trace_pairs[method].setdefault(int(seed), []).append(
                                (
                                    int(test_idx),
                                    gt_eval.astype(np.float64),
                                    pred_arr.astype(np.float64),
                                )
                            )
                        except Exception as exc:
                            method_errors[method].append(f"trace_idx={test_idx}:seed={seed}:{type(exc).__name__}:{exc}")
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            for method in METHODS:
                method_meta = _splitwise_meta_for_method(method)
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
                        splitwise_source_model=str(method_meta["splitwise_source_model"]),
                        splitwise_source_hardware=str(method_meta["splitwise_source_hardware"]),
                        splitwise_source_tp=int(method_meta["splitwise_source_tp"]),
                        splitwise_style_lut_mode=str(method_meta["splitwise_style_lut_mode"]),
                        splitwise_phase_detection_note=str(method_meta["splitwise_phase_detection_note"]),
                        splitwise_decode_occupancy_note=str(method_meta["splitwise_decode_occupancy_note"]),
                        splitwise_source_resolved_model=str(method_meta["splitwise_source_resolved_model"]),
                        splitwise_source_resolved_hardware=str(method_meta["splitwise_source_resolved_hardware"]),
                        splitwise_source_resolved_tp=int(method_meta["splitwise_source_resolved_tp"]),
                        splitwise_source_match_status=str(method_meta["splitwise_source_match_status"]),
                        splitwise_power_quality_flag=str(method_meta["splitwise_power_quality_flag"]),
                        splitwise_power_support_status=str(method_meta["splitwise_power_support_status"]),
                        splitwise_scheduler_policy=str(method_meta["splitwise_scheduler_policy"]),
                        splitwise_extrapolation_events=int(method_meta["splitwise_extrapolation_events"]),
                        splitwise_power_clamp_events=int(method_meta["splitwise_power_clamp_events"]),
                        splitwise_max_batch_tokens_seen=float(method_meta["splitwise_max_batch_tokens_seen"]),
                    )
                )
            continue

        for method in METHODS:
            aggregated, num_eval_traces, num_eval_seeds = (
                _aggregate_heldout_metrics_by_seed(
                    traces_by_seed=method_seed_trace_pairs[method],
                    dt=float(dt),
                    acf_max_lag=int(acf_max_lag),
                    force_nan_acf=bool(method in CONSTANT_METHODS),
                )
            )
            if aggregated is None:
                reason = method_errors[method][0] if method_errors[method] else "no_valid_trace_metrics"
                method_meta = _splitwise_meta_for_method(method)
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
                        splitwise_source_model=str(method_meta["splitwise_source_model"]),
                        splitwise_source_hardware=str(method_meta["splitwise_source_hardware"]),
                        splitwise_source_tp=int(method_meta["splitwise_source_tp"]),
                        splitwise_style_lut_mode=str(method_meta["splitwise_style_lut_mode"]),
                        splitwise_phase_detection_note=str(method_meta["splitwise_phase_detection_note"]),
                        splitwise_decode_occupancy_note=str(method_meta["splitwise_decode_occupancy_note"]),
                        splitwise_source_resolved_model=str(method_meta["splitwise_source_resolved_model"]),
                        splitwise_source_resolved_hardware=str(method_meta["splitwise_source_resolved_hardware"]),
                        splitwise_source_resolved_tp=int(method_meta["splitwise_source_resolved_tp"]),
                        splitwise_source_match_status=str(method_meta["splitwise_source_match_status"]),
                        splitwise_power_quality_flag=str(method_meta["splitwise_power_quality_flag"]),
                        splitwise_power_support_status=str(method_meta["splitwise_power_support_status"]),
                        splitwise_scheduler_policy=str(method_meta["splitwise_scheduler_policy"]),
                        splitwise_extrapolation_events=int(method_meta["splitwise_extrapolation_events"]),
                        splitwise_power_clamp_events=int(method_meta["splitwise_power_clamp_events"]),
                        splitwise_max_batch_tokens_seen=float(method_meta["splitwise_max_batch_tokens_seen"]),
                    )
                )
                continue

            method_splitwise_meta = _splitwise_meta_for_method(method)
            rows.append(
                {
                    "config_id": config_id,
                    "method": method,
                    "status": "evaluated",
                    "reason": "",
                    "num_test_traces": int(total_test_traces),
                    "num_eval_traces": int(num_eval_traces),
                    "num_failed_traces": int(max(0, total_test_traces - num_eval_traces)),
                    "num_seeds": int(num_eval_seeds),
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
                    "splitwise_source_model": str(method_splitwise_meta["splitwise_source_model"]),
                    "splitwise_source_hardware": str(method_splitwise_meta["splitwise_source_hardware"]),
                    "splitwise_source_tp": int(method_splitwise_meta["splitwise_source_tp"]),
                    "splitwise_style_lut_mode": str(method_splitwise_meta["splitwise_style_lut_mode"]),
                    "splitwise_phase_detection_note": str(
                        method_splitwise_meta["splitwise_phase_detection_note"]
                    ),
                    "splitwise_decode_occupancy_note": str(
                        method_splitwise_meta["splitwise_decode_occupancy_note"]
                    ),
                    "splitwise_source_resolved_model": str(
                        method_splitwise_meta["splitwise_source_resolved_model"]
                    ),
                    "splitwise_source_resolved_hardware": str(
                        method_splitwise_meta["splitwise_source_resolved_hardware"]
                    ),
                    "splitwise_source_resolved_tp": int(
                        method_splitwise_meta["splitwise_source_resolved_tp"]
                    ),
                    "splitwise_source_match_status": str(
                        method_splitwise_meta["splitwise_source_match_status"]
                    ),
                    "splitwise_power_quality_flag": str(
                        method_splitwise_meta["splitwise_power_quality_flag"]
                    ),
                    "splitwise_power_support_status": str(
                        method_splitwise_meta["splitwise_power_support_status"]
                    ),
                    "splitwise_scheduler_policy": str(
                        method_splitwise_meta["splitwise_scheduler_policy"]
                    ),
                    "splitwise_extrapolation_events": int(
                        method_splitwise_meta["splitwise_extrapolation_events"]
                    ),
                    "splitwise_power_clamp_events": int(
                        method_splitwise_meta["splitwise_power_clamp_events"]
                    ),
                    "splitwise_max_batch_tokens_seen": float(
                        method_splitwise_meta["splitwise_max_batch_tokens_seen"]
                    ),
                    "metric_aggregation_scope": "heldout_all_traces_per_seed",
                    "seed_aggregation_mode": "median_over_seed_heldout_metrics",
                    "delta_energy_definition": "absolute_total_trace_energy_pct",
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
        "splitwise_style_lut_mode",
        "splitwise_phase_detection_note",
        "splitwise_decode_occupancy_note",
        "splitwise_source_resolved_model",
        "splitwise_source_resolved_hardware",
        "splitwise_source_resolved_tp",
        "splitwise_source_match_status",
        "splitwise_power_quality_flag",
        "splitwise_power_support_status",
        "splitwise_scheduler_policy",
        "splitwise_extrapolation_events",
        "splitwise_power_clamp_events",
        "splitwise_max_batch_tokens_seen",
        "metric_aggregation_scope",
        "seed_aggregation_mode",
        "delta_energy_definition",
    ]
    _write_csv(out_csv, rows, fieldnames)
    return {
        "rows": rows,
        "out_csv": out_csv,
        "num_configs": len(set(str(r["config_id"]) for r in rows)),
        "num_rows": len(rows),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Node-level baseline comparison (TDP / Mean / Splitwise Strict Emulation / Ours)."
    )
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
    parser.add_argument("--splitwise-source-model", default="llama-3-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=None)
    parser.add_argument(
        "--allow-synthetic-request-timestamps",
        action="store_true",
        help=(
            "Allow synthetic fallback when request_timestamps are missing. "
            "Default requires recorded request_timestamps."
        ),
    )
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
        allow_synthetic_request_timestamps=bool(
            args.allow_synthetic_request_timestamps
        ),
    )
    print("[run_baselines_node] Done")
    print(f"  rows     : {result['num_rows']}")
    print(f"  configs  : {result['num_configs']}")
    print(f"  out_csv  : {result['out_csv']}")


if __name__ == "__main__":
    main()
