#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

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

from model.classifiers.gru import GRUClassifier
from model.scripts.request_data_policy import (
    DEFAULT_ALLOWED_JSON_PREFIX,
    DEFAULT_REQUEST_TIMESTAMP_POLICY,
    REQUEST_TIMESTAMP_POLICIES,
    load_pair_manifest_map_with_policy,
    normalize_request_timestamp_policy,
)
from scripts.eval.baselines import (
    build_splitwise_lut_params,
    generate_mean,
    generate_ours,
    generate_splitwise_lut,
    generate_tdp,
)
from scripts.eval.pipeline_utils import (
    build_rollout_features_from_requests,
    estimate_ar1_params,
    load_gmm_params_json_dict,
    predict_sorted_gmm_labels_from_params,
)
from scripts.eval.run_baselines_node import (
    _estimate_splitwise_phase_targets_from_indices,
)

METHODS = ("tdp", "mean", "splitwise_lut", "ours")
STYLE = {
    "tdp": {"label": "TDP", "color": "#d62728", "linestyle": "--", "linewidth": 3.0},
    "mean": {"label": "Mean", "color": "#ff7f0e", "linestyle": "--", "linewidth": 3.0},
    "splitwise_lut": {
        "label": "Splitwise",
        "color": "#2ca02c",
        "linestyle": "-.",
        "linewidth": 2.6,
    },
    "ours": {"label": "Ours", "color": "#1f77b4", "linestyle": "-", "linewidth": 2.8},
}
CONFIG_ID_RE = re.compile(r"^(.+)_(A100|H100)_tp(\d+)$")
CONFIG_70B_TP4_RE = re.compile(r"^.+-70b_(A100|H100)_tp4$")
CONFIG_MODEL_SIZE_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_csv(
    path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]
) -> None:
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
    config_entry: Dict[str, object], base_dir: str
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


def _resolve_throughput(
    throughput_payload: Dict[str, object], config_id: str
) -> Dict[str, float]:
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
    dataset_path = _resolve_existing_path(
        str(row.get("dataset_npz", "")), experimental_base
    )
    split_path = _resolve_existing_path(
        str(row.get("split_json", "")), experimental_base
    )
    if dataset_path is None:
        raise ValueError(f"Dataset path not found for '{config_id}'")
    if split_path is None:
        raise ValueError(f"Split path not found for '{config_id}'")
    return dataset_path, split_path


def _load_pair_manifest_map(
    pair_manifest_csv: str,
    *,
    request_timestamp_policy: str = DEFAULT_REQUEST_TIMESTAMP_POLICY,
    allowed_json_prefix: str = DEFAULT_ALLOWED_JSON_PREFIX,
) -> Dict[str, str]:
    result = load_pair_manifest_map_with_policy(
        pair_manifest_csv,
        request_timestamp_policy=request_timestamp_policy,
        allowed_json_prefix=allowed_json_prefix,
        resolve_existing_path_fn=_resolve_existing_path,
        include_rejected_rows=False,
    )
    return dict(result.pair_map)


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
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
        and isinstance(state["model_state_dict"], dict)
    ):
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state)
    model.eval()
    return model


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
        sigma_innov = np.asarray(
            payload.get("sigma_innov", []), dtype=np.float64
        ).reshape(-1)
        sigma_marginal = np.asarray(
            payload.get("sigma_marginal", []), dtype=np.float64
        ).reshape(-1)
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


def _parse_rate(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return float(out)


def _build_mean_reference_power_pool(
    *,
    train_indices: Sequence[int],
    power_arr: np.ndarray,
    rate_arr: np.ndarray,
    target_rate: float,
    non_gpu_overhead_w: float,
    fallback_flat_gpu: np.ndarray,
    fallback_num_traces: int,
) -> Dict[str, object]:
    fallback = np.asarray(fallback_flat_gpu, dtype=np.float64).reshape(-1)
    if fallback.size <= 0:
        raise ValueError("fallback mean reference pool is empty")

    train_ids: List[int] = []
    n_power = int(len(power_arr))
    for idx_raw in train_indices:
        idx = int(idx_raw)
        if idx < 0 or idx >= n_power:
            continue
        train_ids.append(idx)

    if rate_arr.size <= 0:
        return {
            "power_flat_gpu": fallback,
            "source": "all_train_pool_no_rate_array",
            "target_rate": float(target_rate),
            "selected_rate": float("nan"),
            "num_traces": int(fallback_num_traces),
            "num_samples": int(fallback.size),
        }

    finite_rate_entries: List[Tuple[int, float]] = []
    max_rate_idx = int(rate_arr.size)
    for idx in train_ids:
        if idx < 0 or idx >= max_rate_idx:
            continue
        rate_v = _parse_rate(rate_arr[idx])
        if np.isfinite(rate_v):
            finite_rate_entries.append((idx, float(rate_v)))

    if len(finite_rate_entries) == 0:
        return {
            "power_flat_gpu": fallback,
            "source": "all_train_pool_no_finite_train_rate",
            "target_rate": float(target_rate),
            "selected_rate": float("nan"),
            "num_traces": int(fallback_num_traces),
            "num_samples": int(fallback.size),
        }

    selected_pair = min(
        finite_rate_entries, key=lambda pair: abs(float(pair[1]) - float(target_rate))
    )
    selected_rate = float(selected_pair[1])
    tol = 1e-9
    matched_idx = [
        int(idx)
        for idx, r in finite_rate_entries
        if abs(float(r) - float(selected_rate)) <= tol
    ]

    matched_traces: List[np.ndarray] = []
    for idx in matched_idx:
        p = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if p.size > 0:
            matched_traces.append(p.astype(np.float64))

    if len(matched_traces) == 0:
        return {
            "power_flat_gpu": fallback,
            "source": "all_train_pool_empty_matched_rate_power",
            "target_rate": float(target_rate),
            "selected_rate": float(selected_rate),
            "num_traces": int(fallback_num_traces),
            "num_samples": int(fallback.size),
        }

    matched_flat = np.concatenate(matched_traces, axis=0).astype(np.float64)
    matched_flat_gpu = np.clip(
        matched_flat - float(non_gpu_overhead_w), a_min=0.0, a_max=None
    )
    if matched_flat_gpu.size <= 0:
        return {
            "power_flat_gpu": fallback,
            "source": "all_train_pool_empty_matched_rate_gpu",
            "target_rate": float(target_rate),
            "selected_rate": float(selected_rate),
            "num_traces": int(fallback_num_traces),
            "num_samples": int(fallback.size),
        }

    return {
        "power_flat_gpu": matched_flat_gpu,
        "source": "closest_train_rate_pool",
        "target_rate": float(target_rate),
        "selected_rate": float(selected_rate),
        "num_traces": int(len(matched_traces)),
        "num_samples": int(matched_flat_gpu.size),
    }


def _extract_token_pools(json_paths: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    input_pool: List[int] = []
    output_pool: List[int] = []
    for path in json_paths:
        try:
            payload = _load_json(path)
        except Exception:
            continue
        input_lens = payload.get("input_lens")
        output_lens = payload.get("output_lens")
        if isinstance(input_lens, list):
            for x in input_lens:
                try:
                    v = int(float(x))
                except Exception:
                    continue
                if v > 0:
                    input_pool.append(v)
        if isinstance(output_lens, list):
            for x in output_lens:
                try:
                    v = int(float(x))
                except Exception:
                    continue
                if v > 0:
                    output_pool.append(v)

    if len(input_pool) == 0:
        input_pool = [256]
    if len(output_pool) == 0:
        output_pool = [128]
    return np.asarray(input_pool, dtype=np.int64), np.asarray(
        output_pool, dtype=np.int64
    )


def _generate_poisson_requests(
    *,
    duration_s: float,
    rate_per_s: float,
    input_pool: np.ndarray,
    output_pool: np.ndarray,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    n_requests = int(rng.poisson(float(rate_per_s) * float(duration_s)))
    if n_requests <= 0:
        return []
    arrivals = np.sort(rng.uniform(0.0, float(duration_s), size=n_requests))
    inputs = rng.choice(input_pool, size=n_requests, replace=True)
    outputs = rng.choice(output_pool, size=n_requests, replace=True)
    return [
        {
            "arrival_time": float(a),
            "input_tokens": float(int(nin)),
            "output_tokens": float(int(nout)),
        }
        for a, nin, nout in zip(arrivals, inputs, outputs)
    ]


def _build_bursty_global_multiplier(
    *,
    duration_s: float,
    dt: float,
    burst_rate_per_min: float,
    burst_mean_duration_s: float,
    burst_peak_scale: float,
    burst_background_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t_horizon = int(np.floor(float(duration_s) / float(dt)))
    if t_horizon <= 0:
        return np.zeros((0,), dtype=np.float64)

    t_sec = (np.arange(t_horizon, dtype=np.float64) + 0.5) * float(dt)
    mult = np.ones((t_horizon,), dtype=np.float64)

    if float(burst_background_sigma) > 0.0:
        tau_s = max(float(dt), float(burst_mean_duration_s))
        rho = float(np.exp(-float(dt) / tau_s))
        noise = np.zeros((t_horizon,), dtype=np.float64)
        noise[0] = float(rng.normal(0.0, 1.0))
        innov_scale = float(np.sqrt(max(1e-9, 1.0 - (rho * rho))))
        for i in range(1, t_horizon):
            noise[i] = (rho * noise[i - 1]) + (
                innov_scale * float(rng.normal(0.0, 1.0))
            )
        mult *= np.exp(float(burst_background_sigma) * noise)

    expected_bursts = max(0.0, float(burst_rate_per_min)) * (float(duration_s) / 60.0)
    n_bursts = int(rng.poisson(expected_bursts))
    for _ in range(n_bursts):
        center_s = float(rng.uniform(0.0, float(duration_s)))
        duration_draw_s = float(
            rng.lognormal(
                mean=float(np.log(max(float(dt), float(burst_mean_duration_s)))),
                sigma=0.45,
            )
        )
        sigma_s = max(float(dt), duration_draw_s / 3.0)
        amp = max(0.0, float(burst_peak_scale)) * float(
            rng.lognormal(mean=0.0, sigma=0.35)
        )
        pulse = amp * np.exp(-0.5 * ((t_sec - center_s) / sigma_s) ** 2)
        mult += pulse.astype(np.float64)

    mean_mult = float(np.mean(mult))
    if np.isfinite(mean_mult) and mean_mult > 0.0:
        mult = mult / mean_mult
    return np.clip(mult, a_min=0.02, a_max=None).astype(np.float64)


def _generate_inhomogeneous_poisson_requests(
    *,
    rate_profile_per_s: np.ndarray,
    dt: float,
    input_pool: np.ndarray,
    output_pool: np.ndarray,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    profile = np.asarray(rate_profile_per_s, dtype=np.float64).reshape(-1)
    if profile.size == 0:
        return []
    lam = np.clip(profile, a_min=0.0, a_max=None) * float(dt)
    n_per_bin = np.asarray(rng.poisson(lam), dtype=np.int64).reshape(-1)
    total = int(np.sum(n_per_bin))
    if total <= 0:
        return []

    bin_ids = np.repeat(np.arange(profile.size, dtype=np.int64), n_per_bin)
    offsets = rng.uniform(0.0, float(dt), size=total)
    arrivals = (bin_ids.astype(np.float64) * float(dt)) + offsets
    inputs = rng.choice(input_pool, size=total, replace=True)
    outputs = rng.choice(output_pool, size=total, replace=True)

    order = np.argsort(arrivals)
    arrivals = arrivals[order]
    inputs = inputs[order]
    outputs = outputs[order]

    return [
        {
            "arrival_time": float(a),
            "input_tokens": float(int(nin)),
            "output_tokens": float(int(nout)),
        }
        for a, nin, nout in zip(arrivals, inputs, outputs)
    ]


def _downsample_to_1s_mean(values: np.ndarray, dt: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    bins_per_sec = max(1, int(round(1.0 / float(dt))))
    usable = (arr.size // bins_per_sec) * bins_per_sec
    if usable <= 0:
        return np.array([float(np.mean(arr))], dtype=np.float64)
    trimmed = arr[:usable].reshape(-1, bins_per_sec)
    return np.mean(trimmed, axis=1).astype(np.float64)


def _value_at_exceedance(sorted_desc: np.ndarray, frac_exceeded: float) -> float:
    arr = np.asarray(sorted_desc, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan")
    idx = int(np.floor(float(frac_exceeded) * float(arr.size)))
    idx = max(0, min(idx, arr.size - 1))
    return float(arr[idx])


def _compute_facility_metrics(
    *,
    facility_w: np.ndarray,
    node_stack_w: np.ndarray,
    dt: float,
    n_nodes: int,
) -> Dict[str, float]:
    facility_w_arr = np.asarray(facility_w, dtype=np.float64).reshape(-1)
    facility_kw = facility_w_arr / 1000.0
    peak_kw = float(np.max(facility_kw))
    avg_kw = float(np.mean(facility_kw))
    par = float(peak_kw / avg_kw) if avg_kw > 0 else float("nan")

    one_sec_kw = _downsample_to_1s_mean(facility_kw, dt=float(dt))
    ramps = np.diff(one_sec_kw)
    if ramps.size > 0:
        ramp_p50 = float(np.percentile(ramps, 50))
        ramp_p95_abs = float(np.percentile(np.abs(ramps), 95))
        ramp_p99_abs = float(np.percentile(np.abs(ramps), 99))
        ramp_max_up = float(np.max(ramps))
        ramp_max_down = float(np.min(ramps))
    else:
        ramp_p50 = float("nan")
        ramp_p95_abs = float("nan")
        ramp_p99_abs = float("nan")
        ramp_max_up = float("nan")
        ramp_max_down = float("nan")

    ldc_sorted_kw = np.sort(facility_kw)[::-1]
    ldc_p99_kw = _value_at_exceedance(ldc_sorted_kw, 0.01)
    ldc_p95_kw = _value_at_exceedance(ldc_sorted_kw, 0.05)

    node_peaks_w = np.max(np.asarray(node_stack_w, dtype=np.float64), axis=1)
    peak_single_node_w = (
        float(np.max(node_peaks_w)) if node_peaks_w.size > 0 else float("nan")
    )
    peak_facility_w = float(np.max(facility_w_arr))
    denom = float(n_nodes) * peak_single_node_w
    diversity = (
        float(peak_facility_w / denom)
        if np.isfinite(denom) and denom > 0
        else float("nan")
    )

    return {
        "peak_power_kw": peak_kw,
        "avg_power_kw": avg_kw,
        "par": par,
        "ramp_p50_kw_per_s": ramp_p50,
        "ramp_p95_abs_kw_per_s": ramp_p95_abs,
        "ramp_p99_abs_kw_per_s": ramp_p99_abs,
        "ramp_max_up_kw_per_s": ramp_max_up,
        "ramp_max_down_kw_per_s": ramp_max_down,
        "ldc_p99_kw": ldc_p99_kw,
        "ldc_p95_kw": ldc_p95_kw,
        "diversity_factor": diversity,
        "peak_single_node_kw": peak_single_node_w / 1000.0,
    }


def _pretty_config_label(config_id: str) -> str:
    match = CONFIG_ID_RE.match(config_id)
    if not match:
        return config_id
    model_name, hw, tp = match.groups()
    return f"{model_name}, {hw} TP{tp}"


def _is_70b_tp4_config(config_id: str) -> bool:
    return CONFIG_70B_TP4_RE.match(str(config_id).strip()) is not None


def _plot_facility_traces(
    *,
    out_path: str,
    facility_kw_by_method: Mapping[str, np.ndarray],
    dt: float,
    n_nodes: int,
    config_id: str,
    lambda_req_per_s_per_node: float,
) -> None:
    _ensure_dir(os.path.dirname(out_path) or ".")
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 4))
    duration_min = 0.0
    for method in METHODS:
        if method not in facility_kw_by_method:
            continue
        style = STYLE[method]
        arr_kw = np.asarray(facility_kw_by_method[method], dtype=np.float64).reshape(-1)
        t_min = (np.arange(arr_kw.size, dtype=np.float64) * float(dt)) / 60.0
        if t_min.size > 0:
            duration_min = max(duration_min, float(t_min[-1] + (float(dt) / 60.0)))
        ax.plot(
            t_min,
            arr_kw,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=0.7,
        )
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 200.0)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Facility power (kW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_load_duration_curves(
    *,
    out_path: str,
    facility_kw_by_method: Mapping[str, np.ndarray],
) -> None:
    _ensure_dir(os.path.dirname(out_path) or ".")
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for method in METHODS:
        if method not in facility_kw_by_method:
            continue
        style = STYLE[method]
        arr_kw = np.asarray(facility_kw_by_method[method], dtype=np.float64).reshape(-1)
        sorted_desc = np.sort(arr_kw)[::-1]
        x = np.arange(sorted_desc.size, dtype=np.float64) / float(
            max(1, sorted_desc.size)
        )
        ax.plot(
            x,
            sorted_desc,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=0.7,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 200.0)
    ax.set_xlabel("Fraction of time exceeded")
    ax.set_ylabel("Facility power (kW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _export_facility_traces(
    *,
    out_dir: str,
    facility_w_by_method: Mapping[str, np.ndarray],
    config_id: str,
    base_seed: int,
    dt: float,
    duration_s: float,
    n_nodes: int,
    lambda_req_per_s_per_node: float,
    facility_power_mode: str,
    pue: float,
    non_gpu_overhead_w: float,
    traffic_model: str,
    burst_rate_per_min: float,
    burst_mean_duration_s: float,
    burst_peak_scale: float,
    burst_background_sigma: float,
    burst_node_scale_sigma: float,
) -> Dict[str, object]:
    _ensure_dir(out_dir)
    saved_files: Dict[str, str] = {}
    for method in METHODS:
        arr = facility_w_by_method.get(method)
        if arr is None:
            continue
        out_name = f"facility_power_250ms_{method}_w.npy"
        out_path = str(Path(out_dir) / out_name)
        np.save(out_path, np.asarray(arr, dtype=np.float64))
        saved_files[method] = out_path

    manifest_path = str(Path(out_dir) / "facility_trace_manifest.json")
    manifest = {
        "schema_version": "facility-trace-export-v1",
        "config_id": str(config_id),
        "base_seed": int(base_seed),
        "dt": float(dt),
        "duration_s": float(duration_s),
        "n_nodes": int(n_nodes),
        "lambda_req_per_s_per_node": float(lambda_req_per_s_per_node),
        "facility_power_mode": str(facility_power_mode),
        "pue": float(pue),
        "non_gpu_overhead_w": float(non_gpu_overhead_w),
        "traffic": {
            "model": str(traffic_model),
            "burst_rate_per_min": float(burst_rate_per_min),
            "burst_mean_duration_s": float(burst_mean_duration_s),
            "burst_peak_scale": float(burst_peak_scale),
            "burst_background_sigma": float(burst_background_sigma),
            "burst_node_scale_sigma": float(burst_node_scale_sigma),
        },
        "files": saved_files,
        "missing_methods": [
            str(method) for method in METHODS if str(method) not in saved_files
        ],
    }
    _write_json(manifest_path, manifest)
    return {
        "out_dir": str(out_dir),
        "manifest_path": manifest_path,
        "files": saved_files,
    }


def run_baselines_facility(
    *,
    run_manifest: str = "results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1_gru_all/manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    ar1_params_dir: str = "results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/ar1_params",
    out_csv: str = "results/eval_paper/baselines_facility_metrics.csv",
    traces_pdf: str = "figures/baselines_facility_traces.pdf",
    ldc_pdf: str = "figures/baselines_load_duration.pdf",
    config_id: str = "deepseek-r1-distill-70b_H100_tp4",
    n_nodes: int = 60,
    duration_s: float = 3600.0,
    dt: float = 0.25,
    lambda_req_per_s_per_node: float = 0.25,
    tp_gpus: int = 4,
    n_gpus_for_gpu_power: int = 4,
    gpu_tdp_w: float = 700.0,
    pue: float = 1.0,
    non_gpu_overhead_w: float = 0.0,
    facility_power_mode: str = "gpu_sum_only",
    base_seed: int = 42,
    device: str = "auto",
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    ours_std_scale: float = 1.0,
    ours_logit_temperature: float = 1.0,
    splitwise_perf_model_csv: str = "data/perf_model.csv",
    splitwise_source_model: str = "llama2-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: int = 4,
    splitwise_calibration_mode: str = "train_phase_matched_v1",
    traffic_model: str = "poisson",
    burst_rate_per_min: float = 2.0,
    burst_mean_duration_s: float = 20.0,
    burst_peak_scale: float = 6.0,
    burst_background_sigma: float = 0.35,
    burst_node_scale_sigma: float = 0.2,
    save_facility_traces_dir: str = "",
    skip_plots: bool = False,
    request_timestamp_policy: str = DEFAULT_REQUEST_TIMESTAMP_POLICY,
    allowed_json_prefix: str = DEFAULT_ALLOWED_JSON_PREFIX,
) -> Dict[str, object]:
    if int(n_nodes) <= 0:
        raise ValueError("n_nodes must be >= 1")
    if float(duration_s) <= 0:
        raise ValueError("duration_s must be > 0")
    if float(dt) <= 0:
        raise ValueError("dt must be > 0")
    if float(lambda_req_per_s_per_node) < 0:
        raise ValueError("lambda_req_per_s_per_node must be >= 0")
    if int(tp_gpus) <= 0:
        raise ValueError("tp_gpus must be >= 1")
    if int(n_gpus_for_gpu_power) < int(tp_gpus):
        raise ValueError("n_gpus_for_gpu_power must be >= tp_gpus")
    if float(gpu_tdp_w) <= 0.0:
        raise ValueError("gpu_tdp_w must be > 0")
    if float(pue) <= 0:
        raise ValueError("pue must be > 0")
    if float(non_gpu_overhead_w) < 0:
        raise ValueError("non_gpu_overhead_w must be >= 0")
    facility_mode = str(facility_power_mode).strip().lower()
    if facility_mode not in {"gpu_sum_only", "legacy_pue_overhead"}:
        raise ValueError(
            "facility_power_mode must be one of {'gpu_sum_only', 'legacy_pue_overhead'}"
        )
    if float(ours_std_scale) <= 0:
        raise ValueError("ours_std_scale must be > 0")
    if float(ours_logit_temperature) <= 0:
        raise ValueError("ours_logit_temperature must be > 0")
    if int(splitwise_source_tp) != 4:
        raise ValueError("splitwise_source_tp must be 4 for 70B TP4-only comparison.")
    if not _is_70b_tp4_config(config_id):
        raise ValueError(
            f"config_id '{config_id}' is out of scope for Splitwise baseline comparison (expected *-70b_*_tp4)."
        )
    traffic_mode = str(traffic_model).strip().lower()
    if traffic_mode not in {"poisson", "bursty"}:
        raise ValueError("traffic_model must be one of {'poisson', 'bursty'}")
    if float(burst_rate_per_min) < 0:
        raise ValueError("burst_rate_per_min must be >= 0")
    if float(burst_mean_duration_s) <= 0:
        raise ValueError("burst_mean_duration_s must be > 0")
    if float(burst_peak_scale) < 0:
        raise ValueError("burst_peak_scale must be >= 0")
    if float(burst_background_sigma) < 0:
        raise ValueError("burst_background_sigma must be >= 0")
    if float(burst_node_scale_sigma) < 0:
        raise ValueError("burst_node_scale_sigma must be >= 0")
    request_timestamp_policy = normalize_request_timestamp_policy(request_timestamp_policy)

    run_manifest_payload = _load_json(run_manifest)
    run_cfgs = run_manifest_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    run_manifest_base = str(Path(run_manifest).resolve().parent)

    cfg_entry = run_cfgs.get(config_id)
    if not isinstance(cfg_entry, dict):
        raise ValueError(f"config_id '{config_id}' not found in run manifest")
    if str(cfg_entry.get("status", "")) != "trained":
        raise ValueError(f"config '{config_id}' is not trained in run manifest")

    checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(
        cfg_entry, run_manifest_base
    )
    norm_payload = _load_json(norm_path)
    norm_cfg = _extract_norm_for_eval(norm_payload)
    gmm_cfg = load_gmm_params_json_dict(_load_json(gmm_path))
    k = int(cfg_entry.get("k", gmm_cfg["k"]))
    if k != int(gmm_cfg["k"]):
        raise ValueError(f"k mismatch: manifest={k}, gmm={int(gmm_cfg['k'])}")

    feature_set = str(
        cfg_entry.get("feature_set", norm_payload.get("feature_set", "f2"))
    ).lower()
    if feature_set not in {"f2", "f3"}:
        raise ValueError(f"invalid feature_set: {feature_set}")
    input_dim = int(cfg_entry.get("input_dim", 2 if feature_set == "f2" else 3))
    hidden_dim = int(cfg_entry.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
    num_layers = int(cfg_entry.get("num_layers", norm_payload.get("num_layers", 1)))

    resolved_device = _resolve_device(device)
    model = _load_model(
        checkpoint_path=checkpoint_path,
        k=k,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=resolved_device,
    )

    throughput_payload = _load_json(throughput_db)
    throughput = _resolve_throughput(throughput_payload, config_id)
    experimental_payload = _load_json(experimental_manifest)
    experimental_base = str(Path(experimental_manifest).resolve().parent)
    dataset_path, split_path = _resolve_experimental_paths(
        experimental_payload,
        config_id=config_id,
        experimental_base=experimental_base,
    )
    split_payload = _load_json(split_path)
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]
    test_indices = [int(x) for x in split_payload.get("test_indices", [])]
    if len(train_indices) == 0 and len(test_indices) == 0:
        raise ValueError("both train and test splits are empty")

    pair_map = _load_pair_manifest_map(
        pair_manifest_csv,
        request_timestamp_policy=request_timestamp_policy,
        allowed_json_prefix=allowed_json_prefix,
    )
    with np.load(dataset_path, allow_pickle=True) as data:
        pair_key_arr = np.asarray(data["pair_key"], dtype=object)
        power_arr = np.asarray(data["power"], dtype=object)
        power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
        rate_arr = (
            np.asarray(data["rate"], dtype=object)
            if "rate" in data
            else np.asarray([], dtype=object)
        )
    n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))

    train_power_traces: List[np.ndarray] = []
    for idx in train_indices:
        if idx < 0 or idx >= n_total:
            continue
        p = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if p.size > 0:
            train_power_traces.append(p.astype(np.float64))
    if len(train_power_traces) == 0:
        for i in range(n_total):
            p = np.asarray(power_arr[i], dtype=np.float64).reshape(-1)
            if p.size > 0:
                train_power_traces.append(p.astype(np.float64))
                if len(train_power_traces) >= 3:
                    break
    if len(train_power_traces) == 0:
        raise ValueError("unable to build training power pool")
    train_power_flat = np.concatenate(train_power_traces, axis=0).astype(np.float64)
    train_power_flat_gpu = np.clip(
        train_power_flat - float(non_gpu_overhead_w), a_min=0.0, a_max=None
    )
    mean_pool = _build_mean_reference_power_pool(
        train_indices=train_indices,
        power_arr=power_arr,
        rate_arr=rate_arr,
        target_rate=float(lambda_req_per_s_per_node),
        non_gpu_overhead_w=float(non_gpu_overhead_w),
        fallback_flat_gpu=train_power_flat_gpu,
        fallback_num_traces=len(train_power_traces),
    )
    mean_train_power_flat_gpu = np.asarray(
        mean_pool["power_flat_gpu"], dtype=np.float64
    ).reshape(-1)
    mean_pool_source = str(mean_pool.get("source", ""))
    mean_pool_target_rate = float(
        mean_pool.get("target_rate", float(lambda_req_per_s_per_node))
    )
    mean_pool_selected_rate = float(mean_pool.get("selected_rate", float("nan")))
    mean_pool_num_traces = int(mean_pool.get("num_traces", 0))
    mean_pool_num_samples = int(mean_pool.get("num_samples", 0))
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
        non_gpu_overhead_w=float(non_gpu_overhead_w),
    )
    splitwise_lut_params = build_splitwise_lut_params(
        config_id=config_id,
        perf_model_csv=splitwise_perf_model_csv,
        train_power_flat=train_power_flat_gpu,
        splitwise_source_model=splitwise_source_model,
        splitwise_source_hardware=splitwise_source_hardware,
        splitwise_source_tp=int(splitwise_source_tp),
        splitwise_calibration_mode=splitwise_calibration_mode,
        n_gpus_per_node=int(n_gpus_for_gpu_power),
        per_gpu_tdp_cap_w=float(gpu_tdp_w),
        target_idle_node_gpu_w=phase_targets.get("target_idle_node_gpu_w"),
        target_decode_node_gpu_w=phase_targets.get("target_decode_node_gpu_w"),
        target_prefill_node_gpu_w=phase_targets.get("target_prefill_node_gpu_w"),
    )
    splitwise_phase_detection_note = str(
        splitwise_lut_params.get("phase_detection_note", "")
    )
    splitwise_decode_occupancy_note = str(
        splitwise_lut_params.get("decode_occupancy_note", "")
    )

    heldout_json_paths: List[str] = []
    for idx in test_indices:
        if idx < 0 or idx >= n_total:
            continue
        pair_key = str(pair_key_arr[idx])
        json_path = pair_map.get(pair_key)
        if json_path is not None:
            heldout_json_paths.append(json_path)
    input_pool, output_pool = _extract_token_pools(heldout_json_paths)

    ar1_params = None
    if _is_moe_config(config_id):
        ar1_params = _load_or_estimate_ar1_params(
            config_id=config_id,
            gmm_params=gmm_cfg,
            train_power_traces=train_power_traces,
            ar1_params_dir=ar1_params_dir,
        )

    t_horizon = int(np.floor(float(duration_s) / float(dt)))
    if t_horizon <= 0:
        raise ValueError("computed horizon is zero; increase duration_s or reduce dt")

    global_multiplier = np.ones((t_horizon,), dtype=np.float64)
    if traffic_mode == "bursty":
        rng_global = np.random.default_rng(int(base_seed) + 7919)
        global_multiplier = _build_bursty_global_multiplier(
            duration_s=float(duration_s),
            dt=float(dt),
            burst_rate_per_min=float(burst_rate_per_min),
            burst_mean_duration_s=float(burst_mean_duration_s),
            burst_peak_scale=float(burst_peak_scale),
            burst_background_sigma=float(burst_background_sigma),
            rng=rng_global,
        )

    traces_by_method: Dict[str, List[np.ndarray]] = {method: [] for method in METHODS}
    method_errors: Dict[str, List[str]] = {method: [] for method in METHODS}
    for node_idx in range(int(n_nodes)):
        node_seed = int(base_seed) + int(node_idx) * 1009
        rng_node = np.random.default_rng(node_seed)
        if traffic_mode == "poisson":
            requests = _generate_poisson_requests(
                duration_s=float(duration_s),
                rate_per_s=float(lambda_req_per_s_per_node),
                input_pool=input_pool,
                output_pool=output_pool,
                rng=rng_node,
            )
        else:
            node_scale_sigma = float(burst_node_scale_sigma)
            if node_scale_sigma > 0.0:
                node_scale = float(
                    rng_node.lognormal(
                        mean=-0.5 * (node_scale_sigma**2),
                        sigma=node_scale_sigma,
                    )
                )
            else:
                node_scale = 1.0
            rate_profile = (
                float(lambda_req_per_s_per_node)
                * float(node_scale)
                * np.asarray(global_multiplier, dtype=np.float64)
            )
            requests = _generate_inhomogeneous_poisson_requests(
                rate_profile_per_s=rate_profile,
                dt=float(dt),
                input_pool=input_pool,
                output_pool=output_pool,
                rng=rng_node,
            )
        feat = build_rollout_features_from_requests(
            requests=requests,
            throughput=throughput,
            norm=norm_cfg,
            T=t_horizon,
            dt=float(dt),
            feature_set=feature_set,
        )
        features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
        if features_norm.ndim != 2 or features_norm.shape[0] <= 0:
            for method in METHODS:
                method_errors[method].append(f"node={node_idx}:invalid_feature_shape")
            continue
        n_eval = int(min(t_horizon, features_norm.shape[0]))
        feat_eval = features_norm[:n_eval]
        a_raw_eval = np.asarray(feat["A_raw"], dtype=np.float64).reshape(-1)[:n_eval]
        delta_a_raw_eval = np.asarray(feat["delta_A_raw"], dtype=np.float64).reshape(
            -1
        )[:n_eval]
        p0 = float(rng_node.choice(train_power_flat))

        for method in METHODS:
            try:
                if method == "tdp":
                    # TDP baseline here is GPU-only; non-GPU overhead is added once at facility aggregation.
                    pred = generate_tdp(
                        n_eval,
                        {
                            "tdp_node": float(tp_gpus) * float(gpu_tdp_w),
                            "non_gpu_power_w": 0.0,
                        },
                    )
                elif method == "mean":
                    pred = generate_mean(n_eval, {}, mean_train_power_flat_gpu)
                elif method == "splitwise_lut":
                    pred = generate_splitwise_lut(
                        a_raw_eval,
                        delta_a_raw_eval,
                        {
                            "config_id": config_id,
                            "tp": int(tp_gpus),
                            "n_gpus_per_node": int(n_gpus_for_gpu_power),
                            "non_gpu_power_w": 0.0,
                        },
                        splitwise_lut_params,
                    )
                elif method == "ours":
                    pred_node = generate_ours(
                        feat_eval,
                        {
                            "device": str(resolved_device),
                            "p0": p0,
                            "decode_mode": decode_mode,
                            "median_filter_window": int(median_filter_window),
                            "std_scale": float(ours_std_scale),
                            "logit_temperature": float(ours_logit_temperature),
                            "clamp_range": (
                                norm_cfg["power_min"],
                                norm_cfg["power_max"],
                            ),
                            "ar1_params": ar1_params,
                        },
                        model,
                        gmm_cfg,
                        rng=np.random.default_rng(node_seed + 23),
                    )
                    pred = np.clip(
                        np.asarray(pred_node, dtype=np.float64).reshape(-1)
                        - float(non_gpu_overhead_w),
                        a_min=0.0,
                        a_max=None,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                traces_by_method[method].append(
                    np.asarray(pred, dtype=np.float64).reshape(-1)
                )
            except Exception as exc:
                method_errors[method].append(
                    f"node={node_idx}:{type(exc).__name__}:{exc}"
                )

    rows: List[Dict[str, object]] = []
    facility_kw_by_method: Dict[str, np.ndarray] = {}
    facility_w_by_method: Dict[str, np.ndarray] = {}
    for method in METHODS:
        if len(traces_by_method[method]) != int(n_nodes):
            reason = (
                method_errors[method][0]
                if method_errors[method]
                else "missing_node_traces"
            )
            rows.append(
                {
                    "config_id": config_id,
                    "method": method,
                    "status": "failed",
                    "reason": reason,
                    "n_nodes": int(n_nodes),
                    "duration_s": float(duration_s),
                    "dt": float(dt),
                    "lambda_req_per_s_per_node": float(lambda_req_per_s_per_node),
                    "tp_gpus": int(tp_gpus),
                    "n_gpus_for_gpu_power": int(n_gpus_for_gpu_power),
                    "gpu_tdp_w": float(gpu_tdp_w),
                    "facility_power_mode": facility_mode,
                    "pue": float(pue),
                    "non_gpu_overhead_w": float(non_gpu_overhead_w),
                    "mean_pool_source": mean_pool_source,
                    "mean_pool_target_rate": float(mean_pool_target_rate),
                    "mean_pool_selected_rate": float(mean_pool_selected_rate),
                    "mean_pool_num_traces": int(mean_pool_num_traces),
                    "mean_pool_num_samples": int(mean_pool_num_samples),
                    "ours_std_scale": float(ours_std_scale),
                    "ours_logit_temperature": float(ours_logit_temperature),
                    "splitwise_source_model": splitwise_source_model,
                    "splitwise_source_hardware": splitwise_source_hardware,
                    "splitwise_source_tp": int(splitwise_source_tp),
                    "splitwise_calibration_mode": splitwise_calibration_mode,
                    "splitwise_phase_detection_note": splitwise_phase_detection_note,
                    "splitwise_decode_occupancy_note": splitwise_decode_occupancy_note,
                    "traffic_model": traffic_mode,
                    "burst_rate_per_min": float(burst_rate_per_min),
                    "burst_mean_duration_s": float(burst_mean_duration_s),
                    "burst_peak_scale": float(burst_peak_scale),
                    "burst_background_sigma": float(burst_background_sigma),
                    "burst_node_scale_sigma": float(burst_node_scale_sigma),
                    "peak_power_kw": float("nan"),
                    "avg_power_kw": float("nan"),
                    "par": float("nan"),
                    "ramp_p50_kw_per_s": float("nan"),
                    "ramp_p95_abs_kw_per_s": float("nan"),
                    "ramp_p99_abs_kw_per_s": float("nan"),
                    "ramp_max_up_kw_per_s": float("nan"),
                    "ramp_max_down_kw_per_s": float("nan"),
                    "ldc_p99_kw": float("nan"),
                    "ldc_p95_kw": float("nan"),
                    "diversity_factor": float("nan"),
                    "peak_single_node_kw": float("nan"),
                }
            )
            continue

        node_stack = np.stack(traces_by_method[method], axis=0).astype(np.float64)
        if facility_mode == "gpu_sum_only":
            facility_w = np.sum(node_stack, axis=0)
        else:
            facility_w = float(pue) * (
                np.sum(node_stack, axis=0)
                + (float(n_nodes) * float(non_gpu_overhead_w))
            )
        facility_w_by_method[method] = np.asarray(facility_w, dtype=np.float64)
        facility_kw_by_method[method] = facility_w / 1000.0
        metrics = _compute_facility_metrics(
            facility_w=facility_w,
            node_stack_w=node_stack,
            dt=float(dt),
            n_nodes=int(n_nodes),
        )
        rows.append(
            {
                "config_id": config_id,
                "method": method,
                "status": "evaluated",
                "reason": "",
                "n_nodes": int(n_nodes),
                "duration_s": float(duration_s),
                "dt": float(dt),
                "lambda_req_per_s_per_node": float(lambda_req_per_s_per_node),
                "tp_gpus": int(tp_gpus),
                "n_gpus_for_gpu_power": int(n_gpus_for_gpu_power),
                "gpu_tdp_w": float(gpu_tdp_w),
                "facility_power_mode": facility_mode,
                "pue": float(pue),
                "non_gpu_overhead_w": float(non_gpu_overhead_w),
                "mean_pool_source": mean_pool_source,
                "mean_pool_target_rate": float(mean_pool_target_rate),
                "mean_pool_selected_rate": float(mean_pool_selected_rate),
                "mean_pool_num_traces": int(mean_pool_num_traces),
                "mean_pool_num_samples": int(mean_pool_num_samples),
                "ours_std_scale": float(ours_std_scale),
                "ours_logit_temperature": float(ours_logit_temperature),
                "splitwise_source_model": splitwise_source_model,
                "splitwise_source_hardware": splitwise_source_hardware,
                "splitwise_source_tp": int(splitwise_source_tp),
                "splitwise_calibration_mode": splitwise_calibration_mode,
                "splitwise_phase_detection_note": splitwise_phase_detection_note,
                "splitwise_decode_occupancy_note": splitwise_decode_occupancy_note,
                "traffic_model": traffic_mode,
                "burst_rate_per_min": float(burst_rate_per_min),
                "burst_mean_duration_s": float(burst_mean_duration_s),
                "burst_peak_scale": float(burst_peak_scale),
                "burst_background_sigma": float(burst_background_sigma),
                "burst_node_scale_sigma": float(burst_node_scale_sigma),
                **metrics,
            }
        )

    fieldnames = [
        "config_id",
        "method",
        "status",
        "reason",
        "n_nodes",
        "duration_s",
        "dt",
        "lambda_req_per_s_per_node",
        "tp_gpus",
        "n_gpus_for_gpu_power",
        "gpu_tdp_w",
        "facility_power_mode",
        "pue",
        "non_gpu_overhead_w",
        "mean_pool_source",
        "mean_pool_target_rate",
        "mean_pool_selected_rate",
        "mean_pool_num_traces",
        "mean_pool_num_samples",
        "ours_std_scale",
        "ours_logit_temperature",
        "splitwise_source_model",
        "splitwise_source_hardware",
        "splitwise_source_tp",
        "splitwise_calibration_mode",
        "splitwise_phase_detection_note",
        "splitwise_decode_occupancy_note",
        "traffic_model",
        "burst_rate_per_min",
        "burst_mean_duration_s",
        "burst_peak_scale",
        "burst_background_sigma",
        "burst_node_scale_sigma",
        "peak_power_kw",
        "avg_power_kw",
        "par",
        "ramp_p50_kw_per_s",
        "ramp_p95_abs_kw_per_s",
        "ramp_p99_abs_kw_per_s",
        "ramp_max_up_kw_per_s",
        "ramp_max_down_kw_per_s",
        "ldc_p99_kw",
        "ldc_p95_kw",
        "diversity_factor",
        "peak_single_node_kw",
    ]
    _write_csv(out_csv, rows, fieldnames)

    export_result: Dict[str, object] | None = None
    export_dir = str(save_facility_traces_dir).strip()
    if export_dir != "" and len(facility_w_by_method) > 0:
        export_result = _export_facility_traces(
            out_dir=export_dir,
            facility_w_by_method=facility_w_by_method,
            config_id=config_id,
            base_seed=int(base_seed),
            dt=float(dt),
            duration_s=float(duration_s),
            n_nodes=int(n_nodes),
            lambda_req_per_s_per_node=float(lambda_req_per_s_per_node),
            facility_power_mode=facility_mode,
            pue=float(pue),
            non_gpu_overhead_w=float(non_gpu_overhead_w),
            traffic_model=traffic_mode,
            burst_rate_per_min=float(burst_rate_per_min),
            burst_mean_duration_s=float(burst_mean_duration_s),
            burst_peak_scale=float(burst_peak_scale),
            burst_background_sigma=float(burst_background_sigma),
            burst_node_scale_sigma=float(burst_node_scale_sigma),
        )

    if len(facility_kw_by_method) > 0 and (not bool(skip_plots)):
        _plot_facility_traces(
            out_path=traces_pdf,
            facility_kw_by_method=facility_kw_by_method,
            dt=float(dt),
            n_nodes=int(n_nodes),
            config_id=config_id,
            lambda_req_per_s_per_node=float(lambda_req_per_s_per_node),
        )
        _plot_load_duration_curves(
            out_path=ldc_pdf,
            facility_kw_by_method=facility_kw_by_method,
        )

    return {
        "out_csv": out_csv,
        "traces_pdf": traces_pdf,
        "ldc_pdf": ldc_pdf,
        "rows": rows,
        "facility_traces_dir": (
            None if export_result is None else str(export_result["out_dir"])
        ),
        "facility_trace_manifest": (
            None if export_result is None else str(export_result["manifest_path"])
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Facility-level baseline comparison (TDP / Mean / Splitwise / Ours)."
        )
    )
    parser.add_argument(
        "--run-manifest",
        default="results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json",
    )
    parser.add_argument(
        "--experimental-manifest",
        default="results/experimental_continuous_v1_gru_all/manifest.json",
    )
    parser.add_argument(
        "--throughput-db", default="model/config/throughput_database.json"
    )
    parser.add_argument(
        "--pair-manifest-csv", default="results/stage0/pair_manifest.csv"
    )
    parser.add_argument(
        "--request-timestamp-policy",
        default=DEFAULT_REQUEST_TIMESTAMP_POLICY,
        choices=list(REQUEST_TIMESTAMP_POLICIES),
    )
    parser.add_argument("--allowed-json-prefix", default=DEFAULT_ALLOWED_JSON_PREFIX)
    parser.add_argument(
        "--ar1-params-dir",
        default="results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/ar1_params",
        help="Directory containing AR(1) params JSON files (used only for MoE configs).",
    )
    parser.add_argument("--config-id", default="deepseek-r1-distill-70b_H100_tp4")
    parser.add_argument("--n-nodes", type=int, default=60)
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--lambda-req-per-s-per-node", type=float, default=0.25)
    parser.add_argument(
        "--tp-gpus",
        type=int,
        default=4,
        help="Number of active GPUs for the TDP/Splitwise GPU-power baselines.",
    )
    parser.add_argument(
        "--n-gpus-for-gpu-power",
        type=int,
        default=4,
        help="Total GPUs represented in GPU-power accounting.",
    )
    parser.add_argument(
        "--gpu-tdp-w",
        type=float,
        default=700.0,
        help="Per-GPU TDP cap (W) used by the TDP baseline and Splitwise cap.",
    )
    parser.add_argument("--pue", type=float, default=1.0)
    parser.add_argument("--non-gpu-overhead-w", type=float, default=0.0)
    parser.add_argument(
        "--facility-power-mode",
        choices=["gpu_sum_only", "legacy_pue_overhead"],
        default="gpu_sum_only",
        help="gpu_sum_only: facility power = sum of node GPU power (default).",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--decode-mode", choices=["stochastic", "argmax"], default="stochastic"
    )
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument(
        "--ours-std-scale",
        type=float,
        default=1.0,
        help="Scales std-dev used by Ours sampler (e.g., 1.5 increases variation).",
    )
    parser.add_argument(
        "--ours-logit-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for Ours state sampling (>1 gives more switching).",
    )
    parser.add_argument("--splitwise-perf-model-csv", default="data/perf_model.csv")
    parser.add_argument("--splitwise-source-model", default="llama2-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=4)
    parser.add_argument(
        "--traffic-model",
        choices=["poisson", "bursty"],
        default="poisson",
        help="Arrival traffic model for synthetic facility requests.",
    )
    parser.add_argument(
        "--burst-rate-per-min",
        type=float,
        default=2.0,
        help="Expected number of shared global bursts per minute (bursty mode only).",
    )
    parser.add_argument(
        "--burst-mean-duration-s",
        type=float,
        default=20.0,
        help="Mean burst duration in seconds (bursty mode only).",
    )
    parser.add_argument(
        "--burst-peak-scale",
        type=float,
        default=6.0,
        help="Approximate additive burst peak multiplier before normalization.",
    )
    parser.add_argument(
        "--burst-background-sigma",
        type=float,
        default=0.35,
        help="Strength of low-frequency multiplicative background variation.",
    )
    parser.add_argument(
        "--burst-node-scale-sigma",
        type=float,
        default=0.2,
        help="Node-to-node lognormal rate spread (bursty mode only).",
    )
    parser.add_argument(
        "--out-csv", default="results/eval_paper/baselines_facility_metrics.csv"
    )
    parser.add_argument("--traces-pdf", default="figures/baselines_facility_traces.pdf")
    parser.add_argument("--ldc-pdf", default="figures/baselines_load_duration.pdf")
    parser.add_argument(
        "--save-facility-traces-dir",
        default="",
        help="Optional output directory for per-method 250ms facility traces (.npy) and export manifest.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="If set, skips writing traces/LDC plot PDFs.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_baselines_facility(
        run_manifest=args.run_manifest,
        experimental_manifest=args.experimental_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        ar1_params_dir=args.ar1_params_dir,
        out_csv=args.out_csv,
        traces_pdf=args.traces_pdf,
        ldc_pdf=args.ldc_pdf,
        config_id=args.config_id,
        n_nodes=args.n_nodes,
        duration_s=args.duration_s,
        dt=args.dt,
        lambda_req_per_s_per_node=args.lambda_req_per_s_per_node,
        tp_gpus=args.tp_gpus,
        n_gpus_for_gpu_power=args.n_gpus_for_gpu_power,
        gpu_tdp_w=args.gpu_tdp_w,
        pue=args.pue,
        non_gpu_overhead_w=args.non_gpu_overhead_w,
        facility_power_mode=args.facility_power_mode,
        base_seed=args.base_seed,
        device=args.device,
        decode_mode=args.decode_mode,
        median_filter_window=args.median_filter_window,
        ours_std_scale=args.ours_std_scale,
        ours_logit_temperature=args.ours_logit_temperature,
        splitwise_perf_model_csv=args.splitwise_perf_model_csv,
        splitwise_source_model=args.splitwise_source_model,
        splitwise_source_hardware=args.splitwise_source_hardware,
        splitwise_source_tp=args.splitwise_source_tp,
        traffic_model=args.traffic_model,
        burst_rate_per_min=args.burst_rate_per_min,
        burst_mean_duration_s=args.burst_mean_duration_s,
        burst_peak_scale=args.burst_peak_scale,
        burst_background_sigma=args.burst_background_sigma,
        burst_node_scale_sigma=args.burst_node_scale_sigma,
        save_facility_traces_dir=args.save_facility_traces_dir,
        skip_plots=bool(args.skip_plots),
        request_timestamp_policy=args.request_timestamp_policy,
        allowed_json_prefix=args.allowed_json_prefix,
    )
    print("[run_baselines_facility] Done")
    print(f"  metrics_csv : {result['out_csv']}")
    print(f"  traces_pdf  : {result['traces_pdf']}")
    print(f"  ldc_pdf     : {result['ldc_pdf']}")
    if result["facility_trace_manifest"] is not None:
        print(f"  traces_manifest: {result['facility_trace_manifest']}")


if __name__ == "__main__":
    main()
