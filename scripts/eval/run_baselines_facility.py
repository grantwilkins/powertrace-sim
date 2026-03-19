#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from model.utils.io import ensure_dir, load_json
from scripts.eval.baselines import (
    SPLITWISE_REMOVED_MESSAGE,
    SPLITWISE_STYLE_LUT_V1,
    build_splitwise_style_lut_params,
    generate_mean,
    generate_ours,
    generate_splitwise_style_lut_trace,
    generate_tdp,
    normalize_splitwise_style_lut_mode,
)
from scripts.eval.pipeline_utils import (
    build_rollout_features_from_requests,
    estimate_ar1_params,
    extract_norm_params,
    load_gmm_params_json_dict,
    load_gru_classifier,
    predict_sorted_gmm_labels_from_params,
)
from scripts.eval.pipeline_utils import (
    resolve_checkpoint_norm_gmm_paths as _shared_resolve_checkpoint_norm_gmm_paths,
)
from scripts.eval.pipeline_utils import (
    resolve_experimental_paths as _shared_resolve_experimental_paths,
)
from scripts.eval.pipeline_utils import (
    resolve_throughput as _shared_resolve_throughput,
)
from scripts.eval.run_baselines_node import _resolve_per_gpu_chip_tdp_w

DEFAULT_METHODS = ("tdp", "mean", "splitwise_strict", "ours")
STYLE = {
    "tdp": {"label": "TDP", "color": "#006CB8", "linestyle": "--", "linewidth": 3.0},
    "mean": {"label": "Mean", "color": "#620059", "linestyle": "--", "linewidth": 3.0},
    "splitwise_strict": {
        "label": "LUT-based",
        "color": "#E50808",  # Stanford Cardinal red
        "linestyle": ":",
        "linewidth": 2.8,
    },
    "ours": {"label": "Ours", "color": "#006F54", "linestyle": "-", "linewidth": 2.8},
}
CONFIG_ID_RE = re.compile(r"^(.+)_(A100|H100)_tp(\d+)$")
CONFIG_70B_TP4_RE = re.compile(r"^.+-70b_(A100|H100)_tp4$")
CONFIG_MODEL_SIZE_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")


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


def _write_csv(
    path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]
) -> None:
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


_extract_norm_for_eval = extract_norm_params


_resolve_checkpoint_norm_gmm_paths = _shared_resolve_checkpoint_norm_gmm_paths
_resolve_throughput = _shared_resolve_throughput
_resolve_experimental_paths = _shared_resolve_experimental_paths


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


_load_model = load_gru_classifier


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
    del non_gpu_overhead_w
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
    matched_flat_gpu = matched_flat.copy()
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
            payload = load_json(path)
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


def _resolve_methods(splitwise_mode: str) -> List[str]:
    mode = str(splitwise_mode).strip().lower()
    if mode == "strict":
        return list(DEFAULT_METHODS)
    if mode in {"fitted", "both"}:
        raise ValueError(SPLITWISE_REMOVED_MESSAGE)
    raise ValueError("splitwise_mode must be 'strict'")


def _plot_facility_traces(
    *,
    out_path: str,
    facility_kw_by_method: Mapping[str, np.ndarray],
    dt: float,
    n_nodes: int,
    config_id: str,
    lambda_req_per_s_per_node: float,
    methods: Sequence[str],
) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10.5, 4))
    duration_min = 0.0
    for method in methods:
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
    ax.set_ylim(0.0, 300.0)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Facility power (kW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_load_duration_curves(
    *,
    out_path: str,
    facility_kw_by_method: Mapping[str, np.ndarray],
    methods: Sequence[str],
) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for method in methods:
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


def run_baselines_facility(
    *,
    run_manifest: str = "results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1/manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    ar1_params_dir: str = "results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/ar1_params",
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
    gpu_tdp_w: Optional[float] = None,
    pue: float = 1.3,
    non_gpu_overhead_w: float = 1000.0,
    facility_power_mode: str = "gpu_sum_only",
    base_seed: int = 42,
    device: str = "auto",
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    ours_std_scale: float = 1.0,
    ours_logit_temperature: float = 1.0,
    splitwise_perf_model_csv: str = "data/perf_model.csv",
    splitwise_source_model: str = "llama-3-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: Optional[int] = None,
    splitwise_style_lut_mode: str = SPLITWISE_STYLE_LUT_V1,
    splitwise_mode: str = "strict",
    traffic_model: str = "poisson",
    burst_rate_per_min: float = 2.0,
    burst_mean_duration_s: float = 20.0,
    burst_peak_scale: float = 6.0,
    burst_background_sigma: float = 0.35,
    burst_node_scale_sigma: float = 0.2,
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
    resolved_gpu_tdp_w = _resolve_per_gpu_chip_tdp_w(config_id, gpu_tdp_w)
    if float(pue) <= 0:
        raise ValueError("pue must be > 0")
    if float(non_gpu_overhead_w) < 0:
        raise ValueError("non_gpu_overhead_w must be >= 0")
    facility_mode = str(facility_power_mode).strip().lower()
    if facility_mode not in {"gpu_sum_only", "legacy_pue_overhead"}:
        raise ValueError(
            "facility_power_mode must be one of {'gpu_sum_only', 'legacy_pue_overhead'}"
        )
    facility_mode_note = (
        "gpu_sum_only is deprecated; total facility power now always includes "
        "per-node non-GPU overhead before PUE."
        if facility_mode == "gpu_sum_only"
        else (
            "Total facility power = (sum of GPU power + per-node non-GPU overhead) "
            "* PUE."
        )
    )
    if float(ours_std_scale) <= 0:
        raise ValueError("ours_std_scale must be > 0")
    if float(ours_logit_temperature) <= 0:
        raise ValueError("ours_logit_temperature must be > 0")
    splitwise_style_lut_mode = normalize_splitwise_style_lut_mode(
        splitwise_style_lut_mode
    )
    methods = _resolve_methods(splitwise_mode)
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

    run_manifest_payload = load_json(run_manifest)
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
    norm_payload = load_json(norm_path)
    norm_cfg = _extract_norm_for_eval(norm_payload)
    gmm_cfg = load_gmm_params_json_dict(load_json(gmm_path))
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

    throughput_payload = load_json(throughput_db)
    throughput = _resolve_throughput(throughput_payload, config_id)
    experimental_payload = load_json(experimental_manifest)
    experimental_base = str(Path(experimental_manifest).resolve().parent)
    dataset_path, split_path = _resolve_experimental_paths(
        experimental_payload,
        config_id=config_id,
        experimental_base=experimental_base,
    )
    split_payload = load_json(split_path)
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]
    test_indices = [int(x) for x in split_payload.get("test_indices", [])]
    if len(train_indices) == 0 and len(test_indices) == 0:
        raise ValueError("both train and test splits are empty")

    pair_map = _load_pair_manifest_map(pair_manifest_csv)
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
    train_power_flat_gpu = train_power_flat.copy()
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
    requested_splitwise_tp = (
        int(splitwise_source_tp) if splitwise_source_tp is not None else int(tp_gpus)
    )
    splitwise_strict_lut_params = build_splitwise_style_lut_params(
        config_id=config_id,
        perf_model_csv=splitwise_perf_model_csv,
        train_power_flat=train_power_flat_gpu,
        splitwise_source_model=splitwise_source_model,
        splitwise_source_hardware=splitwise_source_hardware,
        splitwise_source_tp=int(requested_splitwise_tp),
        splitwise_style_lut_mode=splitwise_style_lut_mode,
        n_gpus_per_node=int(n_gpus_for_gpu_power),
        per_gpu_tdp_cap_w=float(resolved_gpu_tdp_w),
    )
    splitwise_meta: Dict[str, Dict[str, object]] = {
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
                splitwise_strict_lut_params.get(
                    "splitwise_source_resolved_hardware", ""
                )
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
            "splitwise_extrapolation_events": 0,
            "splitwise_power_clamp_events": 0,
            "splitwise_max_batch_tokens_seen": 0.0,
        }
    }

    def _splitwise_meta_for_method(method: str) -> Dict[str, object]:
        return splitwise_meta.get(
            str(method),
            {
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
            },
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

    traces_by_method: Dict[str, List[np.ndarray]] = {method: [] for method in methods}
    method_errors: Dict[str, List[str]] = {method: [] for method in methods}
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
            for method in methods:
                method_errors[method].append(f"node={node_idx}:invalid_feature_shape")
            continue
        n_eval = int(min(t_horizon, features_norm.shape[0]))
        feat_eval = features_norm[:n_eval]
        p0 = float(rng_node.choice(train_power_flat))

        for method in methods:
            try:
                if method == "tdp":
                    # TDP baseline here is GPU-only; non-GPU overhead is added once at facility aggregation.
                    pred = generate_tdp(
                        n_eval,
                        {
                            "tdp_node": float(tp_gpus) * float(resolved_gpu_tdp_w),
                            "non_gpu_power_w": 0.0,
                        },
                    )
                elif method == "mean":
                    pred = generate_mean(n_eval, {}, mean_train_power_flat_gpu)
                elif method == "splitwise_strict":
                    pred, strict_meta = generate_splitwise_style_lut_trace(
                        requests=requests,
                        T=n_eval,
                        dt=float(dt),
                        config={
                            "config_id": config_id,
                            "tp": int(tp_gpus),
                            "n_gpus_per_node": int(n_gpus_for_gpu_power),
                            "gpu_tdp_w": float(resolved_gpu_tdp_w),
                            "non_gpu_power_w": 0.0,
                        },
                        lut_params=splitwise_strict_lut_params,
                    )
                    splitwise_meta["splitwise_strict"][
                        "splitwise_extrapolation_events"
                    ] = int(
                        splitwise_meta["splitwise_strict"].get(
                            "splitwise_extrapolation_events", 0
                        )
                    ) + int(strict_meta.get("splitwise_extrapolation_events", 0))
                    splitwise_meta["splitwise_strict"][
                        "splitwise_power_clamp_events"
                    ] = int(
                        splitwise_meta["splitwise_strict"].get(
                            "splitwise_power_clamp_events", 0
                        )
                    ) + int(strict_meta.get("splitwise_power_clamp_events", 0))
                    splitwise_meta["splitwise_strict"][
                        "splitwise_max_batch_tokens_seen"
                    ] = float(
                        max(
                            float(
                                splitwise_meta["splitwise_strict"].get(
                                    "splitwise_max_batch_tokens_seen", 0.0
                                )
                            ),
                            float(
                                strict_meta.get("splitwise_max_batch_tokens_seen", 0.0)
                            ),
                        )
                    )
                    splitwise_meta["splitwise_strict"][
                        "splitwise_power_support_status"
                    ] = str(
                        strict_meta.get(
                            "splitwise_power_support_status",
                            splitwise_meta["splitwise_strict"].get(
                                "splitwise_power_support_status", ""
                            ),
                        )
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
                    pred = np.asarray(pred_node, dtype=np.float64).reshape(-1)
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
    for method in methods:
        method_splitwise_meta = _splitwise_meta_for_method(method)
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
                    "gpu_tdp_w": float(resolved_gpu_tdp_w),
                    "facility_power_mode": facility_mode,
                    "power_domain": "total_facility",
                    "facility_power_mode_note": facility_mode_note,
                    "pue": float(pue),
                    "non_gpu_overhead_w": float(non_gpu_overhead_w),
                    "mean_pool_source": mean_pool_source,
                    "mean_pool_target_rate": float(mean_pool_target_rate),
                    "mean_pool_selected_rate": float(mean_pool_selected_rate),
                    "mean_pool_num_traces": int(mean_pool_num_traces),
                    "mean_pool_num_samples": int(mean_pool_num_samples),
                    "ours_std_scale": float(ours_std_scale),
                    "ours_logit_temperature": float(ours_logit_temperature),
                    "splitwise_source_model": str(
                        method_splitwise_meta["splitwise_source_model"]
                    ),
                    "splitwise_source_hardware": str(
                        method_splitwise_meta["splitwise_source_hardware"]
                    ),
                    "splitwise_source_tp": int(
                        method_splitwise_meta["splitwise_source_tp"]
                    ),
                    "splitwise_style_lut_mode": str(
                        method_splitwise_meta["splitwise_style_lut_mode"]
                    ),
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
        node_total_stack = node_stack + float(non_gpu_overhead_w)
        facility_w = float(pue) * np.sum(node_total_stack, axis=0)
        facility_kw_by_method[method] = facility_w / 1000.0
        metrics = _compute_facility_metrics(
            facility_w=facility_w,
            node_stack_w=node_total_stack,
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
                "gpu_tdp_w": float(resolved_gpu_tdp_w),
                "facility_power_mode": facility_mode,
                "power_domain": "total_facility",
                "facility_power_mode_note": facility_mode_note,
                "pue": float(pue),
                "non_gpu_overhead_w": float(non_gpu_overhead_w),
                "mean_pool_source": mean_pool_source,
                "mean_pool_target_rate": float(mean_pool_target_rate),
                "mean_pool_selected_rate": float(mean_pool_selected_rate),
                "mean_pool_num_traces": int(mean_pool_num_traces),
                "mean_pool_num_samples": int(mean_pool_num_samples),
                "ours_std_scale": float(ours_std_scale),
                "ours_logit_temperature": float(ours_logit_temperature),
                "splitwise_source_model": str(
                    method_splitwise_meta["splitwise_source_model"]
                ),
                "splitwise_source_hardware": str(
                    method_splitwise_meta["splitwise_source_hardware"]
                ),
                "splitwise_source_tp": int(
                    method_splitwise_meta["splitwise_source_tp"]
                ),
                "splitwise_style_lut_mode": str(
                    method_splitwise_meta["splitwise_style_lut_mode"]
                ),
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
        "power_domain",
        "facility_power_mode_note",
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

    if len(facility_kw_by_method) > 0:
        _plot_facility_traces(
            out_path=traces_pdf,
            facility_kw_by_method=facility_kw_by_method,
            dt=float(dt),
            n_nodes=int(n_nodes),
            config_id=config_id,
            lambda_req_per_s_per_node=float(lambda_req_per_s_per_node),
            methods=methods,
        )
        _plot_load_duration_curves(
            out_path=ldc_pdf,
            facility_kw_by_method=facility_kw_by_method,
            methods=methods,
        )

    return {
        "out_csv": out_csv,
        "traces_pdf": traces_pdf,
        "ldc_pdf": ldc_pdf,
        "rows": rows,
        "splitwise_mode": str(splitwise_mode),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Facility-level baseline comparison (TDP / Mean / Splitwise Strict "
            "Emulation / Ours) reported as total facility power."
        )
    )
    parser.add_argument(
        "--run-manifest",
        default="results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    )
    parser.add_argument(
        "--experimental-manifest",
        default="results/experimental_continuous_v1/manifest.json",
    )
    parser.add_argument(
        "--throughput-db", default="model/config/throughput_database.json"
    )
    parser.add_argument(
        "--pair-manifest-csv", default="results/stage0/pair_manifest.csv"
    )
    parser.add_argument(
        "--ar1-params-dir",
        default="results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/ar1_params",
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
        default=None,
        help=(
            "Per-GPU TDP cap (W) used by the TDP baseline and Splitwise scaling. "
            "Defaults to 400 W for A100 configs and 700 W for H100 configs."
        ),
    )
    parser.add_argument("--pue", type=float, default=1.3)
    parser.add_argument(
        "--non-gpu-overhead-w",
        type=float,
        default=0.0,
        help=(
            "Per-node constant watts added to GPU-only traces when computing total "
            "node and facility power."
        ),
    )
    parser.add_argument(
        "--facility-power-mode",
        choices=["gpu_sum_only", "legacy_pue_overhead"],
        default="legacy_pue_overhead",
        help=(
            "legacy_pue_overhead: total facility power = (sum GPU power + per-node "
            "non-GPU overhead) * PUE. gpu_sum_only is a deprecated alias for the "
            "same additive total-power semantics."
        ),
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
    parser.add_argument("--splitwise-source-model", default="llama-3-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=None)
    parser.add_argument(
        "--splitwise-mode",
        choices=["strict"],
        default="strict",
        help="Only the strict Splitwise emulation path is maintained.",
    )
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
        splitwise_mode=args.splitwise_mode,
        traffic_model=args.traffic_model,
        burst_rate_per_min=args.burst_rate_per_min,
        burst_mean_duration_s=args.burst_mean_duration_s,
        burst_peak_scale=args.burst_peak_scale,
        burst_background_sigma=args.burst_background_sigma,
        burst_node_scale_sigma=args.burst_node_scale_sigma,
    )
    print("[run_baselines_facility] Done")
    print(f"  metrics_csv : {result['out_csv']}")
    print(f"  traces_pdf  : {result['traces_pdf']}")
    print(f"  ldc_pdf     : {result['ldc_pdf']}")
    print(f"  splitwise_mode: {result['splitwise_mode']}")


if __name__ == "__main__":
    main()
