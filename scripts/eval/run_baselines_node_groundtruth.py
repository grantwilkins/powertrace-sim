#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import seaborn as sns

from model.classifiers.metrics import compute_power_metrics
from model.utils.io import load_json
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
    load_gmm_params_json_dict,
)
from scripts.eval.run_baselines_node import (
    _build_requests_from_stage0_json,
    _ensure_dir,
    _extract_norm_for_eval,
    _is_70b_tp4_config,
    _is_moe_config,
    _load_model,
    _load_or_estimate_ar1_params,
    _load_pair_manifest_map,
    _nanmedian,
    _resolve_checkpoint_norm_gmm_paths,
    _resolve_device,
    _resolve_experimental_paths,
    _resolve_per_gpu_chip_tdp_w,
    _resolve_throughput,
    _write_csv,
)

CONSTANT_METHODS = {"tdp", "mean"}
METRIC_KEYS = (
    "ks_stat",
    "acf_r2",
    "nrmse",
    "p95_error_pct",
    "p99_error_pct",
    "delta_energy_pct",
)

STYLE = {
    "ground_truth": {
        "label": "Measured",
        "color": "#000000",
        "linestyle": "-",
        "linewidth": 2.8,
        "alpha": 1.0,
        "zorder": 1,
    },
    "tdp": {
        "label": "TDP",
        "color": "#D55E00",
        "linestyle": "--",
        "linewidth": 2.3,
        "alpha": 0.7,
        "zorder": 6,
    },
    "mean": {
        "label": "Mean",
        "color": "#E69F00",
        "linestyle": "--",
        "linewidth": 2.1,
        "alpha": 0.7,
        "zorder": 5,
    },
    "splitwise_strict": {
        "label": "LUT-based",
        "color": "#E50808",  # Stanford Cardinal red
        "linestyle": ":",
        "linewidth": 2.4,
        "alpha": 0.7,
        "zorder": 6,
    },
    "ours": {
        "label": "Ours",
        "color": "#006F54",  # forest green
        "linestyle": "-",
        "linewidth": 2.4,
        "alpha": 0.7,
        "zorder": 4,
    },
}


def _parse_rate(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _select_test_trace_index(
    *,
    test_indices: List[int],
    n_total: int,
    rate_arr: np.ndarray,
    target_rate: float,
    explicit_trace_index: Optional[int],
) -> Dict[str, object]:
    if len(test_indices) == 0:
        raise ValueError("empty_test_split")
    if explicit_trace_index is not None:
        idx = int(explicit_trace_index)
        if idx < 0 or idx >= n_total:
            raise ValueError(f"explicit trace index out of range: {idx}")
        if idx not in set(test_indices):
            raise ValueError(
                f"explicit trace index is not in held-out test split: {idx}"
            )
        selected_rate = (
            _parse_rate(rate_arr[idx]) if rate_arr.size > idx else float("nan")
        )
        return {
            "trace_index": idx,
            "selected_rate": selected_rate,
            "selection": "explicit_test_trace_index",
        }

    finite_candidates: List[tuple[int, float]] = []
    if rate_arr.size > 0:
        for idx in test_indices:
            if idx < 0 or idx >= min(n_total, int(rate_arr.size)):
                continue
            r = _parse_rate(rate_arr[idx])
            if np.isfinite(r):
                finite_candidates.append((idx, float(r)))
    if len(finite_candidates) == 0:
        idx0 = int(test_indices[0])
        return {
            "trace_index": idx0,
            "selected_rate": float("nan"),
            "selection": "first_test_trace_no_finite_rate",
        }

    target = float(target_rate)
    selected = min(finite_candidates, key=lambda pair: abs(pair[1] - target))
    return {
        "trace_index": int(selected[0]),
        "selected_rate": float(selected[1]),
        "selection": "closest_rate_in_test_split",
    }


def _select_plot_prediction(
    preds_by_seed: List[np.ndarray], metrics_by_seed: List[Dict[str, float]]
) -> np.ndarray:
    if len(preds_by_seed) == 0:
        raise ValueError("no predictions available for plotting")
    if len(preds_by_seed) == 1:
        return preds_by_seed[0]
    nrmse_vals = np.asarray(
        [float(row["nrmse"]) for row in metrics_by_seed], dtype=np.float64
    )
    if np.all(np.isfinite(nrmse_vals)):
        med = float(np.median(nrmse_vals))
        best_idx = int(np.argmin(np.abs(nrmse_vals - med)))
        return preds_by_seed[best_idx]
    return preds_by_seed[0]


def _resolve_methods(splitwise_mode: str) -> List[str]:
    mode = str(splitwise_mode).strip().lower()
    if mode == "strict":
        return ["ours"]
    if mode in {"fitted", "both"}:
        raise ValueError(SPLITWISE_REMOVED_MESSAGE)
    raise ValueError("splitwise_mode must be 'strict'")


def _plot_trace_overlay(
    *,
    out_pdf: str,
    gt_w: np.ndarray,
    pred_by_method: Dict[str, np.ndarray],
    dt: float,
    config_id: str,
    target_rate: float,
    selected_rate: float,
    trace_index: int,
    methods: List[str],
) -> None:
    _ensure_dir(os.path.dirname(out_pdf) or ".")
    gt = np.asarray(gt_w, dtype=np.float64).reshape(-1)
    t_min = (np.arange(gt.size, dtype=np.float64) * float(dt)) / 60.0
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 4))
    gt_style = STYLE["ground_truth"]
    ax.plot(
        t_min,
        gt / 1000.0,
        label=gt_style["label"],
        color=gt_style["color"],
        linestyle=gt_style["linestyle"],
        linewidth=float(gt_style.get("linewidth", 2.2)),
        alpha=gt_style["alpha"],
        zorder=float(gt_style.get("zorder", 5)),
    )
    for method in methods:
        if method not in pred_by_method:
            continue
        pred = np.asarray(pred_by_method[method], dtype=np.float64).reshape(-1)
        n = int(min(gt.size, pred.size))
        style = STYLE[method]
        ax.plot(
            t_min[:n],
            (pred[:n] / 1000.0),
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=float(style.get("linewidth", 2.0)),
            alpha=style["alpha"],
            zorder=float(style.get("zorder", 3)),
        )

    sel_rate_label = f"{selected_rate:.3f}" if np.isfinite(selected_rate) else "N/A"
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("GPU Power (kW)")
    ax.grid(True, alpha=0.5)
    # ax.legend(bbox_to_anchor=(1.01, 0.75), loc="upper left", frameon=False)
    ax.set_ylim(0.0, 2.0)
    ax.set_xlim(0.0, 10.0)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def run_baselines_node_groundtruth(
    *,
    run_manifest: str = "results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1/manifest.json",
    throughput_db: str = "model/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    ar1_params_dir: str = "results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/ar1_params",
    config_id: str = "llama-3-70b_A100_tp4",
    target_rate: float = 4.0,
    test_trace_index: Optional[int] = None,
    tp_gpus: int = 4,
    n_gpus_for_gpu_power: int = 4,
    gpu_tdp_w: Optional[float] = None,
    non_gpu_overhead_w: float = 0.0,
    num_seeds: int = 5,
    base_seed: int = 42,
    device: str = "auto",
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    ours_std_scale: float = 1.0,
    ours_logit_temperature: float = 1.0,
    acf_max_lag: int = 50,
    splitwise_perf_model_csv: str = "data/perf_model.csv",
    splitwise_source_model: str = "llama-3-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: Optional[int] = None,
    splitwise_style_lut_mode: str = SPLITWISE_STYLE_LUT_V1,
    splitwise_mode: str = "strict",
    out_csv: str = "results/eval_paper/baselines_node_groundtruth_metrics.csv",
    out_plot_pdf: str = "figures/baselines_node_groundtruth_trace.pdf",
) -> Dict[str, object]:
    if not _is_70b_tp4_config(config_id):
        raise ValueError(
            f"config_id '{config_id}' must match *-70b_*_tp4 for this experiment."
        )
    if int(tp_gpus) <= 0:
        raise ValueError("tp_gpus must be >= 1")
    if int(n_gpus_for_gpu_power) < int(tp_gpus):
        raise ValueError("n_gpus_for_gpu_power must be >= tp_gpus")
    resolved_gpu_tdp_w = _resolve_per_gpu_chip_tdp_w(config_id, gpu_tdp_w)
    if float(non_gpu_overhead_w) < 0.0:
        raise ValueError("non_gpu_overhead_w must be >= 0")
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    if float(ours_std_scale) <= 0.0:
        raise ValueError("ours_std_scale must be > 0")
    if float(ours_logit_temperature) <= 0.0:
        raise ValueError("ours_logit_temperature must be > 0")
    splitwise_style_lut_mode = normalize_splitwise_style_lut_mode(
        splitwise_style_lut_mode
    )
    methods = _resolve_methods(splitwise_mode)

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
        dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)
    if dt_arr.size == 0:
        raise ValueError("dataset_dt_missing")
    dt = float(dt_arr[0])
    if (not np.isfinite(dt)) or dt <= 0.0:
        raise ValueError(f"invalid_dt:{dt}")

    n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))
    train_power_pool: List[np.ndarray] = []
    train_power_traces_node: List[np.ndarray] = []
    for idx in train_indices:
        if idx < 0 or idx >= n_total:
            continue
        p = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if p.size == 0:
            continue
        train_power_pool.append(p.astype(np.float64))
        train_power_traces_node.append(p.astype(np.float64))
    if len(train_power_pool) == 0:
        raise ValueError("empty_training_pool")
    train_power_flat_node = np.concatenate(train_power_pool, axis=0).astype(np.float64)
    train_power_flat_gpu = train_power_flat_node.copy()

    trace_sel = _select_test_trace_index(
        test_indices=test_indices,
        n_total=n_total,
        rate_arr=rate_arr,
        target_rate=float(target_rate),
        explicit_trace_index=test_trace_index,
    )
    trace_index = int(trace_sel["trace_index"])
    selected_rate = float(trace_sel["selected_rate"])
    selection_mode = str(trace_sel["selection"])

    power = np.asarray(power_arr[trace_index], dtype=np.float64).reshape(-1)
    if power.size < 2:
        raise ValueError(f"selected trace {trace_index} too short")
    pair_key = str(pair_key_arr[trace_index])
    json_path = pair_map.get(pair_key)
    if json_path is None:
        raise ValueError(f"missing matched request json for pair_key '{pair_key}'")

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
    ar1_params = None
    if _is_moe_config(config_id):
        ar1_params = _load_or_estimate_ar1_params(
            config_id=config_id,
            gmm_params=gmm_cfg,
            train_power_traces=train_power_traces_node,
            ar1_params_dir=ar1_params_dir,
        )

    gt_node = power[1:].astype(np.float64)
    requests = _build_requests_from_stage0_json(
        json_path,
        power_start_epoch_s=float(power_start_arr[trace_index]),
        trace_duration_s=float(power.size * dt),
        dt=dt,
    )
    feat = build_rollout_features_from_requests(
        requests=requests,
        throughput=throughput,
        norm=norm_cfg,
        T=int(gt_node.size),
        dt=dt,
        feature_set=feature_set,
    )
    features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
    if features_norm.ndim != 2:
        raise ValueError("invalid feature shape from replay requests")
    n_eval = int(min(gt_node.size, features_norm.shape[0]))
    if n_eval <= 0:
        raise ValueError("empty aligned horizon")

    gt_eval_gpu = gt_node[:n_eval].astype(np.float64)
    feat_eval = features_norm[:n_eval]
    p0_node = float(gt_node[0]) if gt_node.size > 0 else float(power[0])
    ours_cfg = {
        "device": str(resolved_device),
        "p0": p0_node,
        "decode_mode": decode_mode,
        "median_filter_window": int(median_filter_window),
        "std_scale": float(ours_std_scale),
        "logit_temperature": float(ours_logit_temperature),
        "clamp_range": (norm_cfg["power_min"], norm_cfg["power_max"]),
        "ar1_params": ar1_params,
    }

    pred_by_method: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    for method in methods:
        seeds = (
            [int(base_seed)]
            if method in CONSTANT_METHODS or method in {"splitwise_strict"}
            else [int(base_seed) + i for i in range(int(num_seeds))]
        )
        seed_preds_gpu: List[np.ndarray] = []
        seed_metrics: List[Dict[str, float]] = []
        for seed in seeds:
            if method == "tdp":
                pred_node = generate_tdp(
                    n_eval,
                    {
                        "tdp_node": float(tp_gpus) * float(resolved_gpu_tdp_w),
                        "non_gpu_power_w": 0.0,
                    },
                )
                pred_gpu = np.asarray(pred_node, dtype=np.float64).reshape(-1)[:n_eval]
            elif method == "mean":
                pred_gpu = generate_mean(n_eval, {}, gt_eval_gpu)
                pred_gpu = np.asarray(pred_gpu, dtype=np.float64).reshape(-1)[:n_eval]
            elif method == "splitwise_strict":
                pred_node, strict_meta = generate_splitwise_style_lut_trace(
                    requests=requests,
                    T=n_eval,
                    dt=dt,
                    config={
                        "config_id": config_id,
                        "tp": int(tp_gpus),
                        "n_gpus_per_node": int(n_gpus_for_gpu_power),
                        "gpu_tdp_w": float(resolved_gpu_tdp_w),
                        "non_gpu_power_w": 0.0,
                    },
                    lut_params=splitwise_strict_lut_params,
                )
                splitwise_meta["splitwise_strict"]["splitwise_extrapolation_events"] = (
                    int(
                        splitwise_meta["splitwise_strict"].get(
                            "splitwise_extrapolation_events", 0
                        )
                    )
                    + int(strict_meta.get("splitwise_extrapolation_events", 0))
                )
                splitwise_meta["splitwise_strict"]["splitwise_power_clamp_events"] = (
                    int(
                        splitwise_meta["splitwise_strict"].get(
                            "splitwise_power_clamp_events", 0
                        )
                    )
                    + int(strict_meta.get("splitwise_power_clamp_events", 0))
                )
                splitwise_meta["splitwise_strict"][
                    "splitwise_max_batch_tokens_seen"
                ] = float(
                    max(
                        float(
                            splitwise_meta["splitwise_strict"].get(
                                "splitwise_max_batch_tokens_seen", 0.0
                            )
                        ),
                        float(strict_meta.get("splitwise_max_batch_tokens_seen", 0.0)),
                    )
                )
                splitwise_meta["splitwise_strict"]["splitwise_power_support_status"] = (
                    str(
                        strict_meta.get(
                            "splitwise_power_support_status",
                            splitwise_meta["splitwise_strict"].get(
                                "splitwise_power_support_status", ""
                            ),
                        )
                    )
                )
                pred_gpu = np.asarray(pred_node, dtype=np.float64).reshape(-1)[:n_eval]
            elif method == "ours":
                pred_node = generate_ours(
                    feat_eval,
                    ours_cfg,
                    model,
                    gmm_cfg,
                    rng=np.random.default_rng(seed),
                )
                pred_gpu = np.asarray(pred_node, dtype=np.float64).reshape(-1)[:n_eval]
            else:
                raise ValueError(f"Unknown method: {method}")

            metrics = compute_power_metrics(
                gt_eval_gpu, pred_gpu, dt=dt, acf_max_lag=int(acf_max_lag)
            )
            if method in CONSTANT_METHODS:
                metrics["acf_r2"] = float("nan")
            seed_preds_gpu.append(pred_gpu)
            seed_metrics.append({key: float(metrics[key]) for key in METRIC_KEYS})

        pred_by_method[method] = _select_plot_prediction(seed_preds_gpu, seed_metrics)
        aggregated = {
            key: _nanmedian([row[key] for row in seed_metrics]) for key in METRIC_KEYS
        }
        if method in CONSTANT_METHODS:
            aggregated["acf_r2"] = float("nan")
        method_splitwise_meta = splitwise_meta.get(
            method,
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
        rows.append(
            {
                "config_id": config_id,
                "trace_index": int(trace_index),
                "trace_pair_key": pair_key,
                "target_rate": float(target_rate),
                "selected_rate": float(selected_rate),
                "selection_mode": selection_mode,
                "power_domain": "gpu_only",
                "tp_gpus": int(tp_gpus),
                "n_gpus_for_gpu_power": int(n_gpus_for_gpu_power),
                "gpu_tdp_w": float(resolved_gpu_tdp_w),
                "non_gpu_overhead_w": float(non_gpu_overhead_w),
                "method": method,
                "num_seeds": int(len(seeds)),
                "n_timesteps": int(n_eval),
                "dt": float(dt),
                "ks_stat": float(aggregated["ks_stat"]),
                "acf_r2": float(aggregated["acf_r2"]),
                "nrmse": float(aggregated["nrmse"]),
                "p95_error_pct": float(aggregated["p95_error_pct"]),
                "p99_error_pct": float(aggregated["p99_error_pct"]),
                "delta_energy_pct": float(aggregated["delta_energy_pct"]),
                "acf_note": "N/A_constant_trace" if method in CONSTANT_METHODS else "",
                "decode_mode": decode_mode if method == "ours" else "",
                "median_filter_window": int(median_filter_window)
                if method == "ours"
                else 0,
                "ours_std_scale": float(ours_std_scale) if method == "ours" else 1.0,
                "ours_logit_temperature": float(ours_logit_temperature)
                if method == "ours"
                else 1.0,
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
            }
        )

    fieldnames = [
        "config_id",
        "trace_index",
        "trace_pair_key",
        "target_rate",
        "selected_rate",
        "selection_mode",
        "power_domain",
        "tp_gpus",
        "n_gpus_for_gpu_power",
        "gpu_tdp_w",
        "non_gpu_overhead_w",
        "method",
        "num_seeds",
        "n_timesteps",
        "dt",
        "ks_stat",
        "acf_r2",
        "nrmse",
        "p95_error_pct",
        "p99_error_pct",
        "delta_energy_pct",
        "acf_note",
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
    ]
    _write_csv(out_csv, rows, fieldnames)
    _plot_trace_overlay(
        out_pdf=out_plot_pdf,
        gt_w=gt_eval_gpu,
        pred_by_method=pred_by_method,
        dt=dt,
        config_id=config_id,
        target_rate=float(target_rate),
        selected_rate=float(selected_rate),
        trace_index=int(trace_index),
        methods=methods,
    )
    return {
        "out_csv": out_csv,
        "out_plot_pdf": out_plot_pdf,
        "rows": rows,
        "trace_index": int(trace_index),
        "selected_rate": float(selected_rate),
        "selection_mode": selection_mode,
        "splitwise_mode": str(splitwise_mode),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Node-level held-out replay (GPU-only): measured GPU power vs GPU-only "
            "TDP, replay mean, Splitwise strict emulation, and Ours."
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
    parser.add_argument("--throughput-db", default="model/throughput_database.json")
    parser.add_argument(
        "--pair-manifest-csv", default="results/stage0/pair_manifest.csv"
    )
    parser.add_argument("--config-id", default="llama-3-70b_A100_tp4")
    parser.add_argument(
        "--target-rate",
        type=float,
        default=0.25,
        help="Select held-out trace with rate closest to this value.",
    )
    parser.add_argument(
        "--test-trace-index",
        type=int,
        default=None,
        help="Optional explicit held-out trace index override.",
    )
    parser.add_argument(
        "--tp-gpus",
        type=int,
        default=4,
        help="Number of active GPUs for the TDP baseline.",
    )
    parser.add_argument(
        "--n-gpus-for-gpu-power",
        type=int,
        default=4,
        help="Total GPUs represented in GPU-only power accounting.",
    )
    parser.add_argument(
        "--gpu-tdp-w",
        type=float,
        default=None,
        help=(
            "Per-GPU TDP wattage used for the TDP baseline and Splitwise scaling. "
            "Defaults to 400 W for A100 configs and 700 W for H100 configs."
        ),
    )
    parser.add_argument(
        "--non-gpu-overhead-w",
        type=float,
        default=1000.0,
        help=(
            "Per-node constant overhead assumption used for external total-power "
            "accounting; not subtracted from the GPU-only traces evaluated here."
        ),
    )
    parser.add_argument(
        "--ar1-params-dir",
        default="results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/ar1_params",
        help="Directory containing AR(1) params JSON files (used only for MoE configs).",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--decode-mode", choices=["stochastic", "argmax"], default="stochastic"
    )
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--ours-std-scale", type=float, default=1.0)
    parser.add_argument("--ours-logit-temperature", type=float, default=1.0)
    parser.add_argument("--acf-max-lag", type=int, default=50)
    parser.add_argument("--splitwise-perf-model-csv", default="data/perf_model.csv")
    parser.add_argument("--splitwise-source-model", default="llama-3-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=None)
    parser.add_argument("--splitwise-style-lut-mode", default=SPLITWISE_STYLE_LUT_V1)
    parser.add_argument(
        "--splitwise-mode",
        choices=["strict"],
        default="strict",
        help="Only the strict Splitwise emulation path is maintained.",
    )
    parser.add_argument(
        "--out-csv", default="results/eval_paper/baselines_node_groundtruth_metrics.csv"
    )
    parser.add_argument(
        "--out-plot-pdf", default="figures/baselines_node_groundtruth_trace.pdf"
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_baselines_node_groundtruth(
        run_manifest=args.run_manifest,
        experimental_manifest=args.experimental_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        ar1_params_dir=args.ar1_params_dir,
        config_id=args.config_id,
        target_rate=args.target_rate,
        test_trace_index=args.test_trace_index,
        tp_gpus=args.tp_gpus,
        n_gpus_for_gpu_power=args.n_gpus_for_gpu_power,
        gpu_tdp_w=args.gpu_tdp_w,
        non_gpu_overhead_w=args.non_gpu_overhead_w,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
        device=args.device,
        decode_mode=args.decode_mode,
        median_filter_window=args.median_filter_window,
        ours_std_scale=args.ours_std_scale,
        ours_logit_temperature=args.ours_logit_temperature,
        acf_max_lag=args.acf_max_lag,
        splitwise_perf_model_csv=args.splitwise_perf_model_csv,
        splitwise_source_model=args.splitwise_source_model,
        splitwise_source_hardware=args.splitwise_source_hardware,
        splitwise_source_tp=args.splitwise_source_tp,
        splitwise_style_lut_mode=args.splitwise_style_lut_mode,
        splitwise_mode=args.splitwise_mode,
        out_csv=args.out_csv,
        out_plot_pdf=args.out_plot_pdf,
    )
    print("[run_baselines_node_groundtruth] Done")
    print(f"  out_csv        : {result['out_csv']}")
    print(f"  out_plot_pdf   : {result['out_plot_pdf']}")
    print(f"  trace_index    : {result['trace_index']}")
    print(f"  selected_rate  : {result['selected_rate']}")
    print(f"  selection_mode : {result['selection_mode']}")
    print(f"  splitwise_mode : {result['splitwise_mode']}")


if __name__ == "__main__":
    main()
