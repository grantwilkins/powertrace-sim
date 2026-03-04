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

from model.classifiers.metrics import compute_power_metrics
from model.scripts.request_data_policy import (
    DEFAULT_ALLOWED_JSON_PREFIX,
    DEFAULT_REQUEST_TIMESTAMP_POLICY,
    REQUEST_TIMESTAMP_POLICIES,
    normalize_request_timestamp_policy,
    request_timestamp_policy_requires_recorded,
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
    load_gmm_params_json_dict,
)
from scripts.eval.run_baselines_node import (
    _build_requests_from_stage0_json,
    _ensure_dir,
    _estimate_splitwise_phase_targets_from_indices,
    _extract_norm_for_eval,
    _is_70b_tp4_config,
    _is_moe_config,
    _load_json,
    _load_model,
    _load_or_estimate_ar1_params,
    _load_pair_manifest_map,
    _nanmedian,
    _resolve_checkpoint_norm_gmm_paths,
    _resolve_device,
    _resolve_experimental_paths,
    _resolve_throughput,
    _write_csv,
)

METHODS = ("tdp", "mean", "splitwise_lut", "ours")
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
        "label": "Measured (GPU-only)",
        "color": "#000000",
        "linestyle": "-",
        "linewidth": 2.2,
        "alpha": 0.9,
    },
    "tdp": {
        "label": "TDP (4x700W)",
        "color": "#d62728",
        "linestyle": "--",
        "linewidth": 1.8,
        "alpha": 0.85,
    },
    "mean": {
        "label": "Mean (Replay)",
        "color": "#ff7f0e",
        "linestyle": "--",
        "linewidth": 1.8,
        "alpha": 0.85,
    },
    "splitwise_lut": {
        "label": "Splitwise LUT",
        "color": "#2ca02c",
        "linestyle": "-.",
        "linewidth": 1.8,
        "alpha": 0.9,
    },
    "ours": {
        "label": "Ours",
        "color": "#1f77b4",
        "linestyle": "-",
        "linewidth": 1.9,
        "alpha": 0.9,
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
) -> None:
    _ensure_dir(os.path.dirname(out_pdf) or ".")
    gt = np.asarray(gt_w, dtype=np.float64).reshape(-1)
    t_min = (np.arange(gt.size, dtype=np.float64) * float(dt)) / 60.0

    fig, ax = plt.subplots(figsize=(10, 5))
    gt_style = STYLE["ground_truth"]
    ax.plot(
        t_min,
        gt / 1000.0,
        label=gt_style["label"],
        color=gt_style["color"],
        linestyle=gt_style["linestyle"],
        linewidth=gt_style["linewidth"],
        alpha=gt_style["alpha"],
    )
    for method in METHODS:
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
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )

    sel_rate_label = f"{selected_rate:.3f}" if np.isfinite(selected_rate) else "N/A"
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("GPU power (kW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_ylim(0.0, 3.0)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def run_baselines_node_groundtruth(
    *,
    run_manifest: str = "results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2/run_manifest.json",
    experimental_manifest: str = "results/experimental_continuous_v1_gru_all/manifest.json",
    throughput_db: str = "model/config/throughput_database.json",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    ar1_params_dir: str = "results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/ar1_params",
    config_id: str = "deepseek-r1-distill-70b_H100_tp4",
    target_rate: float = 0.25,
    test_trace_index: Optional[int] = None,
    tp_gpus: int = 4,
    n_gpus_for_gpu_power: int = 4,
    gpu_tdp_w: float = 700.0,
    non_gpu_overhead_w: float = 1000.0,
    num_seeds: int = 5,
    base_seed: int = 42,
    device: str = "auto",
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    ours_std_scale: float = 1.0,
    ours_logit_temperature: float = 1.0,
    acf_max_lag: int = 50,
    splitwise_perf_model_csv: str = "data/perf_model.csv",
    splitwise_source_model: str = "llama2-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: int = 4,
    splitwise_calibration_mode: str = "train_phase_matched_v1",
    out_csv: str = "results/eval_paper/baselines_node_groundtruth_metrics.csv",
    out_plot_pdf: str = "figures/baselines_node_groundtruth_trace.pdf",
    request_timestamp_policy: str = DEFAULT_REQUEST_TIMESTAMP_POLICY,
    allowed_json_prefix: str = DEFAULT_ALLOWED_JSON_PREFIX,
) -> Dict[str, object]:
    if not _is_70b_tp4_config(config_id):
        raise ValueError(
            f"config_id '{config_id}' must match *-70b_*_tp4 for this experiment."
        )
    if int(tp_gpus) <= 0:
        raise ValueError("tp_gpus must be >= 1")
    if int(n_gpus_for_gpu_power) < int(tp_gpus):
        raise ValueError("n_gpus_for_gpu_power must be >= tp_gpus")
    if float(gpu_tdp_w) <= 0.0:
        raise ValueError("gpu_tdp_w must be > 0")
    if float(non_gpu_overhead_w) < 0.0:
        raise ValueError("non_gpu_overhead_w must be >= 0")
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    if float(ours_std_scale) <= 0.0:
        raise ValueError("ours_std_scale must be > 0")
    if float(ours_logit_temperature) <= 0.0:
        raise ValueError("ours_logit_temperature must be > 0")
    if int(splitwise_source_tp) != 4:
        raise ValueError("splitwise_source_tp must be 4 for 70B TP4-only comparison.")
    request_timestamp_policy = normalize_request_timestamp_policy(
        request_timestamp_policy
    )
    require_recorded_timestamps = bool(
        request_timestamp_policy_requires_recorded(request_timestamp_policy)
    )

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
    train_power_flat_gpu = np.clip(
        train_power_flat_node - float(non_gpu_overhead_w), a_min=0.0, a_max=None
    )

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
        require_recorded_timestamps=require_recorded_timestamps,
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

    gt_eval_gpu = np.clip(
        gt_node[:n_eval] - float(non_gpu_overhead_w), a_min=0.0, a_max=None
    )
    feat_eval = features_norm[:n_eval]
    a_raw_eval = np.asarray(feat["A_raw"], dtype=np.float64).reshape(-1)[:n_eval]
    delta_a_raw_eval = np.asarray(feat["delta_A_raw"], dtype=np.float64).reshape(-1)[
        :n_eval
    ]
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
    for method in METHODS:
        seeds = (
            [int(base_seed)]
            if method in CONSTANT_METHODS or method == "splitwise_lut"
            else [int(base_seed) + i for i in range(int(num_seeds))]
        )
        seed_preds_gpu: List[np.ndarray] = []
        seed_metrics: List[Dict[str, float]] = []
        for seed in seeds:
            if method == "tdp":
                pred_node = generate_tdp(
                    n_eval,
                    {
                        "tdp_node": float(tp_gpus) * float(gpu_tdp_w),
                        "non_gpu_power_w": 0.0,
                    },
                )
                pred_gpu = np.asarray(pred_node, dtype=np.float64).reshape(-1)[:n_eval]
            elif method == "mean":
                pred_gpu = generate_mean(n_eval, {}, gt_eval_gpu)
                pred_gpu = np.asarray(pred_gpu, dtype=np.float64).reshape(-1)[:n_eval]
            elif method == "splitwise_lut":
                pred_node = generate_splitwise_lut(
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
                pred_gpu = np.asarray(pred_node, dtype=np.float64).reshape(-1)[:n_eval]
            elif method == "ours":
                pred_node = generate_ours(
                    feat_eval,
                    ours_cfg,
                    model,
                    gmm_cfg,
                    rng=np.random.default_rng(seed),
                )
                pred_gpu = np.clip(
                    np.asarray(pred_node, dtype=np.float64).reshape(-1)[:n_eval]
                    - float(non_gpu_overhead_w),
                    a_min=0.0,
                    a_max=None,
                )
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
                "gpu_tdp_w": float(gpu_tdp_w),
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
        "splitwise_calibration_mode",
        "splitwise_phase_detection_note",
        "splitwise_decode_occupancy_note",
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
    )
    return {
        "out_csv": out_csv,
        "out_plot_pdf": out_plot_pdf,
        "rows": rows,
        "trace_index": int(trace_index),
        "selected_rate": float(selected_rate),
        "selection_mode": selection_mode,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Node-level held-out replay (GPU-only): measured GPU power vs TDP, replay mean, Splitwise LUT, and Ours."
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
    parser.add_argument("--config-id", default="deepseek-r1-distill-70b_H100_tp4")
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
        default=700.0,
        help="Per-GPU TDP wattage used for TDP baseline.",
    )
    parser.add_argument(
        "--non-gpu-overhead-w",
        type=float,
        default=1000.0,
        help="Overhead subtracted from measured node power to obtain GPU-only power.",
    )
    parser.add_argument(
        "--ar1-params-dir",
        default="results/continuous_v1_gmm_bigru_sharegpt_all/kauto_max12_f2_ar1_thresh/ar1_params",
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
    parser.add_argument("--splitwise-source-model", default="llama2-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=4)
    parser.add_argument(
        "--splitwise-calibration-mode", default="train_phase_matched_v1"
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
        splitwise_calibration_mode=args.splitwise_calibration_mode,
        out_csv=args.out_csv,
        out_plot_pdf=args.out_plot_pdf,
        request_timestamp_policy=args.request_timestamp_policy,
        allowed_json_prefix=args.allowed_json_prefix,
    )
    print("[run_baselines_node_groundtruth] Done")
    print(f"  out_csv        : {result['out_csv']}")
    print(f"  out_plot_pdf   : {result['out_plot_pdf']}")
    print(f"  trace_index    : {result['trace_index']}")
    print(f"  selected_rate  : {result['selected_rate']}")
    print(f"  selection_mode : {result['selection_mode']}")


if __name__ == "__main__":
    main()
