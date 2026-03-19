#!/usr/bin/env python3
"""
Experiment 2b: Generate per-node power traces for Azure facility streams.

Pipeline per node:
  requests -> rollout features -> BiGRU logits -> sampling / Splitwise LUT -> power trace
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.azure_defaults import (
    DEFAULT_CONFIG_ID,
    DEFAULT_METHODS_GENERATION,
    DEFAULT_NON_GPU_OVERHEAD_W,
    DEFAULT_SPLITWISE_SOURCE_HARDWARE,
    DEFAULT_SPLITWISE_SOURCE_MODEL,
    DEFAULT_SPLITWISE_SOURCE_TP,
    build_default_paths,
    ensure_dir,
    load_json,
    parse_csv_list,
    write_json,
)
from scripts.eval.baselines import (
    SPLITWISE_REMOVED_MESSAGE,
    SPLITWISE_STYLE_LUT_V1,
    build_splitwise_style_lut_params,
    generate_splitwise_style_lut_trace,
    normalize_splitwise_style_lut_mode,
)
from scripts.eval.facility import FacilityLayout
from scripts.eval.pipeline_utils import (
    build_rollout_features_from_requests,
    estimate_ar1_params,
    extract_norm_params,
    generate_gmm_bigru_trace,
    generate_gmm_bigru_trace_ar1_thresholded,
    load_gmm_params_json_dict,
    load_gru_classifier,
    predict_sorted_gmm_labels_from_params,
    resolve_checkpoint_norm_gmm_paths as _resolve_checkpoint_norm_gmm_paths,
    resolve_experimental_paths as _resolve_experimental_paths,
    resolve_throughput as _resolve_throughput,
)
from scripts.eval.run_baselines_node import _is_moe_config

CONFIG_ID_RE = re.compile(r"^(.+)_(A100|H100)_tp(\d+)$")
ALLOWED_METHODS = {"ours", "splitwise_strict"}


def _validate_config_id(config_id: str) -> None:
    if CONFIG_ID_RE.match(str(config_id).strip()) is None:
        raise ValueError(f"Invalid config_id format: {config_id}")


def _extract_tp_from_config_id(config_id: str) -> int:
    match = CONFIG_ID_RE.match(str(config_id).strip())
    if match is None:
        return int(DEFAULT_SPLITWISE_SOURCE_TP)
    try:
        tp = int(match.group(3))
    except Exception:
        return int(DEFAULT_SPLITWISE_SOURCE_TP)
    return max(1, tp)


def _resolve_device(device: str) -> torch.device:
    if str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


_extract_norm_for_eval = extract_norm_params


def _load_training_bundle(
    *,
    config_id: str,
    experimental_manifest_path: str,
) -> Dict[str, object]:
    manifest = load_json(experimental_manifest_path)
    base = str(Path(experimental_manifest_path).resolve().parent)
    dataset_path, split_path = _resolve_experimental_paths(
        manifest,
        config_id=config_id,
        experimental_base=base,
    )
    split_payload = load_json(split_path)
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]

    with np.load(dataset_path, allow_pickle=True) as data:
        power_arr = np.asarray(data["power"], dtype=object)
    n_total = int(len(power_arr))

    train_traces: List[np.ndarray] = []
    for idx in train_indices:
        if idx < 0 or idx >= n_total:
            continue
        power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if power.size > 0:
            train_traces.append(power.astype(np.float64))

    if len(train_traces) == 0:
        for idx in range(n_total):
            power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
            if power.size > 0:
                train_traces.append(power.astype(np.float64))
                if len(train_traces) >= 3:
                    break

    if len(train_traces) == 0:
        raise ValueError(f"Unable to build training power pool for {config_id}")

    flat = np.concatenate(train_traces, axis=0).astype(np.float64)
    if flat.size == 0:
        raise ValueError("Training power pool is empty after concat")

    return {
        "train_power_traces": train_traces,
        "train_power_flat": flat,
        "train_power_flat_gpu": flat.copy(),
    }


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


def _load_node_requests(path: str) -> List[Dict[str, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Node stream CSV not found: {path}")
    requests: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Node stream CSV missing header: {path}")
        required = {"arrival_time", "n_in", "n_out"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Node stream CSV missing required columns {missing}: {path}")
        for row_idx, row in enumerate(reader, start=2):
            try:
                arrival = float(row["arrival_time"])
                n_in = float(int(float(row["n_in"])))
                n_out = float(int(float(row["n_out"])))
            except Exception as exc:
                raise ValueError(f"Failed parsing {path} row {row_idx}: {exc}") from exc
            if (not np.isfinite(arrival)) or arrival < 0.0:
                raise ValueError(f"Invalid arrival_time at {path}:{row_idx}: {arrival}")
            requests.append(
                {
                    "arrival_time": float(arrival),
                    "input_tokens": float(n_in),
                    "output_tokens": float(n_out),
                }
            )
    requests.sort(key=lambda row: float(row["arrival_time"]))
    return requests


def _list_node_stream_paths(layout: FacilityLayout, node_stream_dir: str) -> List[Tuple[int, int, int, int, str]]:
    out: List[Tuple[int, int, int, int, str]] = []
    for node_id in layout.iter_node_ids():
        row, rack, node = layout.node_id_to_coords(int(node_id))
        path = os.path.join(node_stream_dir, f"node_{row}_{rack}_{node}.csv")
        out.append((int(node_id), int(row), int(rack), int(node), path))
    return out


def _sample_power_from_logits(
    *,
    logits_node: np.ndarray,
    gmm_params: Dict[str, object],
    p0: float,
    seed: int,
    decode_mode: str,
    median_filter_window: int,
    clamp_range: Tuple[float, float],
    std_scale: float,
    use_ar1: bool,
    ar1_params: Optional[Mapping[str, np.ndarray]],
) -> np.ndarray:
    gmm_sampling = dict(gmm_params)
    if abs(float(std_scale) - 1.0) > 1e-12:
        variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
        gmm_sampling["variances"] = np.clip(
            variances * float(std_scale * std_scale),
            a_min=1e-12,
            a_max=None,
        )

    if use_ar1:
        if ar1_params is None:
            raise ValueError("AR(1) requested but ar1_params is None")
        sigma_innov = np.asarray(ar1_params["sigma_innov"], dtype=np.float64).reshape(-1) * float(std_scale)
        sigma_marginal = np.asarray(ar1_params["sigma_marginal"], dtype=np.float64).reshape(-1) * float(std_scale)
        generated = generate_gmm_bigru_trace_ar1_thresholded(
            logits=logits_node,
            gmm_params=gmm_sampling,
            phi=np.asarray(ar1_params["phi"], dtype=np.float64).reshape(-1),
            sigma_innov=sigma_innov,
            sigma_marginal=sigma_marginal,
            p0=float(p0),
            seed=int(seed),
            decode_mode=str(decode_mode),
            median_filter_window=int(median_filter_window),
            phi_threshold=float(ar1_params.get("phi_threshold", 0.3)),
            clamp_range=clamp_range,
        )
    else:
        generated = generate_gmm_bigru_trace(
            logits=logits_node,
            gmm_params=gmm_sampling,
            seed=int(seed),
            decode_mode=str(decode_mode),
            median_filter_window=int(median_filter_window),
            clamp_range=clamp_range,
        )
    return np.asarray(generated["power_w"], dtype=np.float64).reshape(-1)


def _normalize_methods(methods: Sequence[str] | str | None) -> List[str]:
    if methods is None:
        out = list(DEFAULT_METHODS_GENERATION)
    elif isinstance(methods, str):
        out = parse_csv_list(methods)
    else:
        out = [str(x).strip() for x in methods if str(x).strip()]
    if "splitwise_lut" in out:
        raise ValueError(SPLITWISE_REMOVED_MESSAGE)
    invalid = [method for method in out if method not in ALLOWED_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Allowed: {sorted(ALLOWED_METHODS)}")
    dedup: List[str] = []
    for method in out:
        if method not in dedup:
            dedup.append(method)
    if not dedup:
        raise ValueError("No methods selected")
    return dedup


def generate_node_traces(
    *,
    run_manifest: str,
    experimental_manifest: str,
    throughput_db: str,
    ar1_params_dir: str,
    node_stream_dir: str,
    out_root: str,
    config_id: str = DEFAULT_CONFIG_ID,
    methods: Sequence[str] | str | None = None,
    duration_s: float = 86400.0,
    dt: float = 0.25,
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
    batch_size: int = 8,
    base_seed: int = 42,
    device: str = "auto",
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    ours_std_scale: float = 1.0,
    ours_logit_temperature: float = 1.0,
    splitwise_perf_model_csv: str = "data/perf_model.csv",
    splitwise_source_model: str = DEFAULT_SPLITWISE_SOURCE_MODEL,
    splitwise_source_hardware: str = DEFAULT_SPLITWISE_SOURCE_HARDWARE,
    splitwise_source_tp: Optional[int] = None,
    splitwise_style_lut_mode: str = SPLITWISE_STYLE_LUT_V1,
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    tp_gpus: Optional[int] = None,
    n_gpus_per_node: Optional[int] = None,
    non_gpu_overhead_w: float = DEFAULT_NON_GPU_OVERHEAD_W,
    require_recorded_timestamps: bool = True,
) -> Dict[str, object]:
    del pair_manifest_csv
    del require_recorded_timestamps

    _validate_config_id(config_id)
    method_list = _normalize_methods(methods)
    splitwise_style_lut_mode = normalize_splitwise_style_lut_mode(
        splitwise_style_lut_mode
    )
    if float(duration_s) <= 0:
        raise ValueError("duration_s must be > 0")
    if float(dt) <= 0:
        raise ValueError("dt must be > 0")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be >= 1")
    if float(ours_std_scale) <= 0:
        raise ValueError("ours_std_scale must be > 0")
    if float(ours_logit_temperature) <= 0:
        raise ValueError("ours_logit_temperature must be > 0")
    if float(non_gpu_overhead_w) < 0:
        raise ValueError("non_gpu_overhead_w must be >= 0")

    resolved_tp = int(tp_gpus) if tp_gpus is not None else _extract_tp_from_config_id(config_id)
    resolved_tp = max(1, resolved_tp)
    resolved_n_gpus = int(n_gpus_per_node) if n_gpus_per_node is not None else resolved_tp
    resolved_n_gpus = max(resolved_tp, resolved_n_gpus)

    layout = FacilityLayout(
        rows=int(rows),
        racks_per_row=int(racks_per_row),
        nodes_per_rack=int(nodes_per_rack),
    )
    t_horizon = int(np.floor(float(duration_s) / float(dt)))
    if t_horizon <= 0:
        raise ValueError("Computed horizon is zero; increase duration_s or reduce dt.")

    run_manifest_payload = load_json(run_manifest)
    run_cfgs = run_manifest_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    cfg_entry = run_cfgs.get(config_id)
    if not isinstance(cfg_entry, dict):
        raise ValueError(f"config_id '{config_id}' not found in run manifest")
    if str(cfg_entry.get("status", "")) != "trained":
        raise ValueError(f"config '{config_id}' is not trained in run manifest")

    run_manifest_base = str(Path(run_manifest).resolve().parent)
    checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(
        cfg_entry,
        run_manifest_base,
    )
    norm_payload = load_json(norm_path)
    norm_cfg = _extract_norm_for_eval(norm_payload)
    gmm_cfg = load_gmm_params_json_dict(load_json(gmm_path))

    feature_set = str(cfg_entry.get("feature_set", norm_payload.get("feature_set", "f2"))).lower()
    if feature_set == "f3":
        raise ValueError("feature_set='f3' is no longer supported; use 'f2'.")
    if feature_set != "f2":
        raise ValueError(f"invalid feature_set: {feature_set}")
    input_dim = int(cfg_entry.get("input_dim", 2))
    hidden_dim = int(cfg_entry.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
    num_layers = int(cfg_entry.get("num_layers", norm_payload.get("num_layers", 1)))
    k = int(cfg_entry.get("k", gmm_cfg["k"]))
    if int(gmm_cfg["k"]) != k:
        raise ValueError("k mismatch between manifest and gmm params")

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

    train_bundle = _load_training_bundle(
        config_id=config_id,
        experimental_manifest_path=experimental_manifest,
    )
    train_power_flat = np.asarray(train_bundle["train_power_flat"], dtype=np.float64)
    train_power_flat_gpu = np.asarray(train_bundle["train_power_flat_gpu"], dtype=np.float64)
    train_power_traces = list(train_bundle["train_power_traces"])

    use_ar1 = bool(_is_moe_config(config_id))
    ar1_params: Optional[Mapping[str, np.ndarray]] = None
    if use_ar1 and "ours" in method_list:
        ar1_params = _load_or_estimate_ar1_params(
            config_id=config_id,
            gmm_params=gmm_cfg,
            train_power_traces=train_power_traces,
            ar1_params_dir=ar1_params_dir,
        )

    splitwise_requested_tp = int(splitwise_source_tp) if splitwise_source_tp is not None else int(resolved_tp)
    splitwise_strict_params: Optional[Dict[str, object]] = None
    splitwise_meta: Dict[str, object] = {
        "splitwise_source_model": str(splitwise_source_model),
        "splitwise_source_hardware": str(splitwise_source_hardware),
        "splitwise_source_tp": int(splitwise_requested_tp),
        "splitwise_style_lut_mode": str(splitwise_style_lut_mode),
    }
    if "splitwise_strict" in method_list:
        splitwise_strict_params = build_splitwise_style_lut_params(
            config_id=config_id,
            perf_model_csv=splitwise_perf_model_csv,
            train_power_flat=train_power_flat_gpu,
            splitwise_source_model=splitwise_source_model,
            splitwise_source_hardware=splitwise_source_hardware,
            splitwise_source_tp=int(splitwise_requested_tp),
            splitwise_style_lut_mode=splitwise_style_lut_mode,
            n_gpus_per_node=int(resolved_n_gpus),
        )
        splitwise_meta.update(
            {
                "splitwise_source_resolved_model": str(
                    splitwise_strict_params.get("splitwise_source_resolved_model", "")
                ),
                "splitwise_source_resolved_hardware": str(
                    splitwise_strict_params.get("splitwise_source_resolved_hardware", "")
                ),
                "splitwise_source_resolved_tp": int(
                    splitwise_strict_params.get("splitwise_source_resolved_tp", 0)
                ),
                "splitwise_source_match_status": str(
                    splitwise_strict_params.get("splitwise_source_match_status", "")
                ),
                "splitwise_power_quality_flag": str(
                    splitwise_strict_params.get("splitwise_power_quality_flag", "")
                ),
                "splitwise_power_support_status": str(
                    splitwise_strict_params.get("splitwise_power_support_status", "")
                ),
                "splitwise_scheduler_policy": str(
                    splitwise_strict_params.get("splitwise_scheduler_policy", "")
                ),
                "splitwise_extrapolation_events": 0,
                "splitwise_power_clamp_events": 0,
                "splitwise_max_batch_tokens_seen": 0.0,
            }
        )

    clamp_range = (float(norm_cfg["power_min"]), float(norm_cfg["power_max"]))
    ensure_dir(out_root)
    for method in method_list:
        ensure_dir(os.path.join(out_root, method))
    manifest_csv = os.path.join(out_root, "trace_manifest.csv")
    summary_json = os.path.join(out_root, "trace_summary.json")

    node_infos = _list_node_stream_paths(layout, node_stream_dir)
    missing = [path for _, _, _, _, path in node_infos if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} node stream files in {node_stream_dir}; first missing: {missing[0]}"
        )

    rows_out: List[Dict[str, object]] = []
    success_by_method = {method: 0 for method in method_list}
    for start in range(0, len(node_infos), int(batch_size)):
        chunk = node_infos[start : start + int(batch_size)]
        prepared: List[Tuple[int, int, int, int, int, List[Dict[str, float]], np.ndarray]] = []

        for node_id, row, rack, node, path in chunk:
            try:
                requests = _load_node_requests(path)
                feat = build_rollout_features_from_requests(
                    requests=requests,
                    throughput=throughput,
                    norm=norm_cfg,
                    T=t_horizon,
                    dt=float(dt),
                    feature_set=feature_set,
                )
                features = np.asarray(feat["features_norm"], dtype=np.float32)
                if features.ndim != 2 or features.shape[0] != t_horizon:
                    raise ValueError(
                        f"Feature shape mismatch for node {node_id}: {features.shape}, expected ({t_horizon},D)"
                    )
                prepared.append((node_id, row, rack, node, len(requests), requests, features))
            except Exception as exc:
                for method in method_list:
                    rows_out.append(
                        {
                            "method": method,
                            "node_id": int(node_id),
                            "row": int(row),
                            "rack": int(rack),
                            "node": int(node),
                            "file": f"{method}/node_{row}_{rack}_{node}.npy",
                            "num_requests": 0,
                            "seed": int(base_seed + node_id * 1009),
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "num_timesteps": 0,
                            "min_power_w": float("nan"),
                            "max_power_w": float("nan"),
                            "mean_power_w": float("nan"),
                            "uses_ar1": bool(use_ar1 and method == "ours"),
                        }
                    )

        if len(prepared) == 0:
            continue

        logits_np = None
        if "ours" in method_list:
            features_batch = np.stack([item[6] for item in prepared], axis=0).astype(np.float32)
            with torch.no_grad():
                try:
                    x_t = torch.from_numpy(features_batch)
                except Exception:
                    x_t = torch.tensor(features_batch.tolist(), dtype=torch.float32)
                x_t = x_t.to(device=resolved_device, dtype=torch.float32)
                logits = model(x_t)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                try:
                    logits_np = np.asarray(logits.detach().cpu().numpy(), dtype=np.float64)
                except Exception:
                    logits_np = np.asarray(logits.detach().cpu().tolist(), dtype=np.float64)
            if abs(float(ours_logit_temperature) - 1.0) > 1e-12:
                logits_np = logits_np / float(ours_logit_temperature)

        for batch_idx, (node_id, row, rack, node, num_requests, requests, _features) in enumerate(prepared):
            node_seed = int(base_seed + node_id * 1009)
            for method in method_list:
                try:
                    if method == "ours":
                        assert logits_np is not None
                        rng_node = np.random.default_rng(node_seed)
                        p0 = float(rng_node.choice(train_power_flat))
                        trace = _sample_power_from_logits(
                            logits_node=logits_np[batch_idx],
                            gmm_params=gmm_cfg,
                            p0=p0,
                            seed=node_seed + 23,
                            decode_mode=decode_mode,
                            median_filter_window=median_filter_window,
                            clamp_range=clamp_range,
                            std_scale=float(ours_std_scale),
                            use_ar1=bool(use_ar1),
                            ar1_params=ar1_params,
                        )
                    elif method == "splitwise_strict":
                        if splitwise_strict_params is None:
                            raise ValueError("splitwise_strict params unavailable")
                        trace, strict_runtime_meta = generate_splitwise_style_lut_trace(
                            requests=requests,
                            T=t_horizon,
                            dt=float(dt),
                            config={
                                "config_id": config_id,
                                "tp": int(resolved_tp),
                                "n_gpus_per_node": int(resolved_n_gpus),
                                "non_gpu_power_w": 0.0,
                            },
                            lut_params=splitwise_strict_params,
                        )
                        splitwise_meta["splitwise_extrapolation_events"] = int(
                            splitwise_meta.get("splitwise_extrapolation_events", 0)
                        ) + int(strict_runtime_meta.get("splitwise_extrapolation_events", 0))
                        splitwise_meta["splitwise_power_clamp_events"] = int(
                            splitwise_meta.get("splitwise_power_clamp_events", 0)
                        ) + int(strict_runtime_meta.get("splitwise_power_clamp_events", 0))
                        splitwise_meta["splitwise_max_batch_tokens_seen"] = float(
                            max(
                                float(splitwise_meta.get("splitwise_max_batch_tokens_seen", 0.0)),
                                float(strict_runtime_meta.get("splitwise_max_batch_tokens_seen", 0.0)),
                            )
                        )
                        splitwise_meta["splitwise_power_support_status"] = str(
                            strict_runtime_meta.get(
                                "splitwise_power_support_status",
                                splitwise_meta.get("splitwise_power_support_status", ""),
                            )
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
                    if trace.size != t_horizon:
                        if trace.size > t_horizon:
                            trace = trace[:t_horizon]
                        else:
                            fill = trace[-1] if trace.size > 0 else 0.0
                            padded = np.empty((t_horizon,), dtype=np.float64)
                            if trace.size > 0:
                                padded[: trace.size] = trace
                            padded[trace.size :] = float(fill)
                            trace = padded

                    out_path = os.path.join(out_root, method, f"node_{row}_{rack}_{node}.npy")
                    np.save(out_path, np.asarray(trace, dtype=np.float32))
                    rows_out.append(
                        {
                            "method": method,
                            "node_id": int(node_id),
                            "row": int(row),
                            "rack": int(rack),
                            "node": int(node),
                            "file": f"{method}/{os.path.basename(out_path)}",
                            "num_requests": int(num_requests),
                            "seed": int(node_seed),
                            "status": "evaluated",
                            "reason": "",
                            "num_timesteps": int(t_horizon),
                            "min_power_w": float(np.min(trace)),
                            "max_power_w": float(np.max(trace)),
                            "mean_power_w": float(np.mean(trace)),
                            "uses_ar1": bool(use_ar1 and method == "ours"),
                        }
                    )
                    success_by_method[method] += 1
                except Exception as exc:
                    rows_out.append(
                        {
                            "method": method,
                            "node_id": int(node_id),
                            "row": int(row),
                            "rack": int(rack),
                            "node": int(node),
                            "file": f"{method}/node_{row}_{rack}_{node}.npy",
                            "num_requests": int(num_requests),
                            "seed": int(node_seed),
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "num_timesteps": 0,
                            "min_power_w": float("nan"),
                            "max_power_w": float("nan"),
                            "mean_power_w": float("nan"),
                            "uses_ar1": bool(use_ar1 and method == "ours"),
                        }
                    )

    rows_out.sort(key=lambda row: (str(row["method"]), int(row["node_id"])))
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "node_id",
                "row",
                "rack",
                "node",
                "file",
                "num_requests",
                "seed",
                "status",
                "reason",
                "num_timesteps",
                "min_power_w",
                "max_power_w",
                "mean_power_w",
                "uses_ar1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    failed = [row for row in rows_out if str(row["status"]) != "evaluated"]
    summary = {
        "status": "ok" if len(failed) == 0 else "failed",
        "config_id": config_id,
        "methods": list(method_list),
        "node_stream_dir": node_stream_dir,
        "out_root": out_root,
        "trace_manifest_csv": manifest_csv,
        "layout": {
            "rows": int(layout.rows),
            "racks_per_row": int(layout.racks_per_row),
            "nodes_per_rack": int(layout.nodes_per_rack),
            "n_nodes": int(layout.n_nodes),
        },
        "timing": {
            "duration_s": float(duration_s),
            "dt": float(dt),
            "timesteps": int(t_horizon),
        },
        "generation": {
            "batch_size": int(batch_size),
            "base_seed": int(base_seed),
            "device": str(resolved_device),
            "decode_mode": str(decode_mode),
            "median_filter_window": int(median_filter_window),
            "ours_std_scale": float(ours_std_scale),
            "ours_logit_temperature": float(ours_logit_temperature),
            "uses_ar1": bool(use_ar1),
            "tp_gpus": int(resolved_tp),
            "n_gpus_per_node": int(resolved_n_gpus),
            "non_gpu_overhead_w": float(non_gpu_overhead_w),
        },
        "counts": {
            "evaluated_by_method": {key: int(value) for key, value in success_by_method.items()},
            "failed_rows": int(len(failed)),
        },
        "splitwise": {
            "splitwise_source_model": str(splitwise_source_model),
            "splitwise_source_hardware": str(splitwise_source_hardware),
            "splitwise_source_tp": int(splitwise_requested_tp),
            "splitwise_style_lut_mode": str(
                splitwise_meta.get("splitwise_style_lut_mode", splitwise_style_lut_mode)
            ),
            "meta": splitwise_meta,
        },
    }
    write_json(summary_json, summary)

    if len(failed) > 0:
        first_fail = failed[0]
        raise RuntimeError(
            f"Node trace generation failed for {len(failed)} method-node rows. "
            f"First failure method={first_fail['method']} node_id={first_fail['node_id']}: {first_fail['reason']}"
        )
    return summary


def main() -> None:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(
        description="Generate top-level Azure node traces with Splitwise baselines included."
    )
    parser.add_argument("--run-manifest", default=defaults["run_manifest"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--throughput-db", default=defaults["throughput_db"])
    parser.add_argument("--pair-manifest-csv", default=defaults["pair_manifest_csv"])
    parser.add_argument("--splitwise-perf-model-csv", default=defaults["splitwise_perf_model_csv"])
    parser.add_argument("--ar1-params-dir", default=defaults["ar1_params_dir"])
    parser.add_argument("--node-stream-dir", default=defaults["node_stream_dir"])
    parser.add_argument("--output-root", default=defaults["node_traces_root"])
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS_GENERATION))
    parser.add_argument("--duration-s", type=float, default=86400.0)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--nodes-per-rack", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decode-mode", default="stochastic")
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--ours-std-scale", type=float, default=1.0)
    parser.add_argument("--ours-logit-temperature", type=float, default=1.0)
    parser.add_argument("--splitwise-source-model", default=DEFAULT_SPLITWISE_SOURCE_MODEL)
    parser.add_argument("--splitwise-source-hardware", default=DEFAULT_SPLITWISE_SOURCE_HARDWARE)
    parser.add_argument("--splitwise-source-tp", type=int, default=DEFAULT_SPLITWISE_SOURCE_TP)
    parser.add_argument("--splitwise-style-lut-mode", default=SPLITWISE_STYLE_LUT_V1)
    parser.add_argument(
        "--tp-gpus",
        type=int,
        default=None,
        help="Active TP GPUs per node for Splitwise accounting; defaults to tp in config_id.",
    )
    parser.add_argument(
        "--n-gpus-per-node",
        type=int,
        default=None,
        help="Total GPUs per node for Splitwise accounting; defaults to tp_gpus.",
    )
    parser.add_argument("--non-gpu-overhead-w", type=float, default=DEFAULT_NON_GPU_OVERHEAD_W)
    parser.add_argument(
        "--allow-synthetic-request-timestamps",
        action="store_true",
        help="Accepted for compatibility; recorded timestamps are not required here.",
    )
    args = parser.parse_args()

    summary = generate_node_traces(
        run_manifest=str(args.run_manifest),
        experimental_manifest=str(args.experimental_manifest),
        throughput_db=str(args.throughput_db),
        ar1_params_dir=str(args.ar1_params_dir),
        node_stream_dir=str(args.node_stream_dir),
        out_root=str(args.output_root),
        config_id=str(args.config_id),
        methods=str(args.methods),
        duration_s=float(args.duration_s),
        dt=float(args.dt),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        nodes_per_rack=int(args.nodes_per_rack),
        batch_size=int(args.batch_size),
        base_seed=int(args.base_seed),
        device=str(args.device),
        decode_mode=str(args.decode_mode),
        median_filter_window=int(args.median_filter_window),
        ours_std_scale=float(args.ours_std_scale),
        ours_logit_temperature=float(args.ours_logit_temperature),
        splitwise_perf_model_csv=str(args.splitwise_perf_model_csv),
        splitwise_source_model=str(args.splitwise_source_model),
        splitwise_source_hardware=str(args.splitwise_source_hardware),
        splitwise_source_tp=int(args.splitwise_source_tp),
        splitwise_style_lut_mode=str(args.splitwise_style_lut_mode),
        pair_manifest_csv=str(args.pair_manifest_csv),
        tp_gpus=(int(args.tp_gpus) if args.tp_gpus is not None else None),
        n_gpus_per_node=(int(args.n_gpus_per_node) if args.n_gpus_per_node is not None else None),
        non_gpu_overhead_w=float(args.non_gpu_overhead_w),
        require_recorded_timestamps=not bool(args.allow_synthetic_request_timestamps),
    )

    print("=" * 72)
    print("Azure Node Trace Generation")
    print("=" * 72)
    print(f"Config             : {summary['config_id']}")
    print(f"Methods            : {', '.join(summary['methods'])}")
    print(f"Node streams       : {summary['node_stream_dir']}")
    print(f"Output root        : {summary['out_root']}")
    print(f"Nodes              : {summary['layout']['n_nodes']}")
    print(f"Timesteps/node     : {summary['timing']['timesteps']}")
    print(f"AR(1) enabled      : {summary['generation']['uses_ar1']}")
    print(f"Manifest           : {summary['trace_manifest_csv']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
