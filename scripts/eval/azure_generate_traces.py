#!/usr/bin/env python3
"""
Experiment 2b: Generate per-node power traces for Azure facility streams.

Pipeline per node:
  requests -> rollout features -> BiGRU logits -> GMM/AR(1) sampling -> power trace
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from model.classifiers.gru import GRUClassifier
from scripts.eval.pipeline_utils import (
    build_rollout_features_from_requests,
    estimate_ar1_params,
    generate_gmm_bigru_trace,
    generate_gmm_bigru_trace_ar1_thresholded,
    load_gmm_params_json_dict,
    predict_sorted_gmm_labels_from_params,
)
from scripts.eval.run_baselines_node import _is_moe_config

CONFIG_ID_RE = re.compile(r"^(.+)_(A100|H100)_tp(\d+)$")


@dataclass(frozen=True)
class FacilityLayout:
    rows: int = 10
    racks_per_row: int = 6
    nodes_per_rack: int = 4

    @property
    def n_nodes(self) -> int:
        return int(self.rows) * int(self.racks_per_row) * int(self.nodes_per_rack)

    def node_id_to_coords(self, node_id: int) -> Tuple[int, int, int]:
        npr = int(self.nodes_per_rack)
        rpr = int(self.racks_per_row)
        per_row = rpr * npr
        row = int(node_id) // per_row
        rem = int(node_id) % per_row
        rack = rem // npr
        node = rem % npr
        return int(row), int(rack), int(node)

    def node_coords_to_id(self, row: int, rack: int, node: int) -> int:
        if row < 0 or row >= int(self.rows):
            raise ValueError(f"row out of bounds: {row}")
        if rack < 0 or rack >= int(self.racks_per_row):
            raise ValueError(f"rack out of bounds: {rack}")
        if node < 0 or node >= int(self.nodes_per_rack):
            raise ValueError(f"node out of bounds: {node}")
        return int(row) * int(self.racks_per_row) * int(self.nodes_per_rack) + int(rack) * int(
            self.nodes_per_rack
        ) + int(node)

    def iter_node_ids(self) -> Sequence[int]:
        return range(self.n_nodes)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _resolve_device(device: str) -> torch.device:
    if str(device).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _extract_norm_for_eval(norm_payload: Mapping[str, object]) -> Dict[str, float]:
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


def _validate_config_id(config_id: str) -> None:
    if CONFIG_ID_RE.match(str(config_id).strip()) is None:
        raise ValueError(f"Invalid config_id format: {config_id}")


def _resolve_throughput(throughput_payload: Mapping[str, object], config_id: str) -> Dict[str, float]:
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
    experimental_manifest: Mapping[str, object],
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
    requests.sort(key=lambda r: float(r["arrival_time"]))
    return requests


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "run_manifest": str(repo_root / "results" / "continuous_v1_gmm_bigru" / "k10_f2" / "run_manifest.json"),
        "experimental_manifest": str(repo_root / "results" / "experimental_continuous_v1" / "manifest.json"),
        "throughput_db": str(repo_root / "model" / "config" / "throughput_database.json"),
        "ar1_params_dir": str(
            repo_root
            / "results"
            / "continuous_v1_gmm_bigru"
            / "k10_f2_ar1_thresh"
            / "ar1_params"
        ),
        "node_stream_dir": str(repo_root / "data" / "azure_facility" / "node_streams"),
        "out_dir": str(repo_root / "results" / "azure_facility" / "node_traces"),
    }


def _load_training_power_pool(
    *,
    config_id: str,
    experimental_manifest_path: str,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    manifest = _load_json(experimental_manifest_path)
    base = str(Path(experimental_manifest_path).resolve().parent)
    dataset_path, split_path = _resolve_experimental_paths(manifest, config_id=config_id, experimental_base=base)
    split_payload = _load_json(split_path)
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]

    with np.load(dataset_path, allow_pickle=True) as data:
        power_arr = np.asarray(data["power"], dtype=object)
    n_total = int(len(power_arr))

    train_traces: List[np.ndarray] = []
    for idx in train_indices:
        if idx < 0 or idx >= n_total:
            continue
        p = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if p.size > 0:
            train_traces.append(p.astype(np.float64))

    if len(train_traces) == 0:
        for i in range(n_total):
            p = np.asarray(power_arr[i], dtype=np.float64).reshape(-1)
            if p.size > 0:
                train_traces.append(p.astype(np.float64))
                if len(train_traces) >= 3:
                    break
    if len(train_traces) == 0:
        raise ValueError(f"Unable to build training power pool for {config_id}")

    flat = np.concatenate(train_traces, axis=0).astype(np.float64)
    if flat.size == 0:
        raise ValueError("Training power pool is empty after concat")
    return flat, train_traces


def _resolve_checkpoint_norm_gmm_paths(
    config_entry: Mapping[str, object],
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
            variances * float(std_scale * std_scale), a_min=1e-12, a_max=None
        )

    if use_ar1:
        if ar1_params is None:
            raise ValueError("AR(1) requested but ar1_params is None")
        sigma_innov = np.asarray(ar1_params["sigma_innov"], dtype=np.float64).reshape(-1) * float(
            std_scale
        )
        sigma_marginal = np.asarray(ar1_params["sigma_marginal"], dtype=np.float64).reshape(-1) * float(
            std_scale
        )
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


def generate_node_traces(
    *,
    run_manifest: str,
    experimental_manifest: str,
    throughput_db: str,
    ar1_params_dir: str,
    node_stream_dir: str,
    out_dir: str,
    config_id: str = "deepseek-r1-distill-70b_H100_tp4",
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
) -> Dict[str, object]:
    _validate_config_id(config_id)
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

    layout = FacilityLayout(rows=int(rows), racks_per_row=int(racks_per_row), nodes_per_rack=int(nodes_per_rack))
    n_nodes = int(layout.n_nodes)
    t_horizon = int(np.floor(float(duration_s) / float(dt)))
    if t_horizon <= 0:
        raise ValueError("Computed horizon is zero; increase duration_s or reduce dt.")

    run_manifest_payload = _load_json(run_manifest)
    run_cfgs = run_manifest_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    cfg_entry = run_cfgs.get(config_id)
    if not isinstance(cfg_entry, dict):
        raise ValueError(f"config_id '{config_id}' not found in run manifest")
    if str(cfg_entry.get("status", "")) != "trained":
        raise ValueError(f"config '{config_id}' is not trained in run manifest")

    run_manifest_base = str(Path(run_manifest).resolve().parent)
    checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(cfg_entry, run_manifest_base)
    norm_payload = _load_json(norm_path)
    norm_cfg = _extract_norm_for_eval(norm_payload)
    gmm_cfg = load_gmm_params_json_dict(_load_json(gmm_path))

    feature_set = str(cfg_entry.get("feature_set", norm_payload.get("feature_set", "f2"))).lower()
    if feature_set not in {"f2", "f3"}:
        raise ValueError(f"invalid feature_set: {feature_set}")
    input_dim = int(cfg_entry.get("input_dim", 2 if feature_set == "f2" else 3))
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

    throughput_payload = _load_json(throughput_db)
    throughput = _resolve_throughput(throughput_payload, config_id)
    train_power_flat, train_power_traces = _load_training_power_pool(
        config_id=config_id,
        experimental_manifest_path=experimental_manifest,
    )

    use_ar1 = bool(_is_moe_config(config_id))
    ar1_params: Optional[Mapping[str, np.ndarray]] = None
    if use_ar1:
        ar1_params = _load_or_estimate_ar1_params(
            config_id=config_id,
            gmm_params=gmm_cfg,
            train_power_traces=train_power_traces,
            ar1_params_dir=ar1_params_dir,
        )

    clamp_range = (float(norm_cfg["power_min"]), float(norm_cfg["power_max"]))
    _ensure_dir(out_dir)
    manifest_csv = os.path.join(out_dir, "trace_manifest.csv")
    summary_json = os.path.join(out_dir, "trace_summary.json")

    node_infos = _list_node_stream_paths(layout, node_stream_dir)
    missing = [path for _, _, _, _, path in node_infos if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} node stream files in {node_stream_dir}; first missing: {missing[0]}"
        )

    rows_out: List[Dict[str, object]] = []
    success_count = 0
    for start in range(0, len(node_infos), int(batch_size)):
        chunk = node_infos[start : start + int(batch_size)]
        prepared: List[Tuple[int, int, int, int, str, int, np.ndarray]] = []

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
                prepared.append((node_id, row, rack, node, path, len(requests), features))
            except Exception as exc:
                rows_out.append(
                    {
                        "node_id": int(node_id),
                        "row": int(row),
                        "rack": int(rack),
                        "node": int(node),
                        "file": f"node_{row}_{rack}_{node}.npy",
                        "num_requests": 0,
                        "seed": int(base_seed + node_id * 1009),
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "num_timesteps": 0,
                        "min_power_w": float("nan"),
                        "max_power_w": float("nan"),
                        "mean_power_w": float("nan"),
                        "uses_ar1": bool(use_ar1),
                    }
                )

        if len(prepared) == 0:
            continue

        features_batch = np.stack([x[6] for x in prepared], axis=0).astype(np.float32)
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

        for b_idx, (node_id, row, rack, node, _path, n_requests, _features) in enumerate(prepared):
            node_seed = int(base_seed + node_id * 1009)
            try:
                rng_node = np.random.default_rng(node_seed)
                p0 = float(rng_node.choice(train_power_flat))
                trace = _sample_power_from_logits(
                    logits_node=logits_np[b_idx],
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
                if trace.size != t_horizon:
                    if trace.size > t_horizon:
                        trace = trace[:t_horizon]
                    else:
                        fill = trace[-1] if trace.size > 0 else p0
                        padded = np.empty((t_horizon,), dtype=np.float64)
                        if trace.size > 0:
                            padded[: trace.size] = trace
                        padded[trace.size :] = float(fill)
                        trace = padded
                out_path = os.path.join(out_dir, f"node_{row}_{rack}_{node}.npy")
                np.save(out_path, np.asarray(trace, dtype=np.float32))

                rows_out.append(
                    {
                        "node_id": int(node_id),
                        "row": int(row),
                        "rack": int(rack),
                        "node": int(node),
                        "file": os.path.basename(out_path),
                        "num_requests": int(n_requests),
                        "seed": int(node_seed),
                        "status": "evaluated",
                        "reason": "",
                        "num_timesteps": int(t_horizon),
                        "min_power_w": float(np.min(trace)),
                        "max_power_w": float(np.max(trace)),
                        "mean_power_w": float(np.mean(trace)),
                        "uses_ar1": bool(use_ar1),
                    }
                )
                success_count += 1
            except Exception as exc:
                rows_out.append(
                    {
                        "node_id": int(node_id),
                        "row": int(row),
                        "rack": int(rack),
                        "node": int(node),
                        "file": f"node_{row}_{rack}_{node}.npy",
                        "num_requests": int(n_requests),
                        "seed": int(node_seed),
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "num_timesteps": 0,
                        "min_power_w": float("nan"),
                        "max_power_w": float("nan"),
                        "mean_power_w": float("nan"),
                        "uses_ar1": bool(use_ar1),
                    }
                )

    rows_out.sort(key=lambda r: int(r["node_id"]))
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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

    failed = [r for r in rows_out if str(r["status"]) != "evaluated"]
    summary = {
        "status": "ok" if len(failed) == 0 else "failed",
        "config_id": config_id,
        "node_stream_dir": node_stream_dir,
        "out_dir": out_dir,
        "trace_manifest_csv": manifest_csv,
        "layout": {
            "rows": int(layout.rows),
            "racks_per_row": int(layout.racks_per_row),
            "nodes_per_rack": int(layout.nodes_per_rack),
            "n_nodes": int(layout.n_nodes),
        },
        "timing": {"duration_s": float(duration_s), "dt": float(dt), "timesteps": int(t_horizon)},
        "generation": {
            "batch_size": int(batch_size),
            "base_seed": int(base_seed),
            "device": str(resolved_device),
            "decode_mode": str(decode_mode),
            "median_filter_window": int(median_filter_window),
            "ours_std_scale": float(ours_std_scale),
            "ours_logit_temperature": float(ours_logit_temperature),
            "uses_ar1": bool(use_ar1),
        },
        "counts": {
            "evaluated_nodes": int(success_count),
            "failed_nodes": int(len(failed)),
        },
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if len(failed) > 0:
        first_fail = failed[0]
        raise RuntimeError(
            f"Node trace generation failed for {len(failed)} nodes. "
            f"First failure node_id={first_fail['node_id']}: {first_fail['reason']}"
        )
    return summary


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(description="Experiment 2b: generate per-node Azure power traces.")
    parser.add_argument("--run-manifest", default=defaults["run_manifest"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--throughput-db", default=defaults["throughput_db"])
    parser.add_argument("--ar1-params-dir", default=defaults["ar1_params_dir"])
    parser.add_argument("--node-stream-dir", default=defaults["node_stream_dir"])
    parser.add_argument("--out-dir", default=defaults["out_dir"])
    parser.add_argument("--config-id", default="deepseek-r1-distill-70b_H100_tp4")
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
    args = parser.parse_args()

    summary = generate_node_traces(
        run_manifest=str(args.run_manifest),
        experimental_manifest=str(args.experimental_manifest),
        throughput_db=str(args.throughput_db),
        ar1_params_dir=str(args.ar1_params_dir),
        node_stream_dir=str(args.node_stream_dir),
        out_dir=str(args.out_dir),
        config_id=str(args.config_id),
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
    )

    print("=" * 72)
    print("Azure Node Trace Generation (Experiment 2b)")
    print("=" * 72)
    print(f"Config             : {summary['config_id']}")
    print(f"Node streams       : {summary['node_stream_dir']}")
    print(f"Output dir         : {summary['out_dir']}")
    print(f"Nodes evaluated    : {summary['counts']['evaluated_nodes']}/{summary['layout']['n_nodes']}")
    print(f"Timesteps/node     : {summary['timing']['timesteps']}")
    print(f"AR(1) enabled      : {summary['generation']['uses_ar1']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
