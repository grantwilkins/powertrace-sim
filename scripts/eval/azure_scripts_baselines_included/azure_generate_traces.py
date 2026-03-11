#!/usr/bin/env python3
"""
Isolated Azure pipeline: generate per-node power traces with Splitwise baselines.

Pipeline per node:
  requests -> rollout features -> BiGRU logits -> sampling / Splitwise LUT -> power trace
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# Allow running via: python3 scripts/eval/azure_scripts_baselines_included/*.py
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.classifiers.gru import GRUClassifier
from scripts.eval.baselines import build_splitwise_lut_params, generate_splitwise_lut
from scripts.eval.pipeline_utils import (
    build_rollout_features_from_requests,
    estimate_ar1_params,
    generate_gmm_bigru_trace,
    generate_gmm_bigru_trace_ar1_thresholded,
    load_gmm_params_json_dict,
    predict_sorted_gmm_labels_from_params,
)
from scripts.eval.run_baselines_node import _is_moe_config

from scripts.eval.azure_scripts_baselines_included.defaults import (
    DEFAULT_CONFIG_ID,
    DEFAULT_METHODS_GENERATION,
    build_default_paths,
)
from scripts.eval.azure_scripts_baselines_included.io_utils import (
    ensure_dir,
    load_json,
    parse_csv_list,
    resolve_existing_path,
    write_json,
)
from scripts.eval.azure_scripts_baselines_included.splitwise_helpers import (
    estimate_splitwise_phase_targets_from_indices,
    load_pair_manifest_map,
)

CONFIG_ID_RE = re.compile(r"^(.+)_(A100|H100)_tp(\d+)$")
ALLOWED_METHODS = {"ours", "splitwise_lut", "splitwise_strict"}


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

    def iter_node_ids(self) -> Sequence[int]:
        return range(self.n_nodes)


def _validate_config_id(config_id: str) -> None:
    if CONFIG_ID_RE.match(str(config_id).strip()) is None:
        raise ValueError(f"Invalid config_id format: {config_id}")


def _extract_tp_from_config_id(config_id: str) -> int:
    m = CONFIG_ID_RE.match(str(config_id).strip())
    if m is None:
        return 4
    try:
        tp = int(m.group(3))
    except Exception:
        return 4
    return max(1, tp)


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
    dataset_path = resolve_existing_path(str(row.get("dataset_npz", "")), experimental_base)
    split_path = resolve_existing_path(str(row.get("split_json", "")), experimental_base)
    if dataset_path is None:
        raise ValueError(f"Dataset path not found for '{config_id}'")
    if split_path is None:
        raise ValueError(f"Split path not found for '{config_id}'")
    return dataset_path, split_path


def _load_training_bundle(
    *,
    config_id: str,
    experimental_manifest_path: str,
    non_gpu_overhead_w: float,
) -> Dict[str, object]:
    manifest = load_json(experimental_manifest_path)
    base = str(Path(experimental_manifest_path).resolve().parent)
    dataset_path, split_path = _resolve_experimental_paths(manifest, config_id=config_id, experimental_base=base)
    split_payload = load_json(split_path)
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]

    with np.load(dataset_path, allow_pickle=True) as data:
        power_arr = np.asarray(data["power"], dtype=object)
        pair_key_arr = np.asarray(data["pair_key"], dtype=object) if "pair_key" in data else np.asarray([], dtype=object)
        power_start_arr = (
            np.asarray(data["power_start_epoch_s"], dtype=np.float64)
            if "power_start_epoch_s" in data
            else np.asarray([], dtype=np.float64)
        )

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
    flat_gpu = np.clip(flat - float(non_gpu_overhead_w), a_min=0.0, a_max=None)

    return {
        "dataset_path": dataset_path,
        "split_path": split_path,
        "train_indices": train_indices,
        "train_power_traces": train_traces,
        "train_power_flat": flat,
        "train_power_flat_gpu": flat_gpu,
        "power_arr": power_arr,
        "pair_key_arr": pair_key_arr,
        "power_start_arr": power_start_arr,
    }


def _resolve_checkpoint_norm_gmm_paths(
    config_entry: Mapping[str, object],
    base_dir: str,
) -> Tuple[str, str, str]:
    checkpoint_raw = str(config_entry.get("checkpoint_path", ""))
    norm_raw = str(config_entry.get("norm_params_path", ""))
    gmm_raw = str(config_entry.get("gmm_params_path", ""))
    checkpoint_path = resolve_existing_path(checkpoint_raw, base_dir)
    norm_path = resolve_existing_path(norm_raw, base_dir)
    gmm_path = resolve_existing_path(gmm_raw, base_dir)
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint path not found: {checkpoint_raw}")
    if norm_path is None:
        raise ValueError(f"Norm params path not found: {norm_raw}")
    if gmm_path is None:
        raise ValueError(f"GMM path not found: {gmm_raw}")
    return checkpoint_path, norm_path, gmm_path


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
    invalid = [m for m in out if m not in ALLOWED_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Allowed: {sorted(ALLOWED_METHODS)}")
    dedup: List[str] = []
    for m in out:
        if m not in dedup:
            dedup.append(m)
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
    splitwise_source_model: str = "llama2-70b",
    splitwise_source_hardware: str = "a100-80gb",
    splitwise_source_tp: int = 4,
    splitwise_calibration_mode: str = "train_phase_matched_v1",
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    tp_gpus: Optional[int] = None,
    n_gpus_per_node: Optional[int] = None,
    non_gpu_overhead_w: float = 1000.0,
    require_recorded_timestamps: bool = True,
) -> Dict[str, object]:
    _validate_config_id(config_id)
    method_list = _normalize_methods(methods)

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

    layout = FacilityLayout(rows=int(rows), racks_per_row=int(racks_per_row), nodes_per_rack=int(nodes_per_rack))
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
    checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(cfg_entry, run_manifest_base)
    norm_payload = load_json(norm_path)
    norm_cfg = _extract_norm_for_eval(norm_payload)
    gmm_cfg = load_gmm_params_json_dict(load_json(gmm_path))

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

    throughput_payload = load_json(throughput_db)
    throughput = _resolve_throughput(throughput_payload, config_id)

    train_bundle = _load_training_bundle(
        config_id=config_id,
        experimental_manifest_path=experimental_manifest,
        non_gpu_overhead_w=float(non_gpu_overhead_w),
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

    splitwise_lut_params: Optional[Dict[str, float]] = None
    splitwise_strict_params: Optional[Dict[str, float]] = None
    splitwise_meta: Dict[str, object] = {
        "mode": str(splitwise_calibration_mode),
        "phase_targets_status": "not_requested",
        "phase_targets_samples": {},
        "phase_targets_reason": "",
    }
    needs_splitwise = any(m in {"splitwise_lut", "splitwise_strict"} for m in method_list)
    if needs_splitwise:
        pair_map = load_pair_manifest_map(pair_manifest_csv)
        phase_targets: Dict[str, float] = {}
        if "splitwise_lut" in method_list:
            phase_samples_possible = (
                len(pair_map) > 0
                and np.asarray(train_bundle["pair_key_arr"], dtype=object).size > 0
                and np.asarray(train_bundle["power_start_arr"], dtype=np.float64).size > 0
            )
            if phase_samples_possible:
                try:
                    phase_targets = estimate_splitwise_phase_targets_from_indices(
                        indices=list(train_bundle["train_indices"]),
                        pair_key_arr=np.asarray(train_bundle["pair_key_arr"], dtype=object),
                        power_arr=np.asarray(train_bundle["power_arr"], dtype=object),
                        power_start_arr=np.asarray(train_bundle["power_start_arr"], dtype=np.float64),
                        pair_map=pair_map,
                        throughput=throughput,
                        norm_cfg=norm_cfg,
                        feature_set=feature_set,
                        dt=float(dt),
                        non_gpu_overhead_w=float(non_gpu_overhead_w),
                        require_recorded_timestamps=bool(require_recorded_timestamps),
                    )
                    total_phase_samples = int(
                        phase_targets.get("splitwise_phase_samples_idle", 0)
                        + phase_targets.get("splitwise_phase_samples_decode", 0)
                        + phase_targets.get("splitwise_phase_samples_prefill", 0)
                    )
                    if total_phase_samples > 0:
                        splitwise_meta["phase_targets_status"] = "estimated_from_train_pairs"
                    else:
                        splitwise_meta["phase_targets_status"] = "fallback_train_stats"
                        splitwise_meta["phase_targets_reason"] = "phase masks had zero samples"
                except Exception as exc:
                    splitwise_meta["phase_targets_status"] = "fallback_train_stats"
                    splitwise_meta["phase_targets_reason"] = f"{type(exc).__name__}: {exc}"
            else:
                splitwise_meta["phase_targets_status"] = "fallback_train_stats"
                if len(pair_map) == 0:
                    splitwise_meta["phase_targets_reason"] = "pair manifest missing or empty"
                else:
                    splitwise_meta["phase_targets_reason"] = "dataset missing pair_key or power_start_epoch_s"

            splitwise_meta["phase_targets_samples"] = {
                "idle": int(phase_targets.get("splitwise_phase_samples_idle", 0)),
                "decode": int(phase_targets.get("splitwise_phase_samples_decode", 0)),
                "prefill": int(phase_targets.get("splitwise_phase_samples_prefill", 0)),
            }

            splitwise_lut_params = build_splitwise_lut_params(
                config_id=config_id,
                perf_model_csv=splitwise_perf_model_csv,
                train_power_flat=train_power_flat_gpu,
                splitwise_source_model=splitwise_source_model,
                splitwise_source_hardware=splitwise_source_hardware,
                splitwise_source_tp=int(splitwise_source_tp),
                splitwise_calibration_mode=splitwise_calibration_mode,
                n_gpus_per_node=int(resolved_n_gpus),
                target_idle_node_gpu_w=phase_targets.get("target_idle_node_gpu_w"),
                target_decode_node_gpu_w=phase_targets.get("target_decode_node_gpu_w"),
                target_prefill_node_gpu_w=phase_targets.get("target_prefill_node_gpu_w"),
            )

        if "splitwise_strict" in method_list:
            splitwise_strict_params = build_splitwise_lut_params(
                config_id=config_id,
                perf_model_csv=splitwise_perf_model_csv,
                train_power_flat=train_power_flat_gpu,
                splitwise_source_model=splitwise_source_model,
                splitwise_source_hardware=splitwise_source_hardware,
                splitwise_source_tp=int(splitwise_source_tp),
                splitwise_calibration_mode="dgx_fixed_targets_v1",
                n_gpus_per_node=int(resolved_n_gpus),
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
    success_by_method = {m: 0 for m in method_list}
    for start in range(0, len(node_infos), int(batch_size)):
        chunk = node_infos[start : start + int(batch_size)]
        prepared: List[Tuple[int, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]] = []

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
                a_raw = np.asarray(feat["A_raw"], dtype=np.float64).reshape(-1)
                d_raw = np.asarray(feat["delta_A_raw"], dtype=np.float64).reshape(-1)
                if features.ndim != 2 or features.shape[0] != t_horizon:
                    raise ValueError(
                        f"Feature shape mismatch for node {node_id}: {features.shape}, expected ({t_horizon},D)"
                    )
                prepared.append((node_id, row, rack, node, len(requests), features, a_raw[:t_horizon], d_raw[:t_horizon]))
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
            features_batch = np.stack([x[5] for x in prepared], axis=0).astype(np.float32)
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

        for b_idx, (node_id, row, rack, node, n_requests, _features, a_raw, d_raw) in enumerate(prepared):
            node_seed = int(base_seed + node_id * 1009)
            for method in method_list:
                try:
                    if method == "ours":
                        assert logits_np is not None
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
                    elif method == "splitwise_lut":
                        if splitwise_lut_params is None:
                            raise ValueError("splitwise_lut params unavailable")
                        trace = generate_splitwise_lut(
                            a_raw,
                            d_raw,
                            {
                                "config_id": config_id,
                                "tp": int(resolved_tp),
                                "n_gpus_per_node": int(resolved_n_gpus),
                                "non_gpu_power_w": 0.0,
                            },
                            splitwise_lut_params,
                        )
                    elif method == "splitwise_strict":
                        if splitwise_strict_params is None:
                            raise ValueError("splitwise_strict params unavailable")
                        trace = generate_splitwise_lut(
                            a_raw,
                            d_raw,
                            {
                                "config_id": config_id,
                                "tp": int(resolved_tp),
                                "n_gpus_per_node": int(resolved_n_gpus),
                                "non_gpu_power_w": 0.0,
                            },
                            splitwise_strict_params,
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
                            "num_requests": int(n_requests),
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
                            "num_requests": int(n_requests),
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

    rows_out.sort(key=lambda r: (str(r["method"]), int(r["node_id"])))
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

    failed = [r for r in rows_out if str(r["status"]) != "evaluated"]
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
            "tp_gpus": int(resolved_tp),
            "n_gpus_per_node": int(resolved_n_gpus),
            "non_gpu_overhead_w": float(non_gpu_overhead_w),
        },
        "counts": {
            "evaluated_by_method": {k: int(v) for k, v in success_by_method.items()},
            "failed_rows": int(len(failed)),
        },
        "splitwise": {
            "splitwise_source_model": str(splitwise_source_model),
            "splitwise_source_hardware": str(splitwise_source_hardware),
            "splitwise_source_tp": int(splitwise_source_tp),
            "splitwise_calibration_mode": str(splitwise_calibration_mode),
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
        description="Isolated Azure trace generation with Splitwise baselines included."
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
    parser.add_argument("--splitwise-source-model", default="llama2-70b")
    parser.add_argument("--splitwise-source-hardware", default="a100-80gb")
    parser.add_argument("--splitwise-source-tp", type=int, default=4)
    parser.add_argument("--splitwise-calibration-mode", default="train_phase_matched_v1")
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
    parser.add_argument("--non-gpu-overhead-w", type=float, default=1000.0)
    parser.add_argument(
        "--allow-synthetic-request-timestamps",
        action="store_true",
        help="Allow synthetic request timestamps when stage0 json omits recorded timestamps.",
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
        splitwise_calibration_mode=str(args.splitwise_calibration_mode),
        pair_manifest_csv=str(args.pair_manifest_csv),
        tp_gpus=(int(args.tp_gpus) if args.tp_gpus is not None else None),
        n_gpus_per_node=(int(args.n_gpus_per_node) if args.n_gpus_per_node is not None else None),
        non_gpu_overhead_w=float(args.non_gpu_overhead_w),
        require_recorded_timestamps=not bool(args.allow_synthetic_request_timestamps),
    )

    print("=" * 72)
    print("Azure Node Trace Generation (Baselines Included)")
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
