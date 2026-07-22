#!/usr/bin/env python3
"""
Experiment 2b: Generate per-node power traces for Azure facility streams.

Pipeline per node:
  requests -> selected timing/work model -> clean power surface / Splitwise LUT
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

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
from scripts.eval.splitwise import (
    SPLITWISE_REMOVED_MESSAGE,
    SPLITWISE_STYLE_LUT_V1,
    build_splitwise_style_lut_params,
    generate_splitwise_style_lut_trace,
    normalize_splitwise_style_lut_mode,
)
from scripts.eval.facility import FacilityLayout
from model.classifiers.physics import load_physics_artifact, predict_mean_node_power
from model.pipeline.physics_inference import build_modeled_work_ledger
from model.pipeline.artifact_resolution import (
    resolve_bound_throughput,
    resolve_experimental_paths,
)
from model.training_data.arch import get_arch
from model.utils.config import tp_gpus_from_config_id
from model.utils.provenance import file_identity, git_state
from model.power import idle_node_power
from model.release import (
    DEFAULT_ARTIFACT,
    load_artifact,
    resolve_deployment,
    support_violations,
)
from model.simulation import simulate

CONFIG_ID_RE = re.compile(r"^(.+)_(A100|H100)_tp(\d+)$")
ALLOWED_METHODS = {"ours", "physics", "splitwise_strict"}
TIMING_MODE = "arrival_only"


def _validate_config_id(config_id: str) -> None:
    if CONFIG_ID_RE.match(str(config_id).strip()) is None:
        raise ValueError(f"Invalid config_id format: {config_id}")


def _load_training_bundle(
    *,
    config_id: str,
    experimental_manifest_path: str,
) -> Dict[str, object]:
    manifest = load_json(experimental_manifest_path)
    base = str(Path(experimental_manifest_path).resolve().parent)
    dataset_path, split_path = resolve_experimental_paths(
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
        raise ValueError(f"Training split has no power traces for {config_id}")

    flat = np.concatenate(train_traces, axis=0).astype(np.float64)
    if flat.size == 0:
        raise ValueError("Training power pool is empty after concat")

    return {
        "train_power_traces": train_traces,
        "train_power_flat": flat,
        "train_power_flat_gpu": flat.copy(),
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
            if not np.all(np.isfinite([arrival, n_in, n_out])):
                raise ValueError(f"Non-finite request fields at {path}:{row_idx}")
            if arrival < 0.0 or n_in < 0.0 or n_out < 0.0:
                raise ValueError(f"Negative request fields at {path}:{row_idx}")
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


def _selected_power_trace(
    requests: List[Dict[str, float]], *, preset: str, artifact_path: str,
    horizon: int,
) -> np.ndarray:
    artifact = load_artifact(artifact_path)
    deployment, _ = resolve_deployment(preset, artifact)
    violations = support_violations(deployment, artifact)
    if violations:
        raise ValueError("unsupported deployment: " + "; ".join(violations))
    model = str(deployment["model"])
    idle = idle_node_power(
        arch=artifact["architectures"][model],
        model=model,
        hardware=str(deployment["hardware"]),
        tp=int(deployment["tp"]),
        artifact=artifact,
    )
    trace = np.full(horizon, idle, dtype=np.float64)
    if requests:
        result = simulate(
            requests,
            deployment=preset,
            artifact=artifact,
        )
        predicted = np.asarray(
            result.power["node_gpu_power_w"], dtype=np.float64
        )
        copied = min(horizon, predicted.size)
        trace[:copied] = predicted[:copied]
    return trace


def generate_node_traces(
    *,
    run_manifest: str,
    experimental_manifest: str,
    throughput_db: str,
    physics_artifact: str = "feature-test/results/physics_artifact_v1.json",
    selected_artifact: str = str(DEFAULT_ARTIFACT),
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
    splitwise_perf_model_csv: str = "data/perf_model.csv",
    splitwise_source_model: str = DEFAULT_SPLITWISE_SOURCE_MODEL,
    splitwise_source_hardware: str = DEFAULT_SPLITWISE_SOURCE_HARDWARE,
    splitwise_source_tp: Optional[int] = None,
    splitwise_style_lut_mode: str = SPLITWISE_STYLE_LUT_V1,
    tp_gpus: Optional[int] = None,
    n_gpus_per_node: Optional[int] = None,
    non_gpu_overhead_w: float = DEFAULT_NON_GPU_OVERHEAD_W,
) -> Dict[str, object]:
    del throughput_db

    _validate_config_id(config_id)
    method_list = _normalize_methods(methods)
    splitwise_style_lut_mode = normalize_splitwise_style_lut_mode(
        splitwise_style_lut_mode
    )
    if float(duration_s) <= 0:
        raise ValueError("duration_s must be > 0")
    if float(dt) <= 0:
        raise ValueError("dt must be > 0")
    if "ours" in method_list and not np.isclose(float(dt), 0.25):
        raise ValueError("selected model inference requires dt=0.25 s")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be >= 1")
    if float(non_gpu_overhead_w) < 0:
        raise ValueError("non_gpu_overhead_w must be >= 0")

    resolved_tp = int(tp_gpus) if tp_gpus is not None else tp_gpus_from_config_id(config_id)
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

    run_cfgs = {}
    if Path(run_manifest).is_file():
        run_manifest_payload = load_json(run_manifest)
        run_cfgs = run_manifest_payload.get("configs", {})
        if not isinstance(run_cfgs, dict):
            raise ValueError("Invalid run manifest format")
    cfg_entry = run_cfgs.get(config_id)
    cfg_entry = cfg_entry if isinstance(cfg_entry, dict) else {}
    throughput = (
        resolve_bound_throughput(cfg_entry, config_id)
        if "physics" in method_list else None
    )
    physics_payload = None
    physics_arch = None
    config_match = CONFIG_ID_RE.match(config_id)
    if "physics" in method_list:
        if config_match is None:
            raise ValueError(f"Cannot resolve physics identity from {config_id!r}")
        physics_payload = load_physics_artifact(physics_artifact)
        physics_model = str(config_match.group(1))
        architectures = physics_payload.get("architectures", {})
        physics_arch = (
            dict(architectures[physics_model])
            if physics_model in architectures
            else get_arch(physics_model)
        )
        if str(config_match.group(2)) not in physics_payload["hardware"]:
            raise ValueError(
                f"Physics artifact has no {config_match.group(2)} coefficients"
            )

    # All model classes generate IID; AR(1) generation was removed.
    generation_mode_by_method = {
        method: (
            "selected_deterministic_mean"
            if method == "ours"
            else "modeled_mean"
            if method == "physics"
            else "splitwise_style_lut"
        )
        for method in method_list
    }

    splitwise_requested_tp = int(splitwise_source_tp) if splitwise_source_tp is not None else int(resolved_tp)
    splitwise_strict_params: Optional[Dict[str, object]] = None
    splitwise_meta: Dict[str, object] = {
        "splitwise_source_model": str(splitwise_source_model),
        "splitwise_source_hardware": str(splitwise_source_hardware),
        "splitwise_source_tp": int(splitwise_requested_tp),
        "splitwise_style_lut_mode": str(splitwise_style_lut_mode),
    }
    if "splitwise_strict" in method_list:
        train_bundle = _load_training_bundle(
            config_id=config_id,
            experimental_manifest_path=experimental_manifest,
        )
        train_power_flat_gpu = np.asarray(
            train_bundle["train_power_flat_gpu"], dtype=np.float64
        )
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
                features = np.empty((t_horizon, 0), dtype=np.float32)
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
                            "generation_mode": str(generation_mode_by_method[method]),
                            "timing_mode": TIMING_MODE,
                            "num_requests": 0,
                            "seed": int(base_seed + node_id * 1009),
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "num_timesteps": 0,
                            "min_power_w": float("nan"),
                            "max_power_w": float("nan"),
                            "mean_power_w": float("nan"),
                        }
                    )

        if len(prepared) == 0:
            continue

        for node_id, row, rack, node, num_requests, requests, _features in prepared:
            node_seed = int(base_seed + node_id * 1009)
            for method in method_list:
                try:
                    if method == "ours":
                        match = CONFIG_ID_RE.fullmatch(config_id)
                        if match is None:
                            raise ValueError(f"Cannot resolve selected identity from {config_id!r}")
                        selected_preset = (
                            f"{match.group(1)}-{match.group(2).lower()}-tp{int(match.group(3))}"
                        )
                        trace = _selected_power_trace(
                            requests,
                            preset=selected_preset,
                            artifact_path=selected_artifact,
                            horizon=t_horizon,
                        )
                    elif method == "physics":
                        assert physics_payload is not None and physics_arch is not None
                        assert throughput is not None
                        ledger = build_modeled_work_ledger(
                            requests,
                            arch=physics_arch,
                            tp=int(resolved_tp),
                            throughput=throughput,
                            dt=float(dt),
                            T=t_horizon,
                        )
                        trace = predict_mean_node_power(
                            ledger,
                            physics_arch,
                            tp=int(resolved_tp),
                            hardware=str(config_match.group(2)),
                            artifact=physics_payload,
                            dt_s=float(dt),
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
                            "generation_mode": str(generation_mode_by_method[method]),
                            "timing_mode": TIMING_MODE,
                            "num_requests": int(num_requests),
                            "seed": int(node_seed),
                            "status": "evaluated",
                            "reason": "",
                            "num_timesteps": int(t_horizon),
                            "min_power_w": float(np.min(trace)),
                            "max_power_w": float(np.max(trace)),
                            "mean_power_w": float(np.mean(trace)),
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
                            "generation_mode": str(generation_mode_by_method[method]),
                            "timing_mode": TIMING_MODE,
                            "num_requests": int(num_requests),
                            "seed": int(node_seed),
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "num_timesteps": 0,
                            "min_power_w": float("nan"),
                            "max_power_w": float("nan"),
                            "mean_power_w": float("nan"),
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
                "generation_mode",
                "timing_mode",
                "num_requests",
                "seed",
                "status",
                "reason",
                "num_timesteps",
                "min_power_w",
                "max_power_w",
                "mean_power_w",
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
            "runtime": "numpy",
            "timing_mode": TIMING_MODE,
            "generation_mode_by_method": {
                key: str(value) for key, value in generation_mode_by_method.items()
            },
            "tp_gpus": int(resolved_tp),
            "n_gpus_per_node": int(resolved_n_gpus),
            "non_gpu_overhead_w": float(non_gpu_overhead_w),
            "physics_artifact": (
                file_identity(physics_artifact) if physics_payload is not None else None
            ),
            "selected_artifact": (
                file_identity(selected_artifact) if "ours" in method_list else None
            ),
            "source_revision": git_state(),
        },
        "input_identities": {
            "run_manifest": (
                file_identity(run_manifest) if Path(run_manifest).is_file() else None
            ),
            "experimental_manifest": (
                file_identity(experimental_manifest)
                if Path(experimental_manifest).is_file() else None
            ),
            "throughput": (
                {
                    "source": "run_manifest_bound",
                    "lambda_prefill": throughput["lambda_prefill"],
                    "lambda_decode": throughput["lambda_decode"],
                }
                if throughput is not None else None
            ),
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
    parser.add_argument("--physics-artifact", default=defaults["physics_artifact"])
    parser.add_argument("--selected-artifact", default=defaults["selected_artifact"])
    parser.add_argument("--splitwise-perf-model-csv", default=defaults["splitwise_perf_model_csv"])
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
    args = parser.parse_args()

    summary = generate_node_traces(
        run_manifest=str(args.run_manifest),
        experimental_manifest=str(args.experimental_manifest),
        throughput_db=str(args.throughput_db),
        physics_artifact=str(args.physics_artifact),
        selected_artifact=str(args.selected_artifact),
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
        splitwise_perf_model_csv=str(args.splitwise_perf_model_csv),
        splitwise_source_model=str(args.splitwise_source_model),
        splitwise_source_hardware=str(args.splitwise_source_hardware),
        splitwise_source_tp=int(args.splitwise_source_tp),
        splitwise_style_lut_mode=str(args.splitwise_style_lut_mode),
        tp_gpus=(int(args.tp_gpus) if args.tp_gpus is not None else None),
        n_gpus_per_node=(int(args.n_gpus_per_node) if args.n_gpus_per_node is not None else None),
        non_gpu_overhead_w=float(args.non_gpu_overhead_w),
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
    print(f"Timing mode        : {summary['generation']['timing_mode']}")
    print(f"Manifest           : {summary['trace_manifest_csv']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
