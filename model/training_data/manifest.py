#!/usr/bin/env python3
"""
Prepare experimental manifest from Stage0 output.

This script bridges the gap between Stage0 (pair_manifest.csv, throughput_database.json)
and the GMM-BiGRU training pipeline (experimental_continuous_v1/manifest.json).

Pipeline: raw data -> Stage0 -> THIS SCRIPT -> train_gmm_bigru.py -> eval_gmm_bigru.py
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from model.training_data.normalization import (
    compute_normalization_stats,
    create_train_val_test_split,
)
from model.training_data.alignment import resample_trace_to_grid
from model.training_data.run_record import (
    RunRecord,
    gru_view_from_record,
    load_bundle_run,
    load_legacy_run,
)
from model.utils.io import (
    ensure_dir as _ensure_dir,
    safe_slug as _safe_slug,
    write_json as _write_json,
)


CONFIG_TIMESTEP_TOLERANCE_S = 1e-6


def _load_pair_manifest_csv(csv_path: str) -> List[Dict[str, str]]:
    """Load pair manifest CSV from Stage0."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "").strip() == "matched":
                rows.append(row)
    return rows


def _group_pairs_by_config(
    pairs: List[Dict[str, str]],
) -> Dict[str, List[Dict[str, str]]]:
    """Group matched pairs by config_id (model_name_hardware_tp)."""
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for pair in pairs:
        model_name = pair.get("model_name", "").strip()
        hardware = pair.get("hardware", "").strip()
        tp = pair.get("tensor_parallelism", "").strip()
        if model_name and hardware and tp:
            config_id = f"{model_name}_{hardware}_tp{tp}"
            grouped[config_id].append(pair)
    return dict(grouped)


def _lineage_entry(
    record: RunRecord,
    *,
    trace_index: int,
    trace: Dict[str, object],
) -> Dict[str, object]:
    provenance = record.provenance
    request_rows = provenance.get("request_rows")
    if not isinstance(request_rows, dict):
        raise ValueError("RunRecord provenance is missing request_rows accounting")
    request_projection = provenance.get("request_projection")
    if not isinstance(request_projection, dict):
        raise ValueError("RunRecord provenance is missing request_projection accounting")
    request_projection_indices = provenance.get("request_projection_indices")
    if not isinstance(request_projection_indices, list):
        raise ValueError("RunRecord provenance is missing request projection indices")
    if record.source_layout == "sharegpt":
        source_paths = {
            "power_csv": provenance["power_csv_path"],
            "requests_json": provenance["json_path"],
        }
    elif record.source_layout == "bundle":
        source_paths = dict(provenance["paths"])
    else:
        raise ValueError(f"Unknown source layout: {record.source_layout}")
    return {
        "trace_index": int(trace_index),
        "source_layout": record.source_layout,
        "source_paths": source_paths,
        "source_sha256": dict(provenance["sha256"]),
        "request_rows": request_rows,
        "request_projection": request_projection,
        "request_projection_indices": request_projection_indices,
        "projected_samples": {
            "power": int(len(trace["power"])),
            "active_requests": int(len(trace["active_requests"])),
            "t_arrive_log": int(len(trace["t_arrive_log"])),
        },
    }


def _fit_training_throughput(traces: List[Dict[str, object]]) -> Dict[str, float]:
    prefill_rates: List[np.ndarray] = []
    decode_rates: List[np.ndarray] = []
    for trace in traces:
        n_in = np.asarray(trace["input_lens"], dtype=np.float64)
        n_out = np.asarray(trace["output_lens"], dtype=np.float64)
        ttft = np.asarray(trace["ttfts"], dtype=np.float64)
        decode = np.asarray(trace["decode_times"], dtype=np.float64)
        prefill = n_in / ttft
        valid_prefill = np.isfinite(prefill) & (prefill > 0.0)
        valid_decode = np.isfinite(decode) & (decode > 0.0) & (n_out > 1.0)
        prefill_rates.append(prefill[valid_prefill])
        decode_rates.append(n_out[valid_decode] / decode[valid_decode])
    prefill = np.concatenate(prefill_rates)
    decode = np.concatenate(decode_rates)
    if prefill.size == 0 or decode.size == 0:
        raise ValueError("Training split cannot calibrate positive prefill/decode throughput")
    return {
        "lambda_prefill": float(np.median(prefill)),
        "lambda_decode": float(np.median(decode)),
    }


def run_prepare_experimental_manifest(
    *,
    pair_manifest_csv: str,
    out_dir: str = "results/experimental_continuous_v1",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_traces_per_config: int = 3,
    require_request_timestamps: bool = True,
    bundle_dirs: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Prepare experimental manifest from Stage0 pair manifest.

    Args:
        pair_manifest_csv: Path to Stage0 pair_manifest.csv
        out_dir: Output directory for experimental manifest and data
        train_ratio: Fraction of traces for training
        val_ratio: Fraction of traces for validation
        seed: Random seed for splits
        min_traces_per_config: Minimum traces required per config
        require_request_timestamps: Require recorded request_timestamps in JSON.
        bundle_dirs: Explicit canonical bundle directories to include.

    Returns:
        Manifest dict written to out_dir/manifest.json
    """
    _ensure_dir(out_dir)
    datasets_dir = os.path.join(out_dir, "datasets")
    splits_dir = os.path.join(out_dir, "splits")
    norms_dir = os.path.join(out_dir, "norm_params")
    _ensure_dir(datasets_dir)
    _ensure_dir(splits_dir)
    _ensure_dir(norms_dir)

    pairs = _load_pair_manifest_csv(pair_manifest_csv)
    grouped = _group_pairs_by_config(pairs)
    bundle_records: Dict[str, List[RunRecord]] = defaultdict(list)
    for bundle_dir in bundle_dirs or []:
        record = load_bundle_run(bundle_dir)
        bundle_records[record.config_id].append(record)

    manifest_configs: Dict[str, Dict[str, object]] = {}
    processing_summary: Dict[str, Dict[str, object]] = {}

    config_ids = sorted(set(grouped) | set(bundle_records))
    for config_id in config_ids:
        config_pairs = grouped.get(config_id, [])
        config_bundles = bundle_records.get(config_id, [])
        traces: List[Dict[str, object]] = []
        pair_keys: List[str] = []
        rates: List[str] = []
        lineage_rows: List[Dict[str, object]] = []
        skipped = 0
        errors: List[str] = []

        for pair in config_pairs:
            power_csv = pair.get("power_csv_path", "")
            json_path = pair.get("json_path", "")
            pair_key = pair.get("pair_key", "")
            rate = pair.get("rate", "")
            if not (power_csv and json_path and os.path.exists(power_csv) and os.path.exists(json_path)):
                skipped += 1
                continue

            record = load_legacy_run(
                pair,
                require_request_timestamps=bool(require_request_timestamps),
                require_arch=False,
            )
            if record is None:
                errors.append(f"run_record_parse_failed:{pair_key}")
                skipped += 1
                continue

            aligned = gru_view_from_record(record)
            if aligned is None:
                errors.append(f"alignment_failed:{pair_key}")
                skipped += 1
                continue

            traces.append(aligned)
            pair_keys.append(pair_key)
            rates.append(rate)
            try:
                lineage_rows.append(
                    _lineage_entry(record, trace_index=len(traces) - 1, trace=aligned)
                )
            except (KeyError, TypeError, ValueError) as exc:
                traces.pop()
                pair_keys.pop()
                rates.pop()
                errors.append(f"lineage_failed:{pair_key}:{exc}")
                skipped += 1

        for record in config_bundles:
            pair_key = str(record.provenance.get("run_id", ""))
            aligned = gru_view_from_record(record)
            if aligned is None:
                errors.append(f"alignment_failed:{pair_key}")
                skipped += 1
                continue
            traces.append(aligned)
            pair_keys.append(pair_key)
            rates.append(str(record.provenance.get("rate", "")))
            try:
                lineage_rows.append(
                    _lineage_entry(record, trace_index=len(traces) - 1, trace=aligned)
                )
            except (KeyError, TypeError, ValueError) as exc:
                traces.pop()
                pair_keys.pop()
                rates.pop()
                errors.append(f"lineage_failed:{pair_key}:{exc}")
                skipped += 1

        processing_summary[config_id] = {
            "num_pairs": len(config_pairs),
            "num_bundles": len(config_bundles),
            "num_traces": len(traces),
            "skipped": skipped,
            "errors": errors,
        }

        required_traces = max(3, int(min_traces_per_config))
        if len(traces) < required_traces:
            manifest_configs[config_id] = {
                "written": False,
                "reason": f"insufficient_traces:{len(traces)}<{required_traces}",
            }
            continue

        dt_values = np.asarray([tr["dt"] for tr in traces], dtype=np.float64)
        dt = float(np.median(dt_values))
        if not np.all(np.isfinite(dt_values)) or not np.all(
            np.isclose(dt_values, dt, rtol=0.01, atol=CONFIG_TIMESTEP_TOLERANCE_S)
        ):
            raise ValueError(
                f"Config {config_id} mixes incompatible sampling intervals: {dt_values.tolist()}"
            )
        traces = [resample_trace_to_grid(trace, dt=dt) for trace in traces]
        for trace, lineage in zip(traces, lineage_rows):
            lineage["projected_samples"] = {
                "power": int(len(trace["power"])),
                "active_requests": int(len(trace["active_requests"])),
                "t_arrive_log": int(len(trace["t_arrive_log"])),
            }

        split = create_train_val_test_split(len(traces), train_ratio, val_ratio, seed)
        train_traces = [traces[i] for i in split["train_indices"]]
        norm_stats = compute_normalization_stats(train_traces)
        throughput = _fit_training_throughput(train_traces)

        slug = _safe_slug(config_id)

        dataset_path = os.path.join(datasets_dir, f"{slug}.npz")
        np.savez(
            dataset_path,
            config_id=np.array([config_id], dtype=object),
            dt=np.array([dt], dtype=np.float64),
            pair_key=np.asarray(pair_keys, dtype=object),
            rate=np.asarray(rates, dtype=object),
            power=np.asarray([tr["power"] for tr in traces], dtype=object),
            power_start_epoch_s=np.array(
                [tr["power_start_epoch_s"] for tr in traces], dtype=np.float64
            ),
            active_requests=np.asarray(
                [tr["active_requests"] for tr in traces], dtype=object
            ),
            t_arrive_log=np.asarray([tr["t_arrive_log"] for tr in traces], dtype=object),
            input_lens=np.asarray([tr["input_lens"] for tr in traces], dtype=object),
            output_lens=np.asarray([tr["output_lens"] for tr in traces], dtype=object),
            ttfts=np.asarray([tr["ttfts"] for tr in traces], dtype=object),
            decode_times=np.asarray([tr["decode_times"] for tr in traces], dtype=object),
        )

        split_path = os.path.join(splits_dir, f"{slug}.json")
        _write_json(
            split_path,
            {
                "config_id": config_id,
                **split,
            },
        )

        norm_path = os.path.join(norms_dir, f"{slug}.json")
        _write_json(
            norm_path,
            {
                "config_id": config_id,
                "dt": dt,
                "fit_split": "train",
                **norm_stats,
            },
        )

        lineage_path = os.path.join(datasets_dir, f"{slug}.lineage.json")
        _write_json(
            lineage_path,
            {
                "schema_version": "gru-dataset-lineage-v1",
                "config_id": config_id,
                "projection": {
                    "source_contract": "RunRecord",
                    "config_timestep_s": dt,
                    "stored_fields": [
                        "power", "active_requests", "t_arrive_log", "input_lens",
                        "output_lens", "ttfts", "decode_times",
                    ],
                    "not_copied_fields": [
                        "power_per_gpu",
                        "util_per_gpu",
                        "mem_per_gpu",
                        "device_ids",
                        "device_table",
                        "request_timestamps",
                        "request_table",
                        "engine_table",
                        "arch",
                    ],
                },
                "traces": lineage_rows,
            },
        )

        manifest_configs[config_id] = {
            "written": True,
            "dataset_npz": dataset_path,
            "split_json": split_path,
            "norm_params_json": norm_path,
            "lineage_json": lineage_path,
            "num_traces": len(traces),
            "num_train": len(split["train_indices"]),
            "num_val": len(split["val_indices"]),
            "num_test": len(split["test_indices"]),
            "throughput": throughput,
            "throughput_fit_split": "train",
        }

    manifest = {
        "schema_version": "experimental-continuous-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "pair_manifest_csv": pair_manifest_csv,
            "bundle_dirs": list(bundle_dirs or []),
        },
        "defaults": {
            "out_dir": out_dir,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "seed": seed,
            "min_traces_per_config": min_traces_per_config,
            "require_request_timestamps": bool(require_request_timestamps),
        },
        "summary": {
            "num_configs_total": len(config_ids),
            "num_configs_written": sum(
                1 for c in manifest_configs.values() if c.get("written", False)
            ),
            "num_configs_skipped": sum(
                1 for c in manifest_configs.values() if not c.get("written", False)
            ),
        },
        "configs": manifest_configs,
        "processing_summary": processing_summary,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    _write_json(manifest_path, manifest)

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare experimental manifest from Stage0 output for GMM-BiGRU training."
    )
    parser.add_argument(
        "--pair-manifest-csv",
        default="results/stage0/pair_manifest.csv",
        help="Path to Stage0 pair_manifest.csv",
    )
    parser.add_argument(
        "--bundle-dir",
        action="append",
        default=[],
        help="Explicit canonical run-bundle directory; may be repeated.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/experimental_continuous_v1",
        help="Output directory for experimental manifest and data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of traces for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of traces for validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test splits (default: 42)",
    )
    parser.add_argument(
        "--min-traces",
        type=int,
        default=3,
        help="Minimum traces per config to include (default: 3)",
    )
    parser.add_argument(
        "--allow-synthetic-request-timestamps",
        action="store_true",
        help="Allow traces without recorded request_timestamps (disabled by default).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = run_prepare_experimental_manifest(
        pair_manifest_csv=args.pair_manifest_csv,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_traces_per_config=args.min_traces,
        require_request_timestamps=not bool(args.allow_synthetic_request_timestamps),
        bundle_dirs=args.bundle_dir,
    )

    print("[prepare_experimental_manifest] Summary:")
    for k, v in manifest.get("summary", {}).items():
        print(f"  {k}: {v}")
    print(f"  manifest: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
