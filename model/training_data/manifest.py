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
from typing import Dict, List

import numpy as np

from model.training_data.alignment import align_trace_to_grid
from model.training_data.normalization import (
    compute_normalization_stats,
    create_train_val_test_split,
)
from model.training_data.power_parsing import parse_power_csv, parse_request_json
from model.utils.io import (
    ensure_dir as _ensure_dir,
    safe_slug as _safe_slug,
    write_json as _write_json,
)


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


def run_prepare_experimental_manifest(
    *,
    pair_manifest_csv: str,
    out_dir: str = "results/experimental_continuous_v1",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_traces_per_config: int = 2,
    require_request_timestamps: bool = True,
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

    manifest_configs: Dict[str, Dict[str, object]] = {}
    processing_summary: Dict[str, Dict[str, object]] = {}

    for config_id, config_pairs in sorted(grouped.items()):
        traces: List[Dict[str, object]] = []
        pair_keys: List[str] = []
        rates: List[str] = []
        skipped = 0
        errors: List[str] = []

        for pair in config_pairs:
            power_csv = pair.get("power_csv_path", "")
            json_path = pair.get("json_path", "")
            pair_key = pair.get("pair_key", "")
            rate = pair.get("rate", "")
            tp = int(pair.get("tensor_parallelism", 1))

            if not (power_csv and json_path and os.path.exists(power_csv) and os.path.exists(json_path)):
                skipped += 1
                continue

            power_data = parse_power_csv(power_csv, tensor_parallelism=tp)
            if power_data is None:
                errors.append(f"power_parse_failed:{pair_key}")
                skipped += 1
                continue

            request_data = parse_request_json(
                json_path,
                require_request_timestamps=bool(require_request_timestamps),
            )
            if request_data is None:
                errors.append(f"json_parse_failed:{pair_key}")
                skipped += 1
                continue

            aligned = align_trace_to_grid(power_data, request_data)
            if aligned is None:
                errors.append(f"alignment_failed:{pair_key}")
                skipped += 1
                continue

            traces.append(aligned)
            pair_keys.append(pair_key)
            rates.append(rate)

        processing_summary[config_id] = {
            "num_pairs": len(config_pairs),
            "num_traces": len(traces),
            "skipped": skipped,
            "errors": errors[:10] if errors else [],
        }

        if len(traces) < min_traces_per_config:
            manifest_configs[config_id] = {
                "written": False,
                "reason": f"insufficient_traces:{len(traces)}<{min_traces_per_config}",
            }
            continue

        dt_values = [tr["dt"] for tr in traces]
        dt = float(np.median(dt_values))

        norm_stats = compute_normalization_stats(traces)
        split = create_train_val_test_split(len(traces), train_ratio, val_ratio, seed)

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
                **norm_stats,
            },
        )

        manifest_configs[config_id] = {
            "written": True,
            "dataset_npz": dataset_path,
            "split_json": split_path,
            "norm_params_json": norm_path,
            "num_traces": len(traces),
            "num_train": len(split["train_indices"]),
            "num_val": len(split["val_indices"]),
            "num_test": len(split["test_indices"]),
        }

    manifest = {
        "schema_version": "experimental-continuous-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "pair_manifest_csv": pair_manifest_csv,
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
            "num_configs_total": len(grouped),
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
        default=2,
        help="Minimum traces per config to include (default: 2)",
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
    )

    print("[prepare_experimental_manifest] Summary:")
    for k, v in manifest.get("summary", {}).items():
        print(f"  {k}: {v}")
    print(f"  manifest: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
