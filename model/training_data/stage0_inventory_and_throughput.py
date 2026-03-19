from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from model.training_data.inventory import (
    MatchedPair,
    discover_dataset_dirs,
    discover_pairs_for_dataset,
)
from model.training_data.throughput import (
    concurrency_binned_decode_medians,
    extract_request_metrics,
    inspect_power_csv,
    select_decode_model_type,
)


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _to_config_id(model_name: str, hardware: str, tp: int) -> str:
    return f"{model_name}_{hardware}_tp{tp}"


def _ensure_parent_dir(path_str: str) -> None:
    parent = Path(path_str).parent
    parent.mkdir(parents=True, exist_ok=True)


def run_stage0_inventory_and_throughput(
    *,
    data_root_dir: str,
    include_families: Sequence[str],
    out_inventory_json: str,
    out_pair_manifest_csv: str,
    out_throughput_db: str,
    min_bin_samples: int,
    batch_variation_threshold: float,
) -> Dict[str, object]:
    families = [f.strip() for f in include_families if f.strip()]
    dataset_dirs = discover_dataset_dirs(data_root_dir, families)

    all_manifest_rows: List[Dict[str, object]] = []
    matched_pairs: List[MatchedPair] = []
    discovery_summaries: List[Dict[str, object]] = []

    for family, dataset_dir in dataset_dirs:
        manifest_rows, local_pairs, discovery_stats = discover_pairs_for_dataset(family, dataset_dir)
        all_manifest_rows.extend(manifest_rows)
        matched_pairs.extend(local_pairs)
        discovery_summaries.append(discovery_stats)

    all_manifest_rows.sort(
        key=lambda r: (
            str(r["family"]),
            str(r["dataset_name"]),
            str(r["status"]),
            str(r["pair_key"]),
        )
    )
    matched_pairs.sort(
        key=lambda p: (
            p.model_name,
            p.hardware,
            p.tensor_parallelism,
            p.family,
            p.dataset_dir,
            p.pair_key,
        )
    )

    per_config = defaultdict(
        lambda: {
            "model_name": "",
            "hardware": "",
            "tensor_parallelism": 0,
            "num_pairs": 0,
            "prefill_rates": [],
            "decode_rates": [],
            "timestamp_windows": [],
            "runs_with_request_timestamps": 0,
            "runs_missing_request_timestamps": 0,
            "runs_with_mismatched_array_lengths": 0,
            "requests_total": 0,
            "requests_aligned": 0,
            "requests_used": 0,
            "requests_dropped_invalid": 0,
            "requests_dropped_decode_invalid": 0,
            "requests_missing_or_invalid_timestamp": 0,
            "requests_timestamp_windows": 0,
        }
    )
    paired_run_diagnostics: List[Dict[str, object]] = []

    for pair in matched_pairs:
        config_key = _to_config_id(pair.model_name, pair.hardware, pair.tensor_parallelism)
        cfg = per_config[config_key]
        cfg["model_name"] = pair.model_name
        cfg["hardware"] = pair.hardware
        cfg["tensor_parallelism"] = int(pair.tensor_parallelism)
        cfg["num_pairs"] += 1

        power_diag = inspect_power_csv(pair.power_csv_path)

        json_payload: Optional[Dict[str, object]] = None
        json_load_error = None
        try:
            with open(pair.json_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    json_payload = loaded
                else:
                    json_load_error = "json payload is not a dict"
        except Exception as exc:
            json_load_error = str(exc)

        if json_payload is None:
            paired_run_diagnostics.append(
                {
                    "family": pair.family,
                    "dataset_dir": pair.dataset_dir,
                    "pair_key": pair.pair_key,
                    "model_name": pair.model_name,
                    "hardware": pair.hardware,
                    "tensor_parallelism": pair.tensor_parallelism,
                    "power_csv_path": pair.power_csv_path,
                    "json_path": pair.json_path,
                    "power_diagnostics": power_diag,
                    "json_error": json_load_error,
                }
            )
            cfg["runs_with_mismatched_array_lengths"] += 1
            continue

        extraction = extract_request_metrics(json_payload)
        schema = extraction["schema"]
        stats = extraction["stats"]

        cfg["prefill_rates"].extend(extraction["prefill_rates"])
        cfg["decode_rates"].extend(extraction["decode_rates"])
        cfg["timestamp_windows"].extend(extraction["timestamp_windows"])

        cfg["requests_total"] += int(stats["num_requests_total"])
        cfg["requests_aligned"] += int(stats["num_requests_aligned"])
        cfg["requests_used"] += int(stats["num_requests_used"])
        cfg["requests_dropped_invalid"] += int(stats["num_requests_dropped_invalid_fields"])
        cfg["requests_dropped_decode_invalid"] += int(stats["num_requests_dropped_decode_time"])
        cfg["requests_missing_or_invalid_timestamp"] += int(
            stats["num_requests_missing_or_invalid_timestamp"]
        )
        cfg["requests_timestamp_windows"] += int(stats["num_timestamp_windows"])

        if bool(stats["missing_request_timestamps_array"]):
            cfg["runs_missing_request_timestamps"] += 1
        else:
            cfg["runs_with_request_timestamps"] += 1
        if bool(stats["mismatched_array_lengths"]):
            cfg["runs_with_mismatched_array_lengths"] += 1

        paired_run_diagnostics.append(
            {
                "family": pair.family,
                "dataset_dir": pair.dataset_dir,
                "pair_key": pair.pair_key,
                "model_name": pair.model_name,
                "hardware": pair.hardware,
                "tensor_parallelism": pair.tensor_parallelism,
                "power_csv_path": pair.power_csv_path,
                "json_path": pair.json_path,
                "power_diagnostics": power_diag,
                "json_schema": schema,
                "request_extraction_stats": stats,
            }
        )

    throughput_configs: Dict[str, Dict[str, object]] = {}
    inventory_config_summary: Dict[str, Dict[str, object]] = {}

    for config_id in sorted(per_config.keys()):
        cfg = per_config[config_id]
        prefill_median = _median_or_none(cfg["prefill_rates"])
        decode_median = _median_or_none(cfg["decode_rates"])

        bins = concurrency_binned_decode_medians(
            cfg["timestamp_windows"],
            min_bin_samples=min_bin_samples,
        )
        decode_model_type, ratio = select_decode_model_type(
            bins,
            batch_variation_threshold=batch_variation_threshold,
        )

        quality_flags = {
            "runs_missing_request_timestamps": int(cfg["runs_missing_request_timestamps"]),
            "runs_with_mismatched_array_lengths": int(cfg["runs_with_mismatched_array_lengths"]),
            "requests_dropped_invalid_fields": int(cfg["requests_dropped_invalid"]),
            "requests_dropped_decode_time": int(cfg["requests_dropped_decode_invalid"]),
            "requests_missing_or_invalid_timestamp": int(
                cfg["requests_missing_or_invalid_timestamp"]
            ),
        }

        throughput_configs[config_id] = {
            "model_name": cfg["model_name"],
            "hardware": cfg["hardware"],
            "tensor_parallelism": int(cfg["tensor_parallelism"]),
            "num_pairs": int(cfg["num_pairs"]),
            "source_run_count": int(cfg["num_pairs"]),
            "num_requests_total": int(cfg["requests_total"]),
            "num_requests_used": int(cfg["requests_used"]),
            "prefill_rate_median_toks_per_s": prefill_median,
            "decode_rate_median_toks_per_s": decode_median,
            "decode_model": {
                "type": decode_model_type,
                "by_concurrency_bins": bins,
                "max_min_median_ratio": ratio,
            },
            "file_counts": {
                "matched_pairs": int(cfg["num_pairs"]),
                "runs_with_request_timestamps": int(cfg["runs_with_request_timestamps"]),
                "runs_missing_request_timestamps": int(cfg["runs_missing_request_timestamps"]),
            },
            "quality_flags": quality_flags,
            "anomaly_counters": quality_flags,
        }

        inventory_config_summary[config_id] = {
            "model_name": cfg["model_name"],
            "hardware": cfg["hardware"],
            "tensor_parallelism": int(cfg["tensor_parallelism"]),
            "num_pairs": int(cfg["num_pairs"]),
            "num_requests_total": int(cfg["requests_total"]),
            "num_requests_aligned": int(cfg["requests_aligned"]),
            "num_requests_used": int(cfg["requests_used"]),
            "runs_with_request_timestamps": int(cfg["runs_with_request_timestamps"]),
            "runs_missing_request_timestamps": int(cfg["runs_missing_request_timestamps"]),
            "runs_with_mismatched_array_lengths": int(cfg["runs_with_mismatched_array_lengths"]),
            "num_timestamp_windows": int(cfg["requests_timestamp_windows"]),
        }

    directory_summary: Dict[str, Dict[str, object]] = {}
    for d in discovery_summaries:
        dataset_dir = str(d["dataset_dir"])
        directory_summary[dataset_dir] = {
            "family": d["family"],
            "dataset_name": d["dataset_name"],
            "num_json_files": int(d["num_json_files"]),
            "num_power_csv_files": int(d["num_power_csv_files"]),
            "num_manifest_rows": int(d["num_manifest_rows"]),
            "num_matched_pairs": int(d["num_matched_pairs"]),
            "num_json_only": int(d["num_json_only"]),
            "num_power_only": int(d["num_power_only"]),
            "duplicate_json_keys_dropped": int(d["duplicate_json_keys_dropped"]),
            "duplicate_power_keys_dropped": int(d["duplicate_power_keys_dropped"]),
            "ignored_files": int(d["ignored_files"]),
        }

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    throughput_db = {
        "schema_version": "stage0-throughput-v1",
        "generated_at_utc": generated_at,
        "defaults": {
            "min_bin_samples": int(min_bin_samples),
            "batch_variation_threshold": float(batch_variation_threshold),
            "decode_fallback_policy": "constant_when_timestamps_missing_or_bins_not_selected",
        },
        "configs": throughput_configs,
    }

    inventory = {
        "schema_version": "stage0-inventory-v1",
        "generated_at_utc": generated_at,
        "data_root_dir": data_root_dir,
        "include_families": list(families),
        "summary": {
            "num_dataset_dirs": len(dataset_dirs),
            "num_manifest_rows": len(all_manifest_rows),
            "num_matched_pairs": len(matched_pairs),
            "num_unique_configs": len(throughput_configs),
            "num_paired_runs_with_diagnostics": len(paired_run_diagnostics),
        },
        "directories": directory_summary,
        "configs": inventory_config_summary,
        "paired_runs": paired_run_diagnostics,
    }

    _ensure_parent_dir(out_pair_manifest_csv)
    _ensure_parent_dir(out_inventory_json)
    _ensure_parent_dir(out_throughput_db)

    manifest_fields = [
        "family",
        "dataset_dir",
        "dataset_name",
        "status",
        "model_name",
        "hardware",
        "tensor_parallelism",
        "rate",
        "iteration",
        "date_key",
        "pair_key",
        "power_csv_path",
        "json_path",
    ]
    with open(out_pair_manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for row in all_manifest_rows:
            writer.writerow(row)

    with open(out_inventory_json, "w") as f:
        json.dump(inventory, f, indent=2, sort_keys=True)

    with open(out_throughput_db, "w") as f:
        json.dump(throughput_db, f, indent=2, sort_keys=True)

    return {
        "inventory": inventory,
        "throughput_db": throughput_db,
        "manifest_rows": all_manifest_rows,
    }


def _parse_include_families(value: str) -> List[str]:
    if not value:
        return ["benchmark", "sharegpt-benchmark"]
    families = [x.strip() for x in value.split(",") if x.strip()]
    valid = {"benchmark", "sharegpt-benchmark"}
    bad = [x for x in families if x not in valid]
    if bad:
        raise ValueError(f"Unsupported families: {bad}. Valid values: {sorted(valid)}")
    return families


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 0 inventory + throughput extraction for benchmark/sharegpt-benchmark data."
    )
    parser.add_argument(
        "--data_root_dir",
        required=True,
        help="Root data directory (e.g. data).",
    )
    parser.add_argument(
        "--include_families",
        default="benchmark,sharegpt-benchmark",
        help="Comma-separated families to include (benchmark,sharegpt-benchmark).",
    )
    parser.add_argument(
        "--out_inventory_json",
        default="results/stage0/data_inventory.json",
        help="Output inventory JSON path.",
    )
    parser.add_argument(
        "--out_pair_manifest_csv",
        default="results/stage0/pair_manifest.csv",
        help="Output pair manifest CSV path.",
    )
    parser.add_argument(
        "--out_throughput_db",
        default="model/config/throughput_database.json",
        help="Output throughput database JSON path.",
    )
    parser.add_argument(
        "--min_bin_samples",
        type=int,
        default=50,
        help="Minimum samples per concurrency bin.",
    )
    parser.add_argument(
        "--batch_variation_threshold",
        type=float,
        default=2.0,
        help="Threshold on max/min median decode throughput to select by_concurrency model.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    families = _parse_include_families(args.include_families)
    run_stage0_inventory_and_throughput(
        data_root_dir=args.data_root_dir,
        include_families=families,
        out_inventory_json=args.out_inventory_json,
        out_pair_manifest_csv=args.out_pair_manifest_csv,
        out_throughput_db=args.out_throughput_db,
        min_bin_samples=args.min_bin_samples,
        batch_variation_threshold=args.batch_variation_threshold,
    )


if __name__ == "__main__":
    main()
