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
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _power_timestamp_to_epoch(ts_text: str) -> Optional[float]:
    """Parse power CSV timestamp to Unix epoch seconds."""
    text = ts_text.strip()
    formats = (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("/", "-")).timestamp()
    except ValueError:
        return None


def _parse_power_csv(
    csv_path: str,
    tensor_parallelism: int,
    gpus_per_node: int = 8,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse power CSV and aggregate across GPUs.

    Returns:
        Dict with 'timestamps' (epoch seconds) and 'power' (watts) arrays.
    """
    try:
        timestamps: List[float] = []
        power_values: List[float] = []
        current_group_ts: Optional[float] = None
        current_group_power: List[float] = []
        row_in_group = 0

        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            header_lower = [h.strip().lower() for h in header]

            ts_col = None
            power_col = None
            for i, h in enumerate(header_lower):
                if "time" in h and ts_col is None:
                    ts_col = i
                if "power" in h and "draw" in h:
                    power_col = i

            if ts_col is None or power_col is None:
                return None

            for row in reader:
                if len(row) <= max(ts_col, power_col):
                    continue

                ts = _power_timestamp_to_epoch(row[ts_col])
                if ts is None:
                    continue

                power_str = re.sub(r"[^\d.]", "", row[power_col])
                try:
                    power = float(power_str)
                except ValueError:
                    continue

                if current_group_ts is None:
                    current_group_ts = ts
                    current_group_power = [power]
                    row_in_group = 1
                elif ts == current_group_ts:
                    current_group_power.append(power)
                    row_in_group += 1
                else:
                    if row_in_group >= tensor_parallelism:
                        agg_power = sum(current_group_power[:tensor_parallelism])
                        timestamps.append(current_group_ts)
                        power_values.append(agg_power)
                    current_group_ts = ts
                    current_group_power = [power]
                    row_in_group = 1

            if current_group_ts is not None and row_in_group >= tensor_parallelism:
                agg_power = sum(current_group_power[:tensor_parallelism])
                timestamps.append(current_group_ts)
                power_values.append(agg_power)

        if len(timestamps) < 2:
            return None

        return {
            "timestamps": np.array(timestamps, dtype=np.float64),
            "power": np.array(power_values, dtype=np.float64),
        }
    except Exception:
        return None


def _derive_decode_time(itl_value: object, output_tokens: object) -> Optional[float]:
    """Derive decode time from ITL value(s)."""
    if isinstance(itl_value, list):
        if len(itl_value) == 0:
            return None
        try:
            arr = np.asarray(itl_value, dtype=float)
        except Exception:
            return None
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return float(np.sum(arr))

    if isinstance(itl_value, (int, float)) and not isinstance(itl_value, bool):
        if not np.isfinite(float(itl_value)):
            return None
        try:
            out_tok = int(float(output_tokens))
        except Exception:
            return None
        return float(itl_value) * float(max(out_tok - 1, 1))

    return None


def _parse_request_json(json_path: str) -> Optional[Dict[str, object]]:
    """
    Parse benchmark JSON to extract request data.

    Returns:
        Dict with input_lens, output_lens, request_timestamps, ttfts, decode_times arrays.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        input_lens = data.get("input_lens", [])
        output_lens = data.get("output_lens", [])
        ttfts = data.get("ttfts", [])
        itls = data.get("itls", [])
        request_timestamps = data.get("request_timestamps", [])

        if not all(isinstance(x, list) for x in [input_lens, output_lens, ttfts, itls]):
            return None

        n = min(len(input_lens), len(output_lens), len(ttfts), len(itls))
        if n == 0:
            return None

        has_timestamps = isinstance(request_timestamps, list) and len(request_timestamps) >= n
        if has_timestamps:
            n = min(n, len(request_timestamps))

        valid_input: List[float] = []
        valid_output: List[float] = []
        valid_ttft: List[float] = []
        valid_decode: List[float] = []
        valid_ts: List[float] = []

        for i in range(n):
            in_tok = input_lens[i]
            out_tok = output_lens[i]
            ttft = ttfts[i]
            decode_time = _derive_decode_time(itls[i], out_tok)

            if decode_time is None or decode_time <= 0:
                continue
            if ttft is None or not np.isfinite(float(ttft)) or float(ttft) <= 0:
                continue
            if not (np.isfinite(float(in_tok)) and np.isfinite(float(out_tok))):
                continue

            valid_input.append(float(in_tok))
            valid_output.append(float(out_tok))
            valid_ttft.append(float(ttft))
            valid_decode.append(float(decode_time))

            if has_timestamps:
                ts = request_timestamps[i]
                if np.isfinite(float(ts)) and float(ts) > 0:
                    valid_ts.append(float(ts))
                else:
                    valid_ts.append(float("nan"))
            else:
                valid_ts.append(float("nan"))

        if len(valid_input) == 0:
            return None

        return {
            "input_lens": np.array(valid_input, dtype=np.float64),
            "output_lens": np.array(valid_output, dtype=np.float64),
            "ttfts": np.array(valid_ttft, dtype=np.float64),
            "decode_times": np.array(valid_decode, dtype=np.float64),
            "request_timestamps": np.array(valid_ts, dtype=np.float64),
            "has_timestamps": has_timestamps and np.all(np.isfinite(valid_ts)),
        }
    except Exception:
        return None


def _compute_active_requests(
    power_timestamps: np.ndarray,
    request_timestamps: np.ndarray,
    ttfts: np.ndarray,
    decode_times: np.ndarray,
) -> np.ndarray:
    """Compute active request count at each power measurement time."""
    n_power = len(power_timestamps)
    active = np.zeros(n_power, dtype=np.float64)

    if len(request_timestamps) == 0:
        return active

    start_times = request_timestamps
    end_times = request_timestamps + ttfts + decode_times

    for i, t in enumerate(power_timestamps):
        count = np.sum((start_times <= t) & (t <= end_times))
        active[i] = float(count)

    return active


def _compute_t_arrive_log(
    power_timestamps: np.ndarray,
    request_timestamps: np.ndarray,
) -> np.ndarray:
    """Compute log(1 + inter-arrival time) for new arrivals at each power measurement."""
    n_power = len(power_timestamps)
    t_arrive_log = np.zeros(n_power, dtype=np.float64)

    if len(request_timestamps) == 0 or len(power_timestamps) < 2:
        return t_arrive_log

    dt = float(np.median(np.diff(power_timestamps)))
    sorted_arrivals = np.sort(request_timestamps)

    for i, t in enumerate(power_timestamps):
        if i == 0:
            interval_start = t - dt / 2
        else:
            interval_start = (t + power_timestamps[i - 1]) / 2

        if i == n_power - 1:
            interval_end = t + dt / 2
        else:
            interval_end = (t + power_timestamps[i + 1]) / 2

        arrivals_in_interval = sorted_arrivals[
            (sorted_arrivals >= interval_start) & (sorted_arrivals < interval_end)
        ]

        if len(arrivals_in_interval) > 0:
            first_arrival = arrivals_in_interval[0]
            idx = np.searchsorted(sorted_arrivals, first_arrival)
            if idx > 0:
                inter_arrival = first_arrival - sorted_arrivals[idx - 1]
                t_arrive_log[i] = np.log1p(max(0.0, inter_arrival))

    return t_arrive_log


def _align_trace_to_grid(
    power_data: Dict[str, np.ndarray],
    request_data: Dict[str, object],
    power_start_offset_s: float = 0.0,
) -> Optional[Dict[str, object]]:
    """
    Align power trace and request data to a common time grid.

    Returns:
        Dict with power, active_requests, t_arrive_log arrays and metadata.
    """
    timestamps = power_data["timestamps"]
    power = power_data["power"]

    if len(timestamps) < 3:
        return None

    request_ts = np.asarray(request_data["request_timestamps"], dtype=np.float64)
    ttfts = np.asarray(request_data["ttfts"], dtype=np.float64)
    decode_times = np.asarray(request_data["decode_times"], dtype=np.float64)

    if not request_data.get("has_timestamps", False):
        power_start = float(timestamps[0])
        power_end = float(timestamps[-1])
        duration = power_end - power_start
        n_requests = len(request_ts)
        if n_requests > 1:
            spacing = duration / (n_requests + 1)
            request_ts = power_start + spacing * (np.arange(n_requests) + 1)
        elif n_requests == 1:
            request_ts = np.array([power_start + duration / 2])
        else:
            request_ts = np.array([])

    active = _compute_active_requests(timestamps, request_ts, ttfts, decode_times)
    t_arrive_log = _compute_t_arrive_log(timestamps, request_ts)

    if not (np.all(np.isfinite(power)) and np.all(np.isfinite(active))):
        return None

    dt_values = np.diff(timestamps)
    dt = float(np.median(dt_values)) if len(dt_values) > 0 else 0.25

    return {
        "power": power,
        "active_requests": active,
        "t_arrive_log": t_arrive_log,
        "timestamps": timestamps,
        "dt": dt,
        "power_start_epoch_s": float(timestamps[0]),
        "num_points": len(power),
        "input_lens": request_data["input_lens"],
        "output_lens": request_data["output_lens"],
    }


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


def _compute_normalization_stats(
    traces: List[Dict[str, object]],
) -> Dict[str, float]:
    """Compute normalization statistics across all traces for a config."""
    all_power: List[np.ndarray] = []
    all_active: List[np.ndarray] = []
    all_t_arrive_log: List[np.ndarray] = []

    for tr in traces:
        all_power.append(tr["power"])
        all_active.append(tr["active_requests"])
        all_t_arrive_log.append(tr["t_arrive_log"])

    power_cat = np.concatenate(all_power)
    active_cat = np.concatenate(all_active)
    t_log_cat = np.concatenate(all_t_arrive_log)

    return {
        "power_mean": float(np.mean(power_cat)),
        "power_std": float(np.std(power_cat) + 1e-6),
        "power_min": float(np.min(power_cat)),
        "power_max": float(np.max(power_cat)),
        "active_mean": float(np.mean(active_cat)),
        "active_std": float(np.std(active_cat) + 1e-6),
        "t_arrive_log_mean": float(np.mean(t_log_cat)),
        "t_arrive_log_std": float(np.std(t_log_cat) + 1e-6),
    }


def _create_train_val_test_split(
    n_traces: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Create train/val/test split indices."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_traces).tolist()

    n_train = max(1, int(n_traces * train_ratio))
    n_val = max(1, int(n_traces * val_ratio))

    if n_traces <= 2:
        return {
            "train_indices": [0] if n_traces >= 1 else [],
            "val_indices": [min(1, n_traces - 1)] if n_traces >= 1 else [],
            "test_indices": [min(1, n_traces - 1)] if n_traces >= 1 else [],
        }

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]

    if len(val_indices) == 0:
        val_indices = [train_indices[-1]] if train_indices else []
    if len(test_indices) == 0:
        test_indices = val_indices.copy()

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }


def run_prepare_experimental_manifest(
    *,
    pair_manifest_csv: str,
    out_dir: str = "results/experimental_continuous_v1",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_traces_per_config: int = 2,
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

            power_data = _parse_power_csv(power_csv, tensor_parallelism=tp)
            if power_data is None:
                errors.append(f"power_parse_failed:{pair_key}")
                skipped += 1
                continue

            request_data = _parse_request_json(json_path)
            if request_data is None:
                errors.append(f"json_parse_failed:{pair_key}")
                skipped += 1
                continue

            aligned = _align_trace_to_grid(power_data, request_data)
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

        norm_stats = _compute_normalization_stats(traces)
        split = _create_train_val_test_split(len(traces), train_ratio, val_ratio, seed)

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
    )

    print("[prepare_experimental_manifest] Summary:")
    for k, v in manifest.get("summary", {}).items():
        print(f"  {k}: {v}")
    print(f"  manifest: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
