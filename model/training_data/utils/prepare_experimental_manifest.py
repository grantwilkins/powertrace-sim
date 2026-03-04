#!/usr/bin/env python3
"""
Prepare experimental manifest from Stage0 output.

This script bridges the gap between Stage0 (pair_manifest.csv, throughput_database.json)
and the GMM-BiGRU training pipeline (experimental_continuous_v1_gru_all/manifest.json).

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

from model.scripts.request_data_policy import (
    DEFAULT_ALLOWED_JSON_PREFIX,
    DEFAULT_REQUEST_TIMESTAMP_POLICY,
    REQUEST_TIMESTAMP_POLICIES,
    load_pair_manifest_map_with_policy,
    normalize_request_timestamp_policy,
    request_timestamp_policy_requires_recorded,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _resolve_existing_path(path_str: str, base_dir: str) -> Optional[str]:
    raw = Path(str(path_str).strip())
    if raw.is_absolute():
        return str(raw) if raw.exists() else None
    local = Path(path_str)
    if local.exists():
        return str(local)
    from_base = Path(base_dir) / raw
    if from_base.exists():
        return str(from_base)
    return None


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
        raw_timestamps: List[float] = []
        raw_powers: List[float] = []

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
                raw_timestamps.append(float(ts))
                raw_powers.append(float(power))

        if len(raw_timestamps) < 2:
            return None

        tp = int(tensor_parallelism)
        if tp <= 0:
            return None
        gpn = int(gpus_per_node)
        if gpn <= 0:
            return None

        raw_t = np.asarray(raw_timestamps, dtype=np.float64)
        raw_p = np.asarray(raw_powers, dtype=np.float64)
        diffs = np.diff(raw_t)
        abs_diffs = np.abs(diffs) if diffs.size > 0 else np.asarray([], dtype=np.float64)
        # Detect raw nvidia-smi per-GPU streams (8 rows/sample, small inter-row deltas).
        # These files often have timestamp jitter, so grouping by exact timestamp is unstable.
        looks_raw_per_gpu = bool(
            raw_t.size >= gpn
            and abs_diffs.size > 0
            and (
                float(np.median(abs_diffs)) < 0.05
                or float(np.mean(abs_diffs < 0.05)) >= 0.5
            )
        )

        timestamps: List[float] = []
        power_values: List[float] = []
        if looks_raw_per_gpu:
            num_rows = int(raw_t.size)
            num_blocks = num_rows // gpn
            if num_blocks <= 0:
                return None
            usable = int(num_blocks * gpn)
            block_t = raw_t[:usable].reshape(num_blocks, gpn)
            block_p = raw_p[:usable].reshape(num_blocks, gpn)
            if tp > gpn:
                return None
            agg_t = block_t[:, 0]
            agg_p = np.sum(block_p[:, :tp], axis=1, dtype=np.float64)
            timestamps = agg_t.astype(np.float64).tolist()
            power_values = agg_p.astype(np.float64).tolist()
        else:
            # Already-aggregated traces: keep one row per sample as-is.
            timestamps = raw_t.astype(np.float64).tolist()
            power_values = raw_p.astype(np.float64).tolist()

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


def _parse_request_json(
    json_path: str,
    *,
    require_recorded_timestamps: bool,
) -> Tuple[Optional[Dict[str, object]], str]:
    """
    Parse benchmark JSON to extract request data.

    Returns:
        Dict with input_lens, output_lens, request_timestamps, ttfts, decode_times arrays.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "request_json_invalid_payload"

        input_lens = data.get("input_lens", [])
        output_lens = data.get("output_lens", [])
        ttfts = data.get("ttfts", [])
        itls = data.get("itls", [])
        request_timestamps = data.get("request_timestamps", [])

        if not all(
            isinstance(x, list) for x in [input_lens, output_lens, ttfts, itls]
        ):
            return None, "missing_required_request_arrays"

        n = min(len(input_lens), len(output_lens), len(ttfts), len(itls))
        if n == 0:
            return None, "empty_request_arrays"

        has_timestamps = isinstance(request_timestamps, list) and len(request_timestamps) >= n
        if bool(require_recorded_timestamps) and not has_timestamps:
            return None, "missing_recorded_request_timestamps"
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
            return None, "no_valid_requests_after_filter"

        return {
            "input_lens": np.array(valid_input, dtype=np.float64),
            "output_lens": np.array(valid_output, dtype=np.float64),
            "ttfts": np.array(valid_ttft, dtype=np.float64),
            "decode_times": np.array(valid_decode, dtype=np.float64),
            "request_timestamps": np.array(valid_ts, dtype=np.float64),
            "has_timestamps": has_timestamps and np.all(np.isfinite(valid_ts)),
        }, "ok"
    except Exception:
        return None, "request_json_parse_failed"


def _compute_active_requests(
    power_timestamps: np.ndarray,
    request_timestamps: np.ndarray,
    ttfts: np.ndarray,
    decode_times: np.ndarray,
) -> np.ndarray:
    """Compute active request count at each power measurement time."""
    t_power = np.asarray(power_timestamps, dtype=np.float64).reshape(-1)
    n_power = int(t_power.size)
    active = np.zeros(n_power, dtype=np.float64)
    if n_power == 0:
        return active

    req_t = np.asarray(request_timestamps, dtype=np.float64).reshape(-1)
    ttft = np.asarray(ttfts, dtype=np.float64).reshape(-1)
    dec = np.asarray(decode_times, dtype=np.float64).reshape(-1)
    n = int(min(req_t.size, ttft.size, dec.size))
    if n <= 0:
        return active

    req_t = req_t[:n]
    ttft = ttft[:n]
    dec = dec[:n]
    valid = np.isfinite(req_t) & np.isfinite(ttft) & np.isfinite(dec)
    if not np.any(valid):
        return active

    start = req_t[valid]
    end = start + ttft[valid] + dec[valid]
    start_sorted = np.sort(start.astype(np.float64))
    end_sorted = np.sort(end.astype(np.float64))

    # active(t) = #{start <= t} - #{end < t}
    started = np.searchsorted(start_sorted, t_power, side="right")
    ended = np.searchsorted(end_sorted, t_power, side="left")
    out = started - ended
    out = np.maximum(out, 0)
    return out.astype(np.float64)


def _compute_t_arrive_log(
    power_timestamps: np.ndarray,
    request_timestamps: np.ndarray,
) -> np.ndarray:
    """Compute log(1 + inter-arrival time) for new arrivals at each power measurement."""
    t_power = np.asarray(power_timestamps, dtype=np.float64).reshape(-1)
    n_power = int(t_power.size)
    out = np.zeros(n_power, dtype=np.float64)
    if n_power < 2:
        return out

    req_t = np.asarray(request_timestamps, dtype=np.float64).reshape(-1)
    req_t = req_t[np.isfinite(req_t)]
    if req_t.size == 0:
        return out
    arr = np.sort(req_t.astype(np.float64))

    dt = float(np.median(np.diff(t_power)))
    starts = np.empty(n_power, dtype=np.float64)
    ends = np.empty(n_power, dtype=np.float64)
    starts[0] = t_power[0] - (dt / 2.0)
    starts[1:] = (t_power[1:] + t_power[:-1]) / 2.0
    ends[:-1] = (t_power[:-1] + t_power[1:]) / 2.0
    ends[-1] = t_power[-1] + (dt / 2.0)

    idx_lo = np.searchsorted(arr, starts, side="left")
    idx_hi = np.searchsorted(arr, ends, side="left")
    has_arrival = idx_hi > idx_lo
    if not np.any(has_arrival):
        return out

    first_idx = idx_lo[has_arrival]
    valid_prev = first_idx > 0
    if not np.any(valid_prev):
        return out

    selected_positions = np.nonzero(has_arrival)[0][valid_prev]
    first_with_prev = first_idx[valid_prev]
    inter = arr[first_with_prev] - arr[first_with_prev - 1]
    out[selected_positions] = np.log1p(np.maximum(0.0, inter))
    return out


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


def _load_pair_manifest_csv(
    csv_path: str,
    *,
    allowed_pair_map: Dict[str, str],
) -> List[Dict[str, str]]:
    """Load matched pair rows filtered to policy-approved pair_keys."""
    rows: List[Dict[str, str]] = []
    base_dir = str(Path(csv_path).resolve().parent)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "").strip() != "matched":
                continue
            pair_key = str(row.get("pair_key", "")).strip()
            if pair_key == "":
                continue
            resolved_json = allowed_pair_map.get(pair_key)
            if resolved_json is None:
                continue
            row_copy = dict(row)
            row_copy["json_path"] = str(resolved_json)
            power_csv_raw = str(row_copy.get("power_csv_path", "")).strip()
            resolved_power_csv = _resolve_existing_path(power_csv_raw, base_dir)
            if resolved_power_csv is not None:
                row_copy["power_csv_path"] = str(resolved_power_csv)
            rows.append(row_copy)
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
    out_dir: str = "results/experimental_continuous_v1_gru_all",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    min_traces_per_config: int = 2,
    request_timestamp_policy: str = DEFAULT_REQUEST_TIMESTAMP_POLICY,
    allowed_json_prefix: str = DEFAULT_ALLOWED_JSON_PREFIX,
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

    request_timestamp_policy = normalize_request_timestamp_policy(
        request_timestamp_policy
    )
    require_recorded_timestamps = bool(
        request_timestamp_policy_requires_recorded(request_timestamp_policy)
    )
    pair_policy_result = load_pair_manifest_map_with_policy(
        pair_manifest_csv,
        request_timestamp_policy=request_timestamp_policy,
        allowed_json_prefix=allowed_json_prefix,
        resolve_existing_path_fn=_resolve_existing_path,
        include_rejected_rows=True,
    )

    pairs = _load_pair_manifest_csv(
        pair_manifest_csv,
        allowed_pair_map=dict(pair_policy_result.pair_map),
    )
    grouped = _group_pairs_by_config(pairs)

    manifest_configs: Dict[str, Dict[str, object]] = {}
    processing_summary: Dict[str, Dict[str, object]] = {}

    for config_id, config_pairs in sorted(grouped.items()):
        traces: List[Dict[str, object]] = []
        pair_keys: List[str] = []
        rates: List[str] = []
        skipped = 0
        skipped_by_reason: Dict[str, int] = defaultdict(int)
        errors: List[str] = []

        for pair in config_pairs:
            power_csv = pair.get("power_csv_path", "")
            json_path = pair.get("json_path", "")
            pair_key = pair.get("pair_key", "")
            rate = pair.get("rate", "")
            tp = int(pair.get("tensor_parallelism", 1))

            if not (
                power_csv
                and json_path
                and os.path.exists(power_csv)
                and os.path.exists(json_path)
            ):
                skipped += 1
                skipped_by_reason["missing_source_files"] += 1
                continue

            power_data = _parse_power_csv(power_csv, tensor_parallelism=tp)
            if power_data is None:
                errors.append(f"power_parse_failed:{pair_key}")
                skipped += 1
                skipped_by_reason["power_parse_failed"] += 1
                continue

            request_data, request_reason = _parse_request_json(
                json_path,
                require_recorded_timestamps=require_recorded_timestamps,
            )
            if request_data is None:
                errors.append(f"{request_reason}:{pair_key}")
                skipped += 1
                skipped_by_reason[str(request_reason)] += 1
                continue

            aligned = _align_trace_to_grid(power_data, request_data)
            if aligned is None:
                errors.append(f"alignment_failed:{pair_key}")
                skipped += 1
                skipped_by_reason["alignment_failed"] += 1
                continue

            traces.append(aligned)
            pair_keys.append(pair_key)
            rates.append(rate)

        processing_summary[config_id] = {
            "num_pairs": len(config_pairs),
            "num_traces": len(traces),
            "skipped": skipped,
            "skipped_by_reason": {
                str(k): int(v) for k, v in sorted(skipped_by_reason.items())
            },
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

    global_skipped_by_reason: Dict[str, int] = defaultdict(int)
    for cfg_summary in processing_summary.values():
        by_reason = cfg_summary.get("skipped_by_reason", {})
        if not isinstance(by_reason, dict):
            continue
        for reason, count in by_reason.items():
            try:
                global_skipped_by_reason[str(reason)] += int(count)
            except Exception:
                continue

    manifest = {
        "schema_version": "experimental-continuous-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "pair_manifest_csv": pair_manifest_csv,
            "request_timestamp_policy": request_timestamp_policy,
            "allowed_json_prefix": allowed_json_prefix,
            "pair_manifest_policy_summary": dict(pair_policy_result.summary),
            "pair_manifest_rejected_rows_captured": int(
                len(pair_policy_result.rejected_rows)
            ),
        },
        "defaults": {
            "out_dir": out_dir,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "seed": seed,
            "min_traces_per_config": min_traces_per_config,
            "require_recorded_timestamps": bool(require_recorded_timestamps),
        },
        "summary": {
            "num_configs_total": len(grouped),
            "num_configs_written": sum(
                1 for c in manifest_configs.values() if c.get("written", False)
            ),
            "num_configs_skipped": sum(
                1 for c in manifest_configs.values() if not c.get("written", False)
            ),
            "num_pair_keys_kept": int(len(pair_policy_result.pair_map)),
            "num_pair_rows_rejected": int(
                pair_policy_result.summary.get("num_rows_rejected", 0)
            ),
            "pair_manifest_rejection_counts": dict(
                pair_policy_result.summary.get("rejection_counts", {})
            ),
            "skipped_trace_reason_counts": {
                str(k): int(v) for k, v in sorted(global_skipped_by_reason.items())
            },
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
        default="results/experimental_continuous_v1_gru_all",
        help="Output directory for experimental manifest and data",
    )
    parser.add_argument(
        "--request-timestamp-policy",
        default=DEFAULT_REQUEST_TIMESTAMP_POLICY,
        choices=list(REQUEST_TIMESTAMP_POLICIES),
        help=(
            "Request timestamp policy for pair-manifest JSON: "
            "'recorded_only' (default) or 'allow_synthesized'."
        ),
    )
    parser.add_argument(
        "--allowed-json-prefix",
        default=DEFAULT_ALLOWED_JSON_PREFIX,
        help="Only include pair-manifest JSON rooted at this prefix.",
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
        request_timestamp_policy=args.request_timestamp_policy,
        allowed_json_prefix=args.allowed_json_prefix,
    )

    print("[prepare_experimental_manifest] Summary:")
    for k, v in manifest.get("summary", {}).items():
        print(f"  {k}: {v}")
    print(f"  manifest: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
