from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from model.utils.io import power_timestamp_to_epoch as _power_timestamp_to_epoch


BENCH_FILE_RE = re.compile(
    r"^(?P<model>.+)_tp(?P<tp>\d+)_rate(?P<rate>[\d.]+)_iter(?P<iter>\d+)_(?P<date>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.(?P<ext>json|csv)$"
)
SHAREGPT_JSON_RE = re.compile(
    r"^vllm-(?P<rate>[\d.]+)qps-tp(?P<tp>\d+)-(?P<served_model>.+)-(?P<date>\d{8}-\d{6})\.json$"
)
SHAREGPT_CSV_RE = re.compile(
    r"^(?P<model>.+)_tp(?P<tp>\d+)_p(?P<rate>[\d.]+)_d(?P<date>\d{8}-\d{6}|\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.csv$"
)
HYPHENATED_DATE_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})-(?P<h>\d{2})-(?P<mi>\d{2})-(?P<s>\d{2})$"
)

BASE_REQUEST_FIELDS = ("input_lens", "output_lens", "ttfts", "itls")
REQUEST_TIMESTAMP_FIELD = "request_timestamps"


@dataclass(frozen=True)
class MatchedPair:
    family: str
    dataset_dir: str
    model_name: str
    hardware: str
    tensor_parallelism: int
    rate: str
    iteration: Optional[int]
    date_key: str
    pair_key: str
    power_csv_path: str
    json_path: str


def canonical_rate(value: object) -> str:
    try:
        return f"{float(value):g}"
    except Exception:
        return str(value)


def normalize_sharegpt_date(date_str: str) -> str:
    if re.fullmatch(r"\d{8}-\d{6}", date_str):
        return date_str
    match = HYPHENATED_DATE_RE.fullmatch(date_str)
    if not match:
        return date_str
    return (
        f"{match.group('y')}{match.group('m')}{match.group('d')}-"
        f"{match.group('h')}{match.group('mi')}{match.group('s')}"
    )


def parse_benchmark_filename(filename: str) -> Optional[Dict[str, object]]:
    match = BENCH_FILE_RE.fullmatch(filename)
    if not match:
        return None
    return {
        "model": match.group("model"),
        "tp": int(match.group("tp")),
        "rate": canonical_rate(match.group("rate")),
        "iteration": int(match.group("iter")),
        "date": match.group("date"),
        "ext": match.group("ext"),
    }


def parse_sharegpt_json_filename(filename: str) -> Optional[Dict[str, object]]:
    match = SHAREGPT_JSON_RE.fullmatch(filename)
    if not match:
        return None
    return {
        "tp": int(match.group("tp")),
        "rate": canonical_rate(match.group("rate")),
        "date": normalize_sharegpt_date(match.group("date")),
        "served_model": match.group("served_model"),
        "ext": "json",
    }


def parse_sharegpt_csv_filename(filename: str) -> Optional[Dict[str, object]]:
    match = SHAREGPT_CSV_RE.fullmatch(filename)
    if not match:
        return None
    return {
        "model": match.group("model"),
        "tp": int(match.group("tp")),
        "rate": canonical_rate(match.group("rate")),
        "date": normalize_sharegpt_date(match.group("date")),
        "ext": "csv",
    }


def parse_dataset_dir_metadata(dir_name: str, family: str) -> Optional[Tuple[str, str]]:
    if family == "benchmark":
        prefix = "benchmark-"
    elif family == "sharegpt-benchmark":
        prefix = "sharegpt-benchmark-"
    else:
        return None
    if not dir_name.startswith(prefix):
        return None
    body = dir_name[len(prefix) :]
    pieces = body.split("-")
    if len(pieces) < 2:
        return None
    hardware = pieces[-1].upper()
    model_name = "-".join(pieces[:-1])
    return model_name, hardware


def inspect_power_csv(csv_path: str, max_rows: int = 4096) -> Dict[str, object]:
    out: Dict[str, object] = {
        "power_header": "",
        "power_parseable": False,
        "power_parse_error": None,
        "sampled_rows": 0,
        "raw_dt_median_s": None,
        "aggregated_dt_median_s": None,
        "rows_per_sample_estimate": None,
        "inferred_sampling_class": "unknown",
    }
    raw_diffs: List[float] = []
    agg_diffs: List[float] = []
    run_lengths: List[int] = []
    parse_failures = 0
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            out["power_header"] = ", ".join(h.strip() for h in header) if header else ""

            prev_ts: Optional[float] = None
            current_group_ts: Optional[float] = None
            current_group_count = 0

            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                if not row:
                    continue
                ts = _power_timestamp_to_epoch(row[0])
                if ts is None:
                    parse_failures += 1
                    continue
                out["sampled_rows"] = int(out["sampled_rows"]) + 1
                if prev_ts is not None:
                    raw_diffs.append(ts - prev_ts)
                prev_ts = ts

                if current_group_ts is None:
                    current_group_ts = ts
                    current_group_count = 1
                elif ts == current_group_ts:
                    current_group_count += 1
                else:
                    run_lengths.append(current_group_count)
                    agg_diffs.append(ts - current_group_ts)
                    current_group_ts = ts
                    current_group_count = 1
            if current_group_count > 0:
                run_lengths.append(current_group_count)

        if int(out["sampled_rows"]) >= 2:
            out["power_parseable"] = True
            if raw_diffs:
                out["raw_dt_median_s"] = float(np.median(np.asarray(raw_diffs)))
            positive_agg = [d for d in agg_diffs if d > 0]
            if positive_agg:
                out["aggregated_dt_median_s"] = float(np.median(np.asarray(positive_agg)))
            if run_lengths:
                out["rows_per_sample_estimate"] = int(round(float(np.median(np.asarray(run_lengths)))))
            rows_per_sample = int(out["rows_per_sample_estimate"] or 1)
            if rows_per_sample >= 2:
                out["inferred_sampling_class"] = "raw_per_gpu"
            else:
                out["inferred_sampling_class"] = "already_aggregated"
        else:
            out["power_parse_error"] = "insufficient parseable timestamp rows"

        if parse_failures > 0:
            out["power_parse_failures"] = parse_failures
    except Exception as exc:
        out["power_parse_error"] = str(exc)

    return out


def _json_itls_format(itls: object) -> str:
    if not isinstance(itls, list):
        return "missing"
    non_null = [x for x in itls if x is not None]
    if not non_null:
        return "empty"
    if all(isinstance(x, list) for x in non_null):
        return "list"
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in non_null):
        return "scalar"
    return "mixed"


def inspect_json_schema(payload: Dict[str, object]) -> Dict[str, object]:
    key_presence = {k: (k in payload) for k in (*BASE_REQUEST_FIELDS, REQUEST_TIMESTAMP_FIELD, "e2els", "tpots")}
    lengths: Dict[str, Optional[int]] = {}
    for key in (*BASE_REQUEST_FIELDS, REQUEST_TIMESTAMP_FIELD):
        value = payload.get(key)
        lengths[key] = len(value) if isinstance(value, list) else None

    base_lengths = [lengths[k] for k in BASE_REQUEST_FIELDS if lengths[k] is not None]
    aligned_without_ts = int(min(base_lengths)) if base_lengths else 0
    aligned_with_ts = aligned_without_ts
    if lengths[REQUEST_TIMESTAMP_FIELD] is not None:
        aligned_with_ts = int(min(aligned_without_ts, int(lengths[REQUEST_TIMESTAMP_FIELD])))

    mismatch_fields: List[str] = []
    if base_lengths:
        for key in BASE_REQUEST_FIELDS:
            value = lengths.get(key)
            if value is not None and value != aligned_without_ts:
                mismatch_fields.append(key)
    if lengths[REQUEST_TIMESTAMP_FIELD] is not None and lengths[REQUEST_TIMESTAMP_FIELD] != aligned_without_ts:
        mismatch_fields.append(REQUEST_TIMESTAMP_FIELD)

    return {
        "json_key_presence": key_presence,
        "json_array_lengths": lengths,
        "aligned_request_count_without_timestamps": aligned_without_ts,
        "aligned_request_count": aligned_with_ts,
        "itls_format": _json_itls_format(payload.get("itls")),
        "mismatched_length_fields": sorted(mismatch_fields),
    }


def derive_decode_time(itl_value: object, output_tokens: object) -> Tuple[Optional[float], str]:
    if isinstance(itl_value, list):
        if len(itl_value) == 0:
            return None, "list"
        try:
            arr = np.asarray(itl_value, dtype=float)
        except Exception:
            return None, "list"
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None, "list"
        return float(np.sum(arr)), "list"

    if isinstance(itl_value, (int, float)) and not isinstance(itl_value, bool):
        if not np.isfinite(float(itl_value)):
            return None, "scalar"
        try:
            out_tok = int(float(output_tokens))
        except Exception:
            return None, "scalar"
        return float(itl_value) * float(max(out_tok - 1, 1)), "scalar"

    return None, "unknown"


def _finite_number(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def extract_request_metrics(payload: Dict[str, object]) -> Dict[str, object]:
    schema = inspect_json_schema(payload)

    input_lens = payload.get("input_lens")
    output_lens = payload.get("output_lens")
    ttfts = payload.get("ttfts")
    itls = payload.get("itls")
    request_ts = payload.get(REQUEST_TIMESTAMP_FIELD)

    if not all(isinstance(x, list) for x in (input_lens, output_lens, ttfts, itls)):
        return {
            "prefill_rates": [],
            "decode_rates": [],
            "timestamp_windows": [],
            "stats": {
                "num_requests_total": 0,
                "num_requests_aligned": 0,
                "num_requests_used": 0,
                "num_requests_dropped_invalid_fields": 0,
                "num_requests_dropped_decode_time": 0,
                "num_requests_missing_or_invalid_timestamp": 0,
                "num_timestamp_windows": 0,
                "mismatched_array_lengths": True,
                "missing_request_timestamps_array": True,
            },
            "schema": schema,
        }

    base_min = int(min(len(input_lens), len(output_lens), len(ttfts), len(itls)))
    has_request_ts_array = isinstance(request_ts, list)
    aligned = base_min if not has_request_ts_array else int(min(base_min, len(request_ts)))

    prefill_rates: List[float] = []
    decode_rates: List[float] = []
    timestamp_windows: List[Tuple[float, float, float, float]] = []

    dropped_invalid = 0
    dropped_decode = 0
    missing_or_bad_ts = 0

    for i in range(aligned):
        in_tok = input_lens[i]
        out_tok = output_lens[i]
        ttft = ttfts[i]
        decode_time, _ = derive_decode_time(itls[i], out_tok)
        if decode_time is None or (not np.isfinite(decode_time)) or decode_time <= 0:
            dropped_decode += 1
            continue

        if (not _finite_number(in_tok)) or (not _finite_number(out_tok)) or (not _finite_number(ttft)):
            dropped_invalid += 1
            continue
        in_tok_f = float(in_tok)
        out_tok_f = float(out_tok)
        ttft_f = float(ttft)

        if ttft_f <= 0 or in_tok_f < 0 or out_tok_f < 0:
            dropped_invalid += 1
            continue

        prefill_rate = in_tok_f / ttft_f
        decode_rate = out_tok_f / decode_time
        if (not np.isfinite(prefill_rate)) or (not np.isfinite(decode_rate)):
            dropped_invalid += 1
            continue

        prefill_rates.append(float(prefill_rate))
        decode_rates.append(float(decode_rate))

        if has_request_ts_array:
            ts = request_ts[i]
            if _finite_number(ts) and float(ts) > 0:
                prefill_end = float(ts) + ttft_f
                decode_end = prefill_end + float(decode_time)
                if decode_end > prefill_end:
                    midpoint = prefill_end + (0.5 * float(decode_time))
                    timestamp_windows.append((prefill_end, decode_end, midpoint, float(decode_rate)))
                else:
                    missing_or_bad_ts += 1
            else:
                missing_or_bad_ts += 1
        else:
            missing_or_bad_ts += 1

    stats = {
        "num_requests_total": base_min,
        "num_requests_aligned": aligned,
        "num_requests_used": len(prefill_rates),
        "num_requests_dropped_invalid_fields": dropped_invalid,
        "num_requests_dropped_decode_time": dropped_decode,
        "num_requests_missing_or_invalid_timestamp": missing_or_bad_ts,
        "num_timestamp_windows": len(timestamp_windows),
        "mismatched_array_lengths": bool(len(schema["mismatched_length_fields"]) > 0),
        "missing_request_timestamps_array": not has_request_ts_array,
    }
    return {
        "prefill_rates": prefill_rates,
        "decode_rates": decode_rates,
        "timestamp_windows": timestamp_windows,
        "stats": stats,
        "schema": schema,
    }


def concurrency_binned_decode_medians(
    timestamp_windows: Sequence[Tuple[float, float, float, float]],
    min_bin_samples: int,
) -> List[Dict[str, object]]:
    if not timestamp_windows:
        return []

    starts = np.asarray([w[0] for w in timestamp_windows], dtype=float)
    ends = np.asarray([w[1] for w in timestamp_windows], dtype=float)
    mids = np.asarray([w[2] for w in timestamp_windows], dtype=float)
    decode_rates = np.asarray([w[3] for w in timestamp_windows], dtype=float)

    starts_sorted = np.sort(starts)
    ends_sorted = np.sort(ends)

    concurrency = (
        np.searchsorted(starts_sorted, mids, side="right")
        - np.searchsorted(ends_sorted, mids, side="right")
    )
    concurrency = np.maximum(concurrency, 1)

    grouped: Dict[int, List[float]] = defaultdict(list)
    for c, r in zip(concurrency, decode_rates):
        grouped[int(c)].append(float(r))

    bins: List[Dict[str, object]] = []
    for c in sorted(grouped.keys()):
        values = grouped[c]
        if len(values) < int(min_bin_samples):
            continue
        bins.append(
            {
                "concurrency": int(c),
                "n": int(len(values)),
                "median_decode_toks_per_s": float(np.median(np.asarray(values, dtype=float))),
            }
        )
    return bins


def select_decode_model_type(
    bins: Sequence[Dict[str, object]],
    batch_variation_threshold: float,
) -> Tuple[str, Optional[float]]:
    if len(bins) < 2:
        return "constant", None
    medians = [float(b["median_decode_toks_per_s"]) for b in bins]
    min_med = min(medians)
    max_med = max(medians)
    ratio = float("inf") if min_med <= 0 else float(max_med / min_med)
    if ratio > float(batch_variation_threshold):
        return "by_concurrency", ratio
    return "constant", ratio


def discover_dataset_dirs(data_root_dir: str, include_families: Sequence[str]) -> List[Tuple[str, str]]:
    root = Path(data_root_dir)
    discovered: List[Tuple[str, str]] = []
    if not root.exists():
        return discovered

    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        for family in include_families:
            if entry.name.startswith(f"{family}-"):
                discovered.append((family, str(entry)))
                break
    return discovered


def _benchmark_key(parsed: Dict[str, object]) -> Tuple[object, ...]:
    return (
        str(parsed["model"]),
        int(parsed["tp"]),
        str(parsed["rate"]),
        int(parsed["iteration"]),
        str(parsed["date"]),
    )


def _sharegpt_key(parsed: Dict[str, object]) -> Tuple[object, ...]:
    return (
        int(parsed["tp"]),
        str(parsed["rate"]),
        str(parsed["date"]),
    )


def discover_pairs_for_dataset(
    family: str,
    dataset_dir: str,
) -> Tuple[List[Dict[str, object]], List[MatchedPair], Dict[str, object]]:
    dataset_name = Path(dataset_dir).name
    metadata = parse_dataset_dir_metadata(dataset_name, family)
    model_from_dir, hardware_from_dir = metadata if metadata else ("unknown", "UNKNOWN")

    json_files: Dict[Tuple[object, ...], str] = {}
    power_files: Dict[Tuple[object, ...], str] = {}
    dup_json = 0
    dup_power = 0
    ignored = 0

    for file_path in sorted(Path(dataset_dir).rglob("*"), key=lambda p: str(p)):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in {".json", ".csv"}:
            continue

        parsed: Optional[Dict[str, object]]
        key: Optional[Tuple[object, ...]]

        if family == "benchmark":
            parsed = parse_benchmark_filename(file_path.name)
            key = _benchmark_key(parsed) if parsed else None
        elif family == "sharegpt-benchmark":
            if suffix == ".json":
                parsed = parse_sharegpt_json_filename(file_path.name)
            else:
                parsed = parse_sharegpt_csv_filename(file_path.name)
            key = _sharegpt_key(parsed) if parsed else None
        else:
            parsed = None
            key = None

        if parsed is None or key is None:
            ignored += 1
            continue

        key_path = str(file_path)
        if suffix == ".json":
            if key in json_files:
                dup_json += 1
                continue
            json_files[key] = key_path
        else:
            if key in power_files:
                dup_power += 1
                continue
            power_files[key] = key_path

    manifest_rows: List[Dict[str, object]] = []
    matched_pairs: List[MatchedPair] = []
    all_keys = sorted(set(json_files.keys()) | set(power_files.keys()), key=lambda x: str(x))

    for key in all_keys:
        json_path = json_files.get(key)
        power_path = power_files.get(key)
        status = "matched"
        if json_path is None:
            status = "power_only"
        elif power_path is None:
            status = "json_only"

        if family == "benchmark":
            key_model, tp, rate, iteration, date_key = key
            pair_key = f"{key_model}|tp={tp}|rate={rate}|iter={iteration}|date={date_key}"
        else:
            tp, rate, date_key = key
            iteration = None
            pair_key = f"tp={tp}|rate={rate}|date={date_key}"

        manifest_row = {
            "family": family,
            "dataset_dir": dataset_dir,
            "dataset_name": dataset_name,
            "status": status,
            "model_name": model_from_dir,
            "hardware": hardware_from_dir,
            "tensor_parallelism": int(tp),
            "rate": str(rate),
            "iteration": "" if iteration is None else int(iteration),
            "date_key": str(date_key),
            "pair_key": pair_key,
            "power_csv_path": power_path or "",
            "json_path": json_path or "",
        }
        manifest_rows.append(manifest_row)

        if status == "matched":
            matched_pairs.append(
                MatchedPair(
                    family=family,
                    dataset_dir=dataset_dir,
                    model_name=model_from_dir,
                    hardware=hardware_from_dir,
                    tensor_parallelism=int(tp),
                    rate=str(rate),
                    iteration=(None if iteration is None else int(iteration)),
                    date_key=str(date_key),
                    pair_key=pair_key,
                    power_csv_path=power_path or "",
                    json_path=json_path or "",
                )
            )

    discovery_stats = {
        "family": family,
        "dataset_dir": dataset_dir,
        "dataset_name": dataset_name,
        "num_json_files": len(json_files),
        "num_power_csv_files": len(power_files),
        "num_manifest_rows": len(manifest_rows),
        "num_matched_pairs": len(matched_pairs),
        "num_json_only": int(sum(1 for r in manifest_rows if r["status"] == "json_only")),
        "num_power_only": int(sum(1 for r in manifest_rows if r["status"] == "power_only")),
        "duplicate_json_keys_dropped": dup_json,
        "duplicate_power_keys_dropped": dup_power,
        "ignored_files": ignored,
    }
    return manifest_rows, matched_pairs, discovery_stats


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
