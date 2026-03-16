from __future__ import annotations

import csv
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.utils.decode_time import derive_decode_time
from model.utils.io import power_timestamp_to_epoch as _power_timestamp_to_epoch

BASE_REQUEST_FIELDS = ("input_lens", "output_lens", "ttfts", "itls")
REQUEST_TIMESTAMP_FIELD = "request_timestamps"


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
