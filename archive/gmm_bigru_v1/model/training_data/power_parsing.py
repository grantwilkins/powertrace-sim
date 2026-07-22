from __future__ import annotations

import csv
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from model.utils.decode_time import derive_decode_time
from model.utils.io import power_timestamp_to_epoch as _power_timestamp_to_epoch


MAX_BUNDLE_SAMPLE_SKEW_S = 0.05


def _clean_float(cell: str) -> float:
    """Strip units/junk from an nvidia-smi CSV cell; NaN when unparseable."""
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(cell))
    try:
        return float(match.group(0)) if match else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _field_name(header: str) -> str:
    """Canonical nvidia-smi field name without display units."""
    return re.sub(r"\s*\[[^]]+\]\s*$", "", header.strip().lower())


def _read_power_rows(csv_path: str, *, local_utc_offset_s: float = 0.0):
    """Read every power column while identifying timestamp and power fields."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        fields = [_field_name(h) for h in header]
        ts_col = next((i for i, h in enumerate(fields) if "time" in h), None)
        power_col = next(
            (i for i, h in enumerate(fields) if "power" in h and "draw" in h), None
        )
        if ts_col is None or power_col is None or len(set(fields)) != len(fields):
            return None
        rows = []
        for row in reader:
            if len(row) <= max(ts_col, power_col):
                continue
            ts = _power_timestamp_to_epoch(row[ts_col])
            if ts is None:
                continue
            values = {
                name: (row[i].strip() if i < len(row) else "")
                for i, name in enumerate(fields)
            }
            rows.append((float(ts) - float(local_utc_offset_s), values))
    return {"fields": fields, "timestamp_field": fields[ts_col],
            "power_field": fields[power_col], "rows": rows}


def _numeric_device_column(field: str, values) -> np.ndarray:
    if field == "uuid":
        return np.asarray(values, dtype=object)
    out = np.asarray([_clean_float(v) for v in values], dtype=np.float64)
    return out if np.any(np.isfinite(out)) else np.asarray(values, dtype=object)


def _strict_gpu_index(value: str) -> int:
    index = _clean_float(value)
    if not np.isfinite(index) or index < 0 or not float(index).is_integer():
        raise ValueError(f"power.csv has an invalid GPU index {value!r}")
    return int(index)


def _group_identified_rows(rows, identity: str):
    grouped = []
    for ts, row in sorted(rows, key=lambda item: item[0]):
        if not grouped or ts - grouped[-1]["start"] > MAX_BUNDLE_SAMPLE_SKEW_S:
            grouped.append({"start": ts, "rows": {}})
        sample = grouped[-1]["rows"]
        device_id = row[identity]
        if not device_id:
            raise ValueError(f"power.csv has an empty {identity}")
        if device_id in sample:
            raise ValueError(
                f"power.csv repeats device {device_id!r} within one sample"
            )
        sample[device_id] = row
    return grouped


def _identified_per_gpu(parsed, expected: int, strict: bool):
    fields = parsed["fields"]
    if strict and not {"index", "uuid"}.issubset(fields):
        raise ValueError("Bundle power.csv requires both GPU index and UUID columns")
    identity = "uuid" if "uuid" in fields else "index" if "index" in fields else None
    if identity is None:
        return None

    grouped = _group_identified_rows(parsed["rows"], identity)
    indices = {}
    uuids = {}
    for sample in grouped:
        for device_id, row in sample["rows"].items():
            if "index" not in fields:
                continue
            index = (
                _strict_gpu_index(row["index"])
                if strict
                else _clean_float(row["index"])
            )
            previous_index = indices.setdefault(device_id, index)
            if strict and previous_index != index:
                raise ValueError(
                    f"power.csv maps GPU UUID {device_id!r} to multiple indices"
                )
            if strict:
                uuid = row["uuid"]
                previous_uuid = uuids.setdefault(index, uuid)
                if previous_uuid != uuid:
                    raise ValueError(
                        f"power.csv maps GPU index {index} to multiple UUIDs"
                    )
    device_set = set().union(*(sample["rows"].keys() for sample in grouped))
    if strict and len(device_set) != expected:
        raise ValueError(
            f"Bundle manifest declares {expected} GPUs but power.csv contains "
            f"{len(device_set)} identified devices"
        )
    if len(grouped) < 2:
        return None
    incomplete = [
        sample["start"] for sample in grouped
        if set(sample["rows"]) != device_set
    ]
    if incomplete:
        raise ValueError(
            f"power.csv has incomplete device sets in {len(incomplete)} samples"
        )
    device_ids = tuple(sorted(
        device_set,
        key=lambda d: (
            not np.isfinite(indices.get(d, np.nan)), indices.get(d, np.inf), d,
        ),
    ))
    timestamps = np.asarray(
        [sample["start"] for sample in grouped], dtype=np.float64
    )
    table = {}
    for field in fields:
        if field == parsed["timestamp_field"]:
            continue
        values = [
            sample["rows"][device][field]
            for sample in grouped
            for device in device_ids
        ]
        table[field] = _numeric_device_column(field, values).reshape(
            len(grouped), len(device_ids)
        )
    return _per_gpu_result(timestamps, table, device_ids, parsed["power_field"])


def _per_gpu_result(timestamps, table, device_ids, power_field):
    shape = (len(timestamps), len(device_ids))
    nan = np.full(shape, np.nan, dtype=np.float64)
    return {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "device_ids": tuple(device_ids),
        "device_table": table,
        "power_per_gpu": np.asarray(table[power_field], dtype=np.float64),
        "util_per_gpu": np.asarray(table.get("utilization.gpu", nan), dtype=np.float64),
        "mem_per_gpu": np.asarray(table.get("memory.used", nan), dtype=np.float64),
    }


def _use_fixed_groups(raw_rows: List[Tuple[float, ...]], gpus_per_node: int) -> bool:
    """Detect the raw nvidia-smi stream layout (N contiguous GPU rows per sample)."""
    if len(raw_rows) < (2 * gpus_per_node):
        return False
    ts_arr = np.asarray([r[0] for r in raw_rows], dtype=np.float64)
    diffs = np.diff(ts_arr)
    zero_frac = float(np.mean(np.isclose(diffs, 0.0))) if diffs.size > 0 else 0.0
    pos_diffs = diffs[diffs > 0.0]
    p10_pos = (
        float(np.percentile(pos_diffs, 10)) if pos_diffs.size > 0 else float("inf")
    )
    # Raw per-GPU logs typically have many repeated timestamps and/or
    # sub-sample spacing between adjacent rows.
    return bool((zero_frac > 0.02) or (p10_pos < 0.05))


def parse_power_csv_per_gpu(
    csv_path: str,
    gpus_per_node: int = 8,
    *,
    strict_topology: bool = False,
    local_utc_offset_s: float = 0.0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse a raw per-GPU power CSV without collapsing devices.

    Returns None unless the file is the raw nvidia-smi stream (fixed blocks of
    ``gpus_per_node`` rows per sample) — already-aggregated traces have no
    per-device information to preserve.

    Returns:
        Dict with 'timestamps' (n,), 'power_per_gpu' (n, gpus_per_node),
        'util_per_gpu' and 'mem_per_gpu' (same shape, NaN where the column is
        absent). Block timestamp is the min over the block, as the legacy
        parser records.
    """
    gpn = int(max(1, gpus_per_node))
    parsed = _read_power_rows(csv_path, local_utc_offset_s=local_utc_offset_s)
    if parsed is None or len(parsed["rows"]) < 2:
        return None
    identified = _identified_per_gpu(parsed, gpn, strict_topology)
    if identified is not None:
        return identified
    rows4 = _legacy_rows(parsed)
    if strict_topology:
        raise ValueError("Bundle power.csv requires GPU index or UUID columns")
    if not _use_fixed_groups(rows4, gpn):
        return None
    return _per_gpu_from_rows(rows4, gpn)


def _legacy_rows(parsed):
    util = "utilization.gpu"
    mem = "memory.used"
    return [
        (ts, _clean_float(row[parsed["power_field"]]),
         _clean_float(row.get(util, "")), _clean_float(row.get(mem, "")))
        for ts, row in parsed["rows"]
    ]


def _per_gpu_from_rows(
    raw_rows: List[Tuple[float, float, float, float]], gpn: int
) -> Optional[Dict[str, np.ndarray]]:
    n_blocks = len(raw_rows) // gpn
    if n_blocks < 1:
        return None
    timestamps = np.empty((n_blocks,), dtype=np.float64)
    power = np.empty((n_blocks, gpn), dtype=np.float64)
    util = np.empty((n_blocks, gpn), dtype=np.float64)
    mem = np.empty((n_blocks, gpn), dtype=np.float64)
    for b in range(n_blocks):
        block = raw_rows[b * gpn : (b + 1) * gpn]
        timestamps[b] = min(r[0] for r in block)
        for g, (_, p, u, m) in enumerate(block):
            power[b, g] = p
            util[b, g] = u
            mem[b, g] = m
    return {
        "timestamps": timestamps,
        "device_ids": tuple(str(g) for g in range(gpn)),
        "device_table": {
            "power.draw": power,
            "utilization.gpu": util,
            "memory.used": mem,
        },
        "power_per_gpu": power,
        "util_per_gpu": util,
        "mem_per_gpu": mem,
    }


def tp_sum_power(power_per_gpu: np.ndarray, tensor_parallelism: int) -> np.ndarray:
    """Node-level TP-group power: nansum over the first TP devices per sample.

    This is an accessor over the per-GPU table, bit-identical to the legacy
    collapsed parse (NaN cells count as 0; an all-NaN group sums to 0.0).
    """
    arr = np.asarray(power_per_gpu, dtype=np.float64)
    tp = int(max(1, tensor_parallelism))
    tp = int(min(tp, arr.shape[1]))
    return np.nansum(arr[:, :tp], axis=1)


def parse_power_csv(
    csv_path: str,
    tensor_parallelism: int,
    gpus_per_node: int = 8,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Parse power CSV and aggregate across GPUs.

    Returns:
        Dict with 'timestamps' (epoch seconds) and 'power' (watts) arrays.
    """
    tp = int(max(1, tensor_parallelism))
    gpn = int(max(1, gpus_per_node))
    tp = int(min(tp, gpn))
    parsed = _read_power_rows(csv_path)
    if parsed is None:
        return None
    rows4 = _legacy_rows(parsed)
    raw_rows: List[Tuple[float, float]] = [(ts, p) for ts, p, _, _ in rows4]

    if len(raw_rows) < 2:
        return None

    timestamps: List[float] = []
    power_values: List[float] = []

    identified = _identified_per_gpu(parsed, gpn, False)
    if identified is not None:
        timestamps = identified["timestamps"].tolist()
        power_values = tp_sum_power(identified["power_per_gpu"], tp).tolist()
    elif _use_fixed_groups(raw_rows, gpn):
            # Preferred path: raw nvidia-smi stream (gpn GPU rows per sample).
            per_gpu = _per_gpu_from_rows(rows4, gpn)
            if per_gpu is None:
                return None
            timestamps = [float(t) for t in per_gpu["timestamps"]]
            power_values = [
                float(x) for x in tp_sum_power(per_gpu["power_per_gpu"], tp)
            ]
    else:
            # Fallback for already-aggregated traces or synthetic fixtures.
            # Keep legacy timestamp-group behavior, but if no group reaches TP
            # treat each row as one aggregated sample.
            current_ts: Optional[float] = None
            current_power: List[float] = []
            has_tp_sized_groups = False
            for ts, p in raw_rows:
                if current_ts is None:
                    current_ts = ts
                    current_power = [p]
                elif ts == current_ts:
                    current_power.append(p)
                else:
                    if len(current_power) >= tp:
                        has_tp_sized_groups = True
                        timestamps.append(float(current_ts))
                        power_values.append(float(sum(current_power[:tp])))
                    current_ts = ts
                    current_power = [p]
            if current_ts is not None and len(current_power) >= tp:
                has_tp_sized_groups = True
                timestamps.append(float(current_ts))
                power_values.append(float(sum(current_power[:tp])))

            if (not has_tp_sized_groups) and (len(timestamps) == 0):
                timestamps = [float(ts) for ts, _ in raw_rows]
                power_values = [float(p) for _, p in raw_rows]

    if len(timestamps) < 2:
        return None

    return {
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "power": np.asarray(power_values, dtype=np.float64),
    }


def extract_request_rows(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Shared row-alignment core for the request-JSON readers (data-path D6).

    Fetches the four per-request arrays, aligns them to their common length,
    and derives per-row decode times. Performs NO validity filtering: each
    caller applies its own explicit rules to the aligned rows so drop counters
    and precedence stay exactly per-caller.

    Returns None when any of the four base arrays is not a list.
    """
    input_lens = payload.get("input_lens")
    output_lens = payload.get("output_lens")
    ttfts = payload.get("ttfts")
    itls = payload.get("itls")
    request_timestamps = payload.get("request_timestamps")

    if not all(isinstance(x, list) for x in (input_lens, output_lens, ttfts, itls)):
        return None

    source_lengths = {
        "input_lens": len(input_lens),
        "output_lens": len(output_lens),
        "ttfts": len(ttfts),
        "itls": len(itls),
        "request_timestamps": (
            len(request_timestamps) if isinstance(request_timestamps, list) else None
        ),
    }
    request_column_lengths = {
        str(key): len(value)
        for key, value in payload.items()
        if isinstance(value, list)
    }
    n_base = int(min(source_lengths[key] for key in ("input_lens", "output_lens", "ttfts", "itls")))
    has_timestamps_array = isinstance(request_timestamps, list)
    decode_times = [
        derive_decode_time(itls[i], output_lens[i])[0] for i in range(n_base)
    ]
    return {
        "input_lens": input_lens,
        "output_lens": output_lens,
        "ttfts": ttfts,
        "itls": itls,
        "request_timestamps": request_timestamps if has_timestamps_array else None,
        "decode_times": decode_times,
        "n_base": n_base,
        "source_lengths": source_lengths,
        "request_column_lengths": request_column_lengths,
        "has_timestamps_array": has_timestamps_array,
    }


def parse_request_json(
    json_path: str,
    *,
    require_request_timestamps: bool = True,
) -> Optional[Dict[str, object]]:
    """
    Parse benchmark JSON to extract request data.

    Filter rules (GRU-stream policy): token counts must be finite and
    nonnegative, TTFT must be finite and positive, and multi-token completions
    require positive finite decode time. Zero/one-token completions are valid
    prefill-only rows with zero decode time. Required timestamps must be finite
    and positive.

    Returns:
        Dict with input_lens, output_lens, request_timestamps, ttfts, decode_times arrays.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    rows = extract_request_rows(data)
    if rows is None or rows["n_base"] == 0:
        return None
    input_lens = rows["input_lens"]
    output_lens = rows["output_lens"]
    ttfts = rows["ttfts"]
    request_timestamps = rows["request_timestamps"] or []
    has_timestamps = len(request_timestamps) > 0
    if require_request_timestamps and not has_timestamps:
        return None
    n = min(rows["n_base"], len(request_timestamps)) if has_timestamps else rows["n_base"]
    if n == 0:
        return None

    columns = {"input_lens": [], "output_lens": [], "ttfts": [],
               "decode_times": [], "request_timestamps": []}
    projected_itls = []
    keep = []
    stats = {
        "num_requests_raw": int(max(
            value for value in rows["source_lengths"].values() if value is not None
        )),
        "num_requests_aligned_base": int(rows["n_base"]),
        "num_requests_aligned": int(n),
        "num_requests_dropped_unaligned": int(max(
            value for value in rows["source_lengths"].values() if value is not None
        ) - n),
        "source_lengths": dict(rows["source_lengths"]),
        "request_column_lengths": dict(rows["request_column_lengths"]),
        "num_requests_used": 0,
        "num_requests_dropped_invalid_fields": 0,
        "num_requests_dropped_decode_time": 0,
        "num_requests_dropped_timestamp": 0,
    }
    for i in range(n):
        try:
            n_in = float(input_lens[i])
            n_out = float(output_lens[i])
            ttft = float(ttfts[i])
        except (TypeError, ValueError):
            stats["num_requests_dropped_invalid_fields"] += 1
            continue
        if not (np.isfinite(n_in) and np.isfinite(n_out) and np.isfinite(ttft)) \
                or n_in < 0 or n_out < 0 or ttft <= 0:
            stats["num_requests_dropped_invalid_fields"] += 1
            continue
        decode = rows["decode_times"][i]
        if n_out <= 1:
            decode = 0.0
        elif decode is None or not np.isfinite(float(decode)) or float(decode) <= 0:
            stats["num_requests_dropped_decode_time"] += 1
            continue
        ts = float("nan")
        if has_timestamps:
            try:
                ts = float(request_timestamps[i])
            except (TypeError, ValueError):
                pass
            if not np.isfinite(ts) or ts <= 0:
                stats["num_requests_dropped_timestamp"] += 1
                continue
        keep.append(i)
        columns["input_lens"].append(n_in)
        columns["output_lens"].append(n_out)
        columns["ttfts"].append(ttft)
        columns["decode_times"].append(float(decode))
        columns["request_timestamps"].append(ts)
        projected_itls.append(rows["itls"][i])

    if not keep:
        return None
    stats["num_requests_used"] = len(keep)
    request_table = {
        key: np.asarray(value, dtype=object)
        for key, value in data.items()
        if isinstance(value, list)
    }
    return {
        **{key: np.asarray(value, dtype=np.float64) for key, value in columns.items()},
        "itls": _object_vector(projected_itls),
        "has_timestamps": has_timestamps,
        "request_table": request_table,
        "projection_indices": np.asarray(keep, dtype=np.int64),
        "stats": stats,
    }


def _object_vector(values: list[object]) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out

