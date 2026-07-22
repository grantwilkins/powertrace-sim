from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from model.training_data.alignment import align_arrivals
from model.training_data.power_parsing import parse_request_json
from model.utils.io import (
    finite_float,
    load_json as _load_json,
    resolve_input_path as _resolve_input_path,
    resolve_existing_path as _resolve_existing_path,
)


def _validated_schedule_rows(rows: object) -> List[Dict[str, float]]:
    if not isinstance(rows, list):
        raise ValueError("requests JSON must be a list or object with key 'requests'.")
    out: List[Dict[str, float]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"request[{i}] must be an object")
        missing = [
            key
            for key in ("arrival_time", "input_tokens", "output_tokens")
            if key not in row
        ]
        if missing:
            raise ValueError(f"request[{i}] missing fields: {missing}")
        values = []
        for key in ("arrival_time", "input_tokens", "output_tokens"):
            value = row[key]
            if isinstance(value, bool):
                raise ValueError(f"request[{i}].{key} must be numeric")
            try:
                values.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"request[{i}].{key} must be numeric") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"request[{i}] contains non-finite values")
        if any(value < 0.0 for value in values):
            raise ValueError(f"request[{i}] fields must be non-negative")
        out.append(
            {
                "arrival_time": values[0],
                "input_tokens": values[1],
                "output_tokens": values[2],
            }
        )
    if not out:
        raise ValueError("request schedule is empty")
    return out


def load_request_schedule(path: str) -> List[Dict[str, float]]:
    """Load the strict standalone inference request contract."""
    resolved = _resolve_input_path(path)
    with open(resolved, "r") as f:
        payload = json.load(f)
    rows = (
        payload
        if isinstance(payload, list)
        else payload.get("requests") if isinstance(payload, dict) else None
    )
    return _validated_schedule_rows(rows)


def _synthesize_request_timestamps(payload: Dict[str, object], n: int) -> Optional[List[float]]:
    if n <= 0:
        return []

    duration = finite_float(payload.get("duration"))
    if duration is not None and duration > 0:
        step = float(duration) / float(max(n, 1))
        if step > 0:
            values = (np.arange(n, dtype=np.float64) + 0.5) * step + 1.0
            return [float(x) for x in values]

    request_rate = finite_float(payload.get("request_rate"))
    poisson_rate = finite_float(payload.get("poisson_rate"))
    rate = request_rate if request_rate is not None else poisson_rate
    if rate is not None and rate > 0:
        step = 1.0 / float(rate)
        values = (np.arange(n, dtype=np.float64) + 1.0) * step + 1.0
        return [float(x) for x in values]
    return None


def _load_pair_manifest_map(pair_manifest_csv: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    base_dir = str(Path(pair_manifest_csv).resolve().parent)
    with open(pair_manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip() != "matched":
                continue
            key = str(row.get("pair_key", "")).strip()
            json_path_raw = str(row.get("json_path", "")).strip()
            if key == "" or json_path_raw == "":
                continue
            json_path = _resolve_existing_path(json_path_raw, base_dir)
            if json_path is not None:
                out[key] = json_path
    return out


def _build_requests_from_stage0_json(
    request_json_path: str,
    *,
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
    alignment_offset_s: float = 0.0,
    require_recorded_timestamps: bool = True,
) -> List[Dict[str, float]]:
    payload = _load_json(request_json_path)
    recorded = payload.get("request_timestamps")
    if require_recorded_timestamps and (
        not isinstance(recorded, list) or len(recorded) == 0
    ):
        raise ValueError("request json missing arrays: ['request_timestamps']")
    parsed = parse_request_json(
        request_json_path,
        require_request_timestamps=bool(require_recorded_timestamps),
    )
    if parsed is None:
        raise ValueError("request json does not satisfy the validated training row policy")
    input_lens = parsed["input_lens"]
    output_lens = parsed["output_lens"]
    n = int(len(input_lens))
    if parsed["has_timestamps"]:
        request_timestamps = parsed["request_timestamps"]
    else:
        request_timestamps = _synthesize_request_timestamps(payload, n)
        if request_timestamps is None:
            raise ValueError("request json missing arrays: ['request_timestamps']")

    arrivals, ok, _ = align_arrivals(
        np.asarray(request_timestamps, dtype=np.float64),
        float(power_start_epoch_s),
        policy="rebase_into_window",
        dt=float(dt),
        trace_duration_s=float(trace_duration_s),
    )
    if not ok:
        raise ValueError("request arrivals do not satisfy alignment policy")
    arrivals = arrivals + float(alignment_offset_s)
    return _validated_schedule_rows(
        [
            {
                "arrival_time": float(arrivals[i]),
                "input_tokens": float(input_lens[i]),
                "output_tokens": float(output_lens[i]),
            }
            for i in range(n)
        ]
    )

