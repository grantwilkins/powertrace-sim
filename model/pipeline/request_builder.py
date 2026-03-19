from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from model.utils.io import (
    load_json as _load_json,
    resolve_existing_path as _resolve_existing_path,
)
from model.utils.numeric import finite_float


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
) -> List[Dict[str, float]]:
    payload = _load_json(request_json_path)
    required = ("input_lens", "output_lens")
    missing = [k for k in required if not isinstance(payload.get(k), list)]
    if missing:
        raise ValueError(f"request json missing arrays: {missing}")

    input_lens = payload["input_lens"]
    output_lens = payload["output_lens"]
    n_base = int(min(len(input_lens), len(output_lens)))
    request_timestamps_raw = payload.get("request_timestamps")
    if isinstance(request_timestamps_raw, list):
        n = int(min(n_base, len(request_timestamps_raw)))
        request_timestamps = request_timestamps_raw[:n]
    else:
        n = int(n_base)
        synth = _synthesize_request_timestamps(payload, n)
        if synth is None:
            raise ValueError("request json missing arrays: ['request_timestamps']")
        request_timestamps = synth
    if n <= 0:
        raise ValueError("request arrays are empty after alignment")

    arrivals = np.asarray(request_timestamps[:n], dtype=np.float64) - float(power_start_epoch_s)
    if arrivals.size > 0 and (
        float(np.min(arrivals)) < -float(dt) or float(np.max(arrivals)) > float(trace_duration_s) + float(dt)
    ):
        arrivals = arrivals - float(np.min(arrivals))
    arrivals = arrivals + float(alignment_offset_s)

    requests: List[Dict[str, float]] = []
    for i in range(n):
        a = float(arrivals[i])
        nin = float(input_lens[i])
        nout = float(output_lens[i])
        if not (np.isfinite(a) and np.isfinite(nin) and np.isfinite(nout)):
            continue
        requests.append(
            {
                "arrival_time": float(a),
                "input_tokens": float(max(0.0, nin)),
                "output_tokens": float(max(0.0, nout)),
            }
        )
    if len(requests) == 0:
        raise ValueError("no valid requests after filtering")
    return requests
