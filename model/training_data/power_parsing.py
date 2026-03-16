from __future__ import annotations

import csv
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from model.utils.decode_time import derive_decode_time
from model.utils.io import power_timestamp_to_epoch as _power_timestamp_to_epoch


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
    try:
        raw_rows: List[Tuple[float, float]] = []
        tp = int(max(1, tensor_parallelism))
        gpn = int(max(1, gpus_per_node))
        tp = int(min(tp, gpn))

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
                    power = float("nan")
                raw_rows.append((float(ts), float(power)))

        if len(raw_rows) < 2:
            return None

        # Preferred path: raw nvidia-smi stream (8 GPU rows per sample).
        # Aggregate by fixed contiguous groups and sum the first TP rows.
        use_fixed_groups = False
        if len(raw_rows) >= (2 * gpn):
            ts_arr = np.asarray([r[0] for r in raw_rows], dtype=np.float64)
            diffs = np.diff(ts_arr)
            zero_frac = float(np.mean(np.isclose(diffs, 0.0))) if diffs.size > 0 else 0.0
            pos_diffs = diffs[diffs > 0.0]
            p10_pos = (
                float(np.percentile(pos_diffs, 10))
                if pos_diffs.size > 0
                else float("inf")
            )
            # Raw per-GPU logs typically have many repeated timestamps and/or
            # sub-sample spacing between adjacent rows.
            use_fixed_groups = bool((zero_frac > 0.02) or (p10_pos < 0.05))

        timestamps: List[float] = []
        power_values: List[float] = []

        if use_fixed_groups:
            for start in range(0, len(raw_rows), gpn):
                block = raw_rows[start : start + gpn]
                if len(block) < gpn:
                    break
                timestamps.append(float(min(ts for ts, _ in block)))
                block_power = np.asarray([p for _, p in block[:tp]], dtype=np.float64)
                power_values.append(float(np.nansum(block_power)))
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
    except Exception:
        return None


def parse_request_json(
    json_path: str,
    *,
    require_request_timestamps: bool = True,
) -> Optional[Dict[str, object]]:
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

        has_timestamps = isinstance(request_timestamps, list) and len(request_timestamps) > 0
        if bool(require_request_timestamps) and (not has_timestamps):
            return None
        if has_timestamps:
            n = min(n, len(request_timestamps))
            if n == 0:
                return None

        valid_input: List[float] = []
        valid_output: List[float] = []
        valid_ttft: List[float] = []
        valid_decode: List[float] = []
        valid_ts: List[float] = []

        for i in range(n):
            in_tok = input_lens[i]
            out_tok = output_lens[i]
            ttft = ttfts[i]
            decode_time, _ = derive_decode_time(itls[i], out_tok)

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

        has_finite_timestamps = bool(
            has_timestamps and len(valid_ts) > 0 and np.all(np.isfinite(valid_ts))
        )
        if bool(require_request_timestamps) and (not has_finite_timestamps):
            return None

        return {
            "input_lens": np.array(valid_input, dtype=np.float64),
            "output_lens": np.array(valid_output, dtype=np.float64),
            "ttfts": np.array(valid_ttft, dtype=np.float64),
            "decode_times": np.array(valid_decode, dtype=np.float64),
            "request_timestamps": np.array(valid_ts, dtype=np.float64),
            "has_timestamps": has_finite_timestamps,
        }
    except Exception:
        return None
