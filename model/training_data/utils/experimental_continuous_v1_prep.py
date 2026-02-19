from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TraceRecord:
    pair_key: str
    family: str
    rate: str
    iteration: str
    power_start_epoch_s: float
    power: np.ndarray
    active_requests: np.ndarray
    t_arrive: np.ndarray
    t_arrive_log: np.ndarray
    requests: List[Dict[str, float]]
    x_norm: Optional[np.ndarray] = None
    y_norm: Optional[np.ndarray] = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, object]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _stable_seed(config_id: str, seed: int) -> int:
    h = hashlib.md5(config_id.encode("utf-8")).hexdigest()
    return int((int(h[:8], 16) + int(seed)) % (2**32 - 1))


def _finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _power_timestamp_to_epoch(ts_text: str) -> Optional[float]:
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


def _power_value_to_float(text: str) -> Optional[float]:
    cleaned = re.sub(r"[^0-9eE+.\-]", "", text)
    return _finite_float(cleaned)


def _resample_to_fixed_dt(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    dt: float,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if timestamps_s.size == 0 or values.size == 0:
        return None, None
    if timestamps_s.size != values.size:
        return None, None

    t0 = float(np.min(timestamps_s))
    t1 = float(np.max(timestamps_s))
    if t1 < t0:
        return None, None

    n_bins = int(math.floor((t1 - t0) / dt)) + 1
    if n_bins <= 0:
        return None, None

    sums = np.zeros((n_bins,), dtype=np.float64)
    counts = np.zeros((n_bins,), dtype=np.float64)

    idx = np.floor((timestamps_s - t0) / dt).astype(np.int64)
    in_range = (idx >= 0) & (idx < n_bins)
    idx = idx[in_range]
    vals = values[in_range]
    finite = np.isfinite(vals)
    idx = idx[finite]
    vals = vals[finite]
    if idx.size == 0:
        return None, None

    np.add.at(sums, idx, vals)
    np.add.at(counts, idx, 1.0)

    out = np.full((n_bins,), np.nan, dtype=np.float64)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    return out, t0


def _interpolate_short_gaps(values: np.ndarray, max_gap_steps: int) -> Optional[np.ndarray]:
    x = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if not np.any(finite):
        return None

    first = int(np.argmax(finite))
    last = int(len(x) - 1 - np.argmax(finite[::-1]))
    x = x[first : last + 1]
    finite = np.isfinite(x)
    n = len(x)
    i = 0
    while i < n:
        if finite[i]:
            i += 1
            continue
        j = i
        while j < n and not finite[j]:
            j += 1
        gap_len = j - i
        left = i - 1
        right = j
        if gap_len <= int(max_gap_steps) and left >= 0 and right < n and finite[left] and finite[right]:
            x[i:j] = np.linspace(x[left], x[right], gap_len + 2)[1:-1]
            finite[i:j] = True
        else:
            return None
        i = j

    if not np.all(np.isfinite(x)):
        return None
    return x


def _extract_power_series(
    csv_path: str,
    tp: int,
    dt: float,
    power_group_window_s: float,
    max_interp_gap_steps: int,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
    raw_ts: List[float] = []
    raw_power: List[float] = []

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                ts = _power_timestamp_to_epoch(row[0])
                p = _power_value_to_float(row[1])
                if ts is None or p is None:
                    continue
                raw_ts.append(float(ts))
                raw_power.append(float(p))
    except Exception as exc:
        return None, None, f"power_csv_read_error:{type(exc).__name__}"

    if len(raw_ts) < max(2, tp):
        return None, None, "power_insufficient_rows"

    grouped_ts: List[float] = []
    grouped_power: List[float] = []

    cur_ts = raw_ts[0]
    cur_powers: List[float] = [raw_power[0]]
    cur_min_ts = raw_ts[0]

    for ts, p in zip(raw_ts[1:], raw_power[1:]):
        if abs(ts - cur_ts) <= float(power_group_window_s):
            cur_powers.append(p)
            cur_min_ts = min(cur_min_ts, ts)
        else:
            if len(cur_powers) >= int(tp):
                grouped_power.append(float(np.sum(cur_powers[: int(tp)])))
            else:
                grouped_power.append(float("nan"))
            grouped_ts.append(float(cur_min_ts))
            cur_ts = ts
            cur_min_ts = ts
            cur_powers = [p]

    if len(cur_powers) >= int(tp):
        grouped_power.append(float(np.sum(cur_powers[: int(tp)])))
    else:
        grouped_power.append(float("nan"))
    grouped_ts.append(float(cur_min_ts))

    ts_arr = np.asarray(grouped_ts, dtype=np.float64)
    pw_arr = np.asarray(grouped_power, dtype=np.float64)
    if ts_arr.size < 2:
        return None, None, "power_insufficient_groups"

    resampled, t0 = _resample_to_fixed_dt(ts_arr, pw_arr, dt=float(dt))
    if resampled is None or t0 is None:
        return None, None, "power_resample_failed"

    interp = _interpolate_short_gaps(resampled, max_gap_steps=int(max_interp_gap_steps))
    if interp is None:
        return None, None, "power_unresolved_gap"
    if len(interp) < 2:
        return None, None, "power_too_short"

    return interp.astype(np.float64), float(t0), None


def _derive_decode_time(itl_value: object, output_tokens: float) -> Optional[float]:
    out_tok = max(float(output_tokens), 0.0)
    if isinstance(itl_value, list):
        if len(itl_value) == 0:
            return None
        try:
            arr = np.asarray(itl_value, dtype=np.float64)
        except Exception:
            return None
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return float(np.sum(arr))

    scalar = _finite_float(itl_value)
    if scalar is None:
        return None
    return float(scalar) * float(max(int(round(out_tok)) - 1, 1))


def _extract_request_log(
    json_path: str,
) -> Tuple[Optional[List[Dict[str, float]]], Optional[str], Dict[str, int]]:
    stats = {
        "num_requests_total": 0,
        "num_requests_aligned": 0,
        "num_requests_used": 0,
        "num_requests_dropped_invalid": 0,
    }
    try:
        with open(json_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        return None, f"json_read_error:{type(exc).__name__}", stats

    if not isinstance(payload, dict):
        return None, "json_not_object", stats

    required = ["input_lens", "output_lens", "ttfts", "itls", "request_timestamps"]
    if any(not isinstance(payload.get(k), list) for k in required):
        return None, "missing_request_timestamps", stats

    input_lens = payload["input_lens"]
    output_lens = payload["output_lens"]
    ttfts = payload["ttfts"]
    itls = payload["itls"]
    request_timestamps = payload["request_timestamps"]
    n = int(min(len(input_lens), len(output_lens), len(ttfts), len(itls), len(request_timestamps)))
    stats["num_requests_total"] = int(min(len(input_lens), len(output_lens), len(ttfts), len(itls)))
    stats["num_requests_aligned"] = n
    if n <= 0:
        return None, "request_arrays_empty", stats

    records: List[Dict[str, float]] = []
    for i in range(n):
        arrival = _finite_float(request_timestamps[i])
        n_in = _finite_float(input_lens[i])
        n_out = _finite_float(output_lens[i])
        ttft = _finite_float(ttfts[i])
        if arrival is None or n_in is None or n_out is None or ttft is None:
            stats["num_requests_dropped_invalid"] += 1
            continue
        if arrival <= 0 or n_in < 0 or n_out < 0 or ttft < 0:
            stats["num_requests_dropped_invalid"] += 1
            continue
        decode_time = _derive_decode_time(itls[i], n_out)
        if decode_time is None or not np.isfinite(decode_time) or decode_time < 0:
            stats["num_requests_dropped_invalid"] += 1
            continue
        total_latency = float(ttft + decode_time)
        if total_latency <= 0:
            stats["num_requests_dropped_invalid"] += 1
            continue
        records.append(
            {
                "arrival_epoch_s": float(arrival),
                "input_tokens": float(n_in),
                "output_tokens": float(n_out),
                "ttft": float(ttft),
                "total_latency": float(total_latency),
            }
        )

    stats["num_requests_used"] = int(len(records))
    if len(records) == 0:
        return None, "no_valid_requests", stats
    return records, None, stats


def _align_request_times(
    requests: Sequence[Dict[str, float]],
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
) -> Tuple[List[Dict[str, float]], bool]:
    arr = np.asarray([float(r["arrival_epoch_s"]) - float(power_start_epoch_s) for r in requests], dtype=np.float64)
    shifted = False
    if arr.size > 0 and (np.min(arr) < -float(dt) or np.max(arr) > float(trace_duration_s) + float(dt)):
        arr = arr - float(np.min(arr))
        shifted = True

    out: List[Dict[str, float]] = []
    for i, r in enumerate(requests):
        rec = {
            "arrival_time": float(arr[i]),
            "input_tokens": int(max(round(float(r["input_tokens"])), 0)),
            "output_tokens": int(max(round(float(r["output_tokens"])), 0)),
            "ttft": float(r["ttft"]),
            "total_latency": float(r["total_latency"]),
        }
        out.append(rec)
    return out, shifted


def _compute_feature_series(
    power: np.ndarray,
    requests: Sequence[Dict[str, float]],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = int(len(power))
    active_diff = np.zeros((T + 1,), dtype=np.float64)
    t_arrive = np.zeros((T,), dtype=np.float64)

    for r in requests:
        arrival = float(r["arrival_time"])
        completion = float(r["arrival_time"] + r["total_latency"])
        in_tok = float(max(r["input_tokens"], 0))

        b = int(math.floor(arrival / dt))
        if 0 <= b < T:
            t_arrive[b] += in_tok

        start_idx = int(math.ceil(arrival / dt))
        end_idx = int(math.ceil(completion / dt))
        if end_idx <= 0 or start_idx >= T:
            continue
        l = max(0, start_idx)
        rr = min(T, end_idx)
        if rr <= l:
            continue
        active_diff[l] += 1.0
        active_diff[rr] -= 1.0

    active = np.clip(np.cumsum(active_diff[:-1]), a_min=0.0, a_max=None)
    t_arrive_log = np.log1p(np.clip(t_arrive, a_min=0.0, a_max=None))
    return active.astype(np.float64), t_arrive.astype(np.float64), t_arrive_log.astype(np.float64)


def _safe_mean_std(x: np.ndarray, default_mean: float = 0.0, default_std: float = 1.0) -> Tuple[float, float]:
    if x.size == 0:
        return float(default_mean), float(default_std)
    return float(np.mean(x)), float(np.std(x) + 1e-6)


def _fit_norm_params(train_traces: Sequence[TraceRecord]) -> Dict[str, float]:
    power = np.concatenate([np.asarray(tr.power, dtype=np.float64).reshape(-1) for tr in train_traces], axis=0)
    active = np.concatenate([np.asarray(tr.active_requests, dtype=np.float64).reshape(-1) for tr in train_traces], axis=0)
    t_arrive_log = np.concatenate([np.asarray(tr.t_arrive_log, dtype=np.float64).reshape(-1) for tr in train_traces], axis=0)

    power_mean, power_std = _safe_mean_std(power, default_mean=0.0, default_std=1.0)
    active_mean, active_std = _safe_mean_std(active, default_mean=0.0, default_std=1.0)
    t_arrive_mean, t_arrive_std = _safe_mean_std(t_arrive_log, default_mean=0.0, default_std=1.0)

    return {
        "power_mean": float(power_mean),
        "power_std": float(max(power_std, 1e-6)),
        "active_mean": float(active_mean),
        "active_std": float(max(active_std, 1e-6)),
        "t_arrive_log_mean": float(t_arrive_mean),
        "t_arrive_log_std": float(max(t_arrive_std, 1e-6)),
        "power_min": float(np.min(power)) if power.size else float("nan"),
        "power_max": float(np.max(power)) if power.size else float("nan"),
    }


def _build_xy_for_trace(trace: TraceRecord, norm: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(trace.power, dtype=np.float64)
    a = np.asarray(trace.active_requests, dtype=np.float64)
    t_log = np.asarray(trace.t_arrive_log, dtype=np.float64)
    L = int(min(len(p) - 1, len(a) - 1, len(t_log) - 1))
    if L <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    p_mean = float(norm["power_mean"])
    p_std = float(norm["power_std"])
    a_mean = float(norm["active_mean"])
    a_std = float(norm["active_std"])
    t_mean = float(norm["t_arrive_log_mean"])
    t_std = float(norm["t_arrive_log_std"])

    x = np.stack(
        [
            (p[:L] - p_mean) / p_std,
            (a[1 : L + 1] - a_mean) / a_std,
            (t_log[1 : L + 1] - t_mean) / t_std,
        ],
        axis=1,
    )
    y = (p[1 : L + 1] - p_mean) / p_std
    return x.astype(np.float32), y.astype(np.float32)


def _split_indices(
    n: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)

    if n <= 2:
        n_train = max(1, n - 1)
        n_val = 0
    else:
        n_train = max(1, int(round(n * float(train_fraction))))
        n_val = int(round(n * float(val_fraction)))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        if n_train >= n:
            n_train = n - 1
        if n_val == 0:
            n_val = 1
            if n_train + n_val >= n:
                n_train = max(1, n - 2)

    n_test = n - n_train - n_val
    if n >= 3 and n_test <= 0:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0

    train = np.sort(idx[:n_train])
    val = np.sort(idx[n_train : n_train + n_val])
    test = np.sort(idx[n_train + n_val : n_train + n_val + n_test])
    return train, val, test


def _config_id_from_row(row: Dict[str, str]) -> str:
    model = str(row.get("model_name", "")).strip()
    hw = str(row.get("hardware", "")).strip().upper()
    tp = int(row.get("tensor_parallelism", "1") or 1)
    return f"{model}_{hw}_tp{tp}"


def _iter_manifest_rows(pair_manifest_csv: str) -> List[Dict[str, str]]:
    with open(pair_manifest_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if str(r.get("status", "")).strip() == "matched"]


def run_experimental_continuous_v1_prep(
    *,
    pair_manifest_csv: str = "results/stage0/pair_manifest.csv",
    out_dir: str = "results/experimental_continuous_v1",
    seed: int = 42,
    dt: float = 0.25,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    max_interp_gap_steps: int = 3,
    power_group_window_s: float = 0.05,
    min_traces_per_config: int = 3,
) -> Dict[str, object]:
    rows = _iter_manifest_rows(pair_manifest_csv)
    by_config: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        cid = _config_id_from_row(row)
        by_config.setdefault(cid, []).append(row)

    datasets_dir = os.path.join(out_dir, "datasets")
    splits_dir = os.path.join(out_dir, "splits")
    norms_dir = os.path.join(out_dir, "norm_params")
    _ensure_dir(datasets_dir)
    _ensure_dir(splits_dir)
    _ensure_dir(norms_dir)

    global_drop_reasons: Dict[str, int] = {}
    per_config_manifest: Dict[str, Dict[str, object]] = {}
    total_written = 0
    configs_written = 0

    def _drop(reason: str, config_drops: Dict[str, int]) -> None:
        global_drop_reasons[reason] = int(global_drop_reasons.get(reason, 0)) + 1
        config_drops[reason] = int(config_drops.get(reason, 0)) + 1

    for config_id, config_rows in sorted(by_config.items()):
        config_drops: Dict[str, int] = {}
        traces: List[TraceRecord] = []
        tp = int(config_rows[0].get("tensor_parallelism", "1") or 1)

        for row in config_rows:
            power, power_start, power_err = _extract_power_series(
                csv_path=str(row.get("power_csv_path", "")),
                tp=tp,
                dt=float(dt),
                power_group_window_s=float(power_group_window_s),
                max_interp_gap_steps=int(max_interp_gap_steps),
            )
            if power is None or power_start is None:
                _drop(power_err or "power_unknown_error", config_drops)
                continue

            reqs_raw, req_err, _ = _extract_request_log(str(row.get("json_path", "")))
            if reqs_raw is None:
                _drop(req_err or "request_unknown_error", config_drops)
                continue

            reqs, _shifted = _align_request_times(
                requests=reqs_raw,
                power_start_epoch_s=float(power_start),
                trace_duration_s=float(len(power) * float(dt)),
                dt=float(dt),
            )

            active, t_arrive, t_arrive_log = _compute_feature_series(
                power=power,
                requests=reqs,
                dt=float(dt),
            )
            if len(power) < 2:
                _drop("power_too_short", config_drops)
                continue

            traces.append(
                TraceRecord(
                    pair_key=str(row.get("pair_key", "")),
                    family=str(row.get("family", "")),
                    rate=str(row.get("rate", "")),
                    iteration=str(row.get("iteration", "")),
                    power_start_epoch_s=float(power_start),
                    power=np.asarray(power, dtype=np.float64),
                    active_requests=np.asarray(active, dtype=np.float64),
                    t_arrive=np.asarray(t_arrive, dtype=np.float64),
                    t_arrive_log=np.asarray(t_arrive_log, dtype=np.float64),
                    requests=list(reqs),
                )
            )

        if len(traces) < int(min_traces_per_config):
            _drop("config_too_few_traces", config_drops)
            per_config_manifest[config_id] = {
                "num_manifest_rows": int(len(config_rows)),
                "num_valid_traces": int(len(traces)),
                "drops": {k: int(v) for k, v in sorted(config_drops.items())},
                "written": False,
            }
            continue

        cfg_seed = _stable_seed(config_id, seed=int(seed))
        train_idx, val_idx, test_idx = _split_indices(
            n=len(traces),
            train_fraction=float(train_fraction),
            val_fraction=float(val_fraction),
            seed=cfg_seed,
        )
        if len(train_idx) == 0 or len(test_idx) == 0:
            _drop("config_split_invalid", config_drops)
            per_config_manifest[config_id] = {
                "num_manifest_rows": int(len(config_rows)),
                "num_valid_traces": int(len(traces)),
                "drops": {k: int(v) for k, v in sorted(config_drops.items())},
                "written": False,
            }
            continue

        train_traces = [traces[int(i)] for i in train_idx]
        norm = _fit_norm_params(train_traces)

        for tr in traces:
            x, y = _build_xy_for_trace(tr, norm=norm)
            tr.x_norm = x
            tr.y_norm = y

        slug = _safe_slug(config_id)
        dataset_path = os.path.join(datasets_dir, f"{slug}.npz")
        split_path = os.path.join(splits_dir, f"{slug}.json")
        norm_path = os.path.join(norms_dir, f"{slug}.json")

        np.savez(
            dataset_path,
            config_id=np.array([config_id], dtype=object),
            dt=np.array([float(dt)], dtype=np.float64),
            pair_key=np.asarray([tr.pair_key for tr in traces], dtype=object),
            family=np.asarray([tr.family for tr in traces], dtype=object),
            rate=np.asarray([tr.rate for tr in traces], dtype=object),
            iteration=np.asarray([tr.iteration for tr in traces], dtype=object),
            power_start_epoch_s=np.asarray([tr.power_start_epoch_s for tr in traces], dtype=np.float64),
            power=np.asarray([tr.power for tr in traces], dtype=object),
            active_requests=np.asarray([tr.active_requests for tr in traces], dtype=object),
            t_arrive=np.asarray([tr.t_arrive for tr in traces], dtype=object),
            t_arrive_log=np.asarray([tr.t_arrive_log for tr in traces], dtype=object),
            x_norm=np.asarray([tr.x_norm for tr in traces], dtype=object),
            y_norm=np.asarray([tr.y_norm for tr in traces], dtype=object),
        )

        _write_json(
            split_path,
            {
                "config_id": config_id,
                "seed": int(cfg_seed),
                "num_traces": int(len(traces)),
                "train_fraction": float(train_fraction),
                "val_fraction": float(val_fraction),
                "train_indices": [int(i) for i in train_idx.tolist()],
                "val_indices": [int(i) for i in val_idx.tolist()],
                "test_indices": [int(i) for i in test_idx.tolist()],
            },
        )

        _write_json(
            norm_path,
            {
                "config_id": config_id,
                "dt": float(dt),
                **{k: float(v) for k, v in norm.items()},
            },
        )

        configs_written += 1
        total_written += int(len(traces))
        per_config_manifest[config_id] = {
            "num_manifest_rows": int(len(config_rows)),
            "num_valid_traces": int(len(traces)),
            "num_train_traces": int(len(train_idx)),
            "num_val_traces": int(len(val_idx)),
            "num_test_traces": int(len(test_idx)),
            "drops": {k: int(v) for k, v in sorted(config_drops.items())},
            "written": True,
            "dataset_npz": dataset_path,
            "split_json": split_path,
            "norm_params_json": norm_path,
        }

    manifest = {
        "schema_version": "experimental-continuous-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "pair_manifest_csv": pair_manifest_csv,
        },
        "defaults": {
            "dt": float(dt),
            "seed": int(seed),
            "train_fraction": float(train_fraction),
            "val_fraction": float(val_fraction),
            "max_interp_gap_steps": int(max_interp_gap_steps),
            "power_group_window_s": float(power_group_window_s),
            "min_traces_per_config": int(min_traces_per_config),
        },
        "summary": {
            "num_manifest_rows": int(len(rows)),
            "num_configs_seen": int(len(by_config)),
            "num_configs_written": int(configs_written),
            "num_traces_written": int(total_written),
            "drop_reasons": {k: int(v) for k, v in sorted(global_drop_reasons.items())},
        },
        "configs": per_config_manifest,
    }
    _write_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare standalone Experimental Continuous v1 datasets from Stage 0 pair manifest."
    )
    parser.add_argument(
        "--pair_manifest_csv",
        default="results/stage0/pair_manifest.csv",
        help="Path to Stage 0 pair manifest CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/experimental_continuous_v1",
        help="Output directory for experimental artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--max_interp_gap_steps", type=int, default=3)
    parser.add_argument("--power_group_window_s", type=float, default=0.05)
    parser.add_argument("--min_traces_per_config", type=int, default=3)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experimental_continuous_v1_prep(
        pair_manifest_csv=args.pair_manifest_csv,
        out_dir=args.out_dir,
        seed=args.seed,
        dt=args.dt,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        max_interp_gap_steps=args.max_interp_gap_steps,
        power_group_window_s=args.power_group_window_s,
        min_traces_per_config=args.min_traces_per_config,
    )


if __name__ == "__main__":
    main()
