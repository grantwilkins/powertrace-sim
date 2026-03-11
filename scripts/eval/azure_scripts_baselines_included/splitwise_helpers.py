#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scripts.eval.pipeline_utils import build_rollout_features_from_requests

from .io_utils import load_json, resolve_existing_path


def load_pair_manifest_map(pair_manifest_csv: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not pair_manifest_csv:
        return out
    csv_path = Path(pair_manifest_csv)
    if not csv_path.exists():
        return out

    base_dir = str(csv_path.resolve().parent)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip() != "matched":
                continue
            key = str(row.get("pair_key", "")).strip()
            json_path_raw = str(row.get("json_path", "")).strip()
            if key == "" or json_path_raw == "":
                continue
            json_path = resolve_existing_path(json_path_raw, base_dir)
            if json_path is not None:
                out[key] = json_path
    return out


def _finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _synthesize_request_timestamps(payload: Dict[str, object], n: int) -> Optional[List[float]]:
    if n <= 0:
        return []

    duration = _finite_float(payload.get("duration"))
    if duration is not None and duration > 0:
        step = float(duration) / float(max(n, 1))
        if step > 0:
            values = (np.arange(n, dtype=np.float64) + 0.5) * step + 1.0
            return [float(x) for x in values]

    request_rate = _finite_float(payload.get("request_rate"))
    poisson_rate = _finite_float(payload.get("poisson_rate"))
    rate = request_rate if request_rate is not None else poisson_rate
    if rate is not None and rate > 0:
        step = 1.0 / float(rate)
        values = (np.arange(n, dtype=np.float64) + 1.0) * step + 1.0
        return [float(x) for x in values]
    return None


def build_requests_from_stage0_json(
    request_json_path: str,
    *,
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
    require_recorded_timestamps: bool = True,
) -> List[Dict[str, float]]:
    payload = load_json(request_json_path)
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
        if bool(require_recorded_timestamps):
            raise ValueError(
                "request json missing arrays: ['request_timestamps'] "
                "(synthetic fallback disabled)"
            )
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
        # Preserve request order, but clamp into feasible rollout window.
        arrivals = np.clip(arrivals, a_min=0.0, a_max=float(trace_duration_s) - float(dt))

    out: List[Dict[str, float]] = []
    for i in range(n):
        try:
            n_in = int(float(input_lens[i]))
            n_out = int(float(output_lens[i]))
        except Exception:
            continue
        if n_in <= 0 or n_out < 0:
            continue
        out.append(
            {
                "arrival_time": float(arrivals[i]),
                "input_tokens": float(n_in),
                "output_tokens": float(n_out),
            }
        )
    out.sort(key=lambda r: float(r["arrival_time"]))
    return out


def suppress_short_true_runs(mask: np.ndarray, min_run_len: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.size == 0 or int(min_run_len) <= 1:
        return arr
    out = arr.copy()
    i = 0
    n = int(arr.size)
    min_len = int(min_run_len)
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        if (j - i) < min_len:
            out[i:j] = False
        i = j
    return out


def splitwise_phase_masks(
    *,
    a_raw: np.ndarray,
    delta_a_raw: np.ndarray,
    phase_eps: float = 1e-6,
    prefill_delta_threshold: float = 0.05,
    prefill_min_steps: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a_raw, dtype=np.float64).reshape(-1)
    da = np.asarray(delta_a_raw, dtype=np.float64).reshape(-1)
    n = int(min(a.size, da.size))
    if n <= 0:
        z = np.zeros((0,), dtype=bool)
        return z, z, z
    a_n = a[:n]
    da_n = da[:n]
    active = a_n > float(phase_eps)
    prefill = active & (da_n > max(float(phase_eps), float(prefill_delta_threshold)))
    prefill = suppress_short_true_runs(prefill, int(prefill_min_steps))
    decode = active & (~prefill)
    idle = ~active
    return idle, decode, prefill


def estimate_splitwise_phase_targets_from_indices(
    *,
    indices: Sequence[int],
    pair_key_arr: np.ndarray,
    power_arr: np.ndarray,
    power_start_arr: np.ndarray,
    pair_map: Mapping[str, str],
    throughput: Mapping[str, float],
    norm_cfg: Mapping[str, float],
    feature_set: str,
    dt: float,
    non_gpu_overhead_w: float,
    phase_eps: float = 1e-6,
    prefill_delta_threshold: float = 0.05,
    prefill_min_steps: int = 2,
    require_recorded_timestamps: bool = True,
) -> Dict[str, float]:
    idle_vals: List[np.ndarray] = []
    decode_vals: List[np.ndarray] = []
    prefill_vals: List[np.ndarray] = []

    n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))
    for idx_raw in indices:
        idx = int(idx_raw)
        if idx < 0 or idx >= n_total:
            continue

        power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if power.size < 2:
            continue

        pair_key = str(pair_key_arr[idx])
        json_path = pair_map.get(pair_key)
        if json_path is None:
            continue

        try:
            requests = build_requests_from_stage0_json(
                json_path,
                power_start_epoch_s=float(power_start_arr[idx]),
                trace_duration_s=float(power.size * dt),
                dt=float(dt),
                require_recorded_timestamps=bool(require_recorded_timestamps),
            )
            gt_node = power[1:].astype(np.float64)
            feat = build_rollout_features_from_requests(
                requests=requests,
                throughput=throughput,
                norm=norm_cfg,
                T=int(gt_node.size),
                dt=float(dt),
                feature_set=str(feature_set),
            )
        except Exception:
            continue

        a_raw = np.asarray(feat.get("A_raw", []), dtype=np.float64).reshape(-1)
        delta_a_raw = np.asarray(feat.get("delta_A_raw", []), dtype=np.float64).reshape(-1)
        n_eval = int(min(gt_node.size, a_raw.size, delta_a_raw.size))
        if n_eval <= 0:
            continue

        gpu_power = np.clip(gt_node[:n_eval] - float(non_gpu_overhead_w), a_min=0.0, a_max=None)
        idle_mask, decode_mask, prefill_mask = splitwise_phase_masks(
            a_raw=a_raw[:n_eval],
            delta_a_raw=delta_a_raw[:n_eval],
            phase_eps=float(phase_eps),
            prefill_delta_threshold=float(prefill_delta_threshold),
            prefill_min_steps=int(prefill_min_steps),
        )

        if np.any(idle_mask):
            idle_vals.append(gpu_power[idle_mask].astype(np.float64))
        if np.any(decode_mask):
            decode_vals.append(gpu_power[decode_mask].astype(np.float64))
        if np.any(prefill_mask):
            prefill_vals.append(gpu_power[prefill_mask].astype(np.float64))

    out: Dict[str, float] = {}
    if len(idle_vals) > 0:
        out["target_idle_node_gpu_w"] = float(np.mean(np.concatenate(idle_vals)))
    if len(decode_vals) > 0:
        out["target_decode_node_gpu_w"] = float(np.mean(np.concatenate(decode_vals)))
    if len(prefill_vals) > 0:
        out["target_prefill_node_gpu_w"] = float(np.mean(np.concatenate(prefill_vals)))
    out["splitwise_phase_samples_idle"] = float(sum(int(x.size) for x in idle_vals))
    out["splitwise_phase_samples_decode"] = float(sum(int(x.size) for x in decode_vals))
    out["splitwise_phase_samples_prefill"] = float(sum(int(x.size) for x in prefill_vals))
    return out
