from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_active_requests(
    power_timestamps: np.ndarray,
    request_timestamps: np.ndarray,
    ttfts: np.ndarray,
    decode_times: np.ndarray,
) -> np.ndarray:
    """Compute active request count at each power measurement time."""
    power_t = np.asarray(power_timestamps, dtype=np.float64).reshape(-1)
    req_t = np.asarray(request_timestamps, dtype=np.float64).reshape(-1)
    ttft = np.asarray(ttfts, dtype=np.float64).reshape(-1)
    dec = np.asarray(decode_times, dtype=np.float64).reshape(-1)

    n_power = int(power_t.size)
    active = np.zeros(n_power, dtype=np.float64)
    n_req = int(min(req_t.size, ttft.size, dec.size))
    if n_req <= 0:
        return active

    start_times = np.sort(req_t[:n_req])
    end_times = np.sort(req_t[:n_req] + ttft[:n_req] + dec[:n_req])
    started_by_t = np.searchsorted(start_times, power_t, side="right")
    ended_before_t = np.searchsorted(end_times, power_t, side="left")
    active = (started_by_t - ended_before_t).astype(np.float64)

    return active


def compute_t_arrive_log(
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


def align_trace_to_grid(
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
    dt_values = np.diff(timestamps)
    dt = float(np.median(dt_values)) if len(dt_values) > 0 else 0.25

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
    elif request_ts.size > 0:
        # Some benchmark JSONs use a different clock origin than nvidia-smi.
        # If arrivals fall well outside the power window, rebase to preserve
        # measured inter-arrival structure while aligning to power start.
        power_start = float(timestamps[0]) + float(power_start_offset_s)
        trace_duration = float(timestamps[-1] - timestamps[0])
        arrivals = request_ts - power_start
        if float(np.min(arrivals)) < -float(dt) or float(np.max(arrivals)) > trace_duration + float(dt):
            arrivals = arrivals - float(np.min(arrivals))
            request_ts = power_start + arrivals

    active = compute_active_requests(timestamps, request_ts, ttfts, decode_times)
    t_arrive_log = compute_t_arrive_log(timestamps, request_ts)

    if not (np.all(np.isfinite(power)) and np.all(np.isfinite(active))):
        return None

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
