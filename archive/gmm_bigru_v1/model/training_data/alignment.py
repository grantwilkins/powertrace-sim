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
    ended_before_t = np.searchsorted(end_times, power_t, side="right")
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


def resample_trace_to_grid(trace: Dict[str, object], *, dt: float) -> Dict[str, object]:
    """Project one aligned trace onto an explicit regular time grid."""
    target_dt = float(dt)
    if not np.isfinite(target_dt) or target_dt <= 0.0:
        raise ValueError("target dt must be positive and finite")
    source_dt = float(trace.get("dt", target_dt))
    if np.isclose(source_dt, target_dt, rtol=0.0, atol=1e-6):
        out = dict(trace)
        out["dt"] = target_dt
        return out
    timestamps = np.asarray(trace["timestamps"], dtype=np.float64).reshape(-1)
    power = np.asarray(trace["power"], dtype=np.float64).reshape(-1)
    if timestamps.size < 3 or power.size != timestamps.size:
        raise ValueError("trace requires aligned power timestamps")
    num_points = int(
        np.floor((float(timestamps[-1]) - float(timestamps[0])) / target_dt)
    ) + 1
    if num_points < 3:
        raise ValueError("resampled trace has fewer than three points")
    grid = float(timestamps[0]) + np.arange(num_points, dtype=np.float64) * target_dt
    request_timestamps = np.asarray(trace["request_timestamps"], dtype=np.float64)
    ttfts = np.asarray(trace["ttfts"], dtype=np.float64)
    decode_times = np.asarray(trace["decode_times"], dtype=np.float64)
    out = dict(trace)
    out.update(
        {
            "power": np.interp(grid, timestamps, power),
            "active_requests": compute_active_requests(
                grid, request_timestamps, ttfts, decode_times
            ),
            "t_arrive_log": compute_t_arrive_log(grid, request_timestamps),
            "timestamps": grid,
            "dt": target_dt,
            "power_start_epoch_s": float(grid[0]),
            "num_points": int(grid.size),
            "power_resampled_to_dt": True,
        }
    )
    return out


def align_arrivals(
    request_timestamps: np.ndarray,
    power_t0: float,
    *,
    policy: str,
    dt: Optional[float] = None,
    trace_duration_s: Optional[float] = None,
) -> tuple[np.ndarray, bool, bool]:
    """One shared implementation of the repo's arrival-alignment policies (D5).

    The two policies intentionally differ and are NOT interchangeable; callers
    pick one by name and should record it in provenance:

    - ``rebase_into_window``: when arrivals fall outside the power window
      (beyond one dt of slack), shift so the earliest arrival lands at the
      window start. Preserves inter-arrival structure, destroys any residual
      clock offset. Never rejects. Requires ``dt`` and ``trace_duration_s``.
    - ``fold_1800``: subtract the nearest whole multiple of 1800 s from the
      earliest arrival's offset — cancels whole/half-hour clock skew while
      preserving the residual offset — then require the earliest arrival in
      [-2 s, 600 s] of power start.

    Returns (arrivals relative to power_t0, ok, shifted). ``ok`` is False only
    when a policy's validity gate rejects the run. ``shifted`` is False when
    the policy left the raw offsets untouched — callers keeping absolute
    timestamps must then keep their originals bit-for-bit rather than
    reconstructing them as ``power_t0 + arrivals`` (float round-trip is not
    exact).
    """
    arr = np.asarray(request_timestamps, dtype=np.float64).reshape(-1) - float(power_t0)

    if policy == "rebase_into_window":
        if dt is None or trace_duration_s is None:
            raise ValueError("rebase_into_window requires dt and trace_duration_s")
        if arr.size > 0 and (
            float(np.min(arr)) < -float(dt)
            or float(np.max(arr)) > float(trace_duration_s) + float(dt)
        ):
            return arr - float(np.min(arr)), True, True
        return arr, True, False

    if policy == "fold_1800":
        if arr.size == 0:
            return arr, False, False
        arr = arr - round(float(np.min(arr)) / 1800.0) * 1800.0
        ok = not (float(np.min(arr)) < -2.0 or float(np.min(arr)) > 600.0)
        return arr, ok, True

    raise ValueError(f"Unknown alignment policy: {policy!r}")


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
    tolerance = max(1e-6, 0.05 * dt)
    if dt <= 0.0:
        return None
    resampled_power = False
    if not np.all(np.abs(dt_values - dt) <= tolerance):
        # nvidia-smi traces can miss up to three consecutive 4 Hz samples.
        # Project only these sub-second holes onto the median-cadence grid;
        # longer discontinuities remain explicit parse failures.
        if np.any(dt_values > 4.05 * dt):
            return None
        num_points = int(np.floor((float(timestamps[-1]) - float(timestamps[0])) / dt)) + 1
        if num_points < 3:
            return None
        regular_timestamps = float(timestamps[0]) + np.arange(num_points) * dt
        power = np.interp(regular_timestamps, timestamps, power)
        timestamps = regular_timestamps
        resampled_power = True

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
        arrivals, _, shifted = align_arrivals(
            request_ts,
            power_start,
            policy="rebase_into_window",
            dt=dt,
            trace_duration_s=trace_duration,
        )
        if shifted:
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
        "power_resampled_to_dt": resampled_power,
        "power_start_epoch_s": float(timestamps[0]),
        "num_points": len(power),
        "input_lens": request_data["input_lens"],
        "output_lens": request_data["output_lens"],
        "ttfts": request_data["ttfts"],
        "decode_times": request_data["decode_times"],
        "request_timestamps": request_ts,
    }

