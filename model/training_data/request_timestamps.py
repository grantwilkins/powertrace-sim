from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


def compute_aligned_request_timestamps(
    *,
    power_timestamps_s: Iterable[float],
    poisson_rate: float,
    time_steps: Optional[Iterable[float]] = None,
    recorded_request_timestamps_s: Optional[Iterable[float]] = None,
    timestamp_source: str = "poisson",
) -> List[float]:
    """
    Compute request timestamps aligned to the power measurement time window.

    This helper is intentionally lightweight (numpy-only) so it can be unit-tested
    without pandas/pyarrow environment coupling.

    Args:
        power_timestamps_s: epoch seconds from power measurements.
        poisson_rate: synthetic request rate (req/s) used when generating timestamps.
        time_steps: optional relative times (seconds) used for linear scaling into the power window.
        recorded_request_timestamps_s: optional epoch seconds recorded by the benchmark client.
        timestamp_source:
          - "poisson": generate synthetic timestamps (legacy)
          - "recorded": use recorded timestamps; requires they all be valid
          - "recorded_or_poisson": use recorded if all valid else fall back
          - "recorded_scaled_or_poisson": linearly rescale recorded into power window if valid else fall back

    Returns:
        List[float] request timestamps in epoch seconds.
    """
    power_ts = np.asarray(list(power_timestamps_s), dtype=float)
    if power_ts.size == 0:
        raise ValueError("power_timestamps_s must be non-empty")

    min_timestamp = float(np.min(power_ts))
    max_timestamp = float(np.max(power_ts))
    time_range = max_timestamp - min_timestamp

    def _all_valid(ts: np.ndarray) -> bool:
        return bool(ts.size > 0 and np.all(np.isfinite(ts)) and np.all(ts > 0))

    recorded = None
    if recorded_request_timestamps_s is not None:
        recorded = np.asarray(list(recorded_request_timestamps_s), dtype=float)

    if timestamp_source in {
        "recorded",
        "recorded_or_poisson",
        "recorded_scaled_or_poisson",
    }:
        if recorded is not None and _all_valid(recorded):
            if timestamp_source == "recorded_scaled_or_poisson":
                rec_min = float(np.min(recorded))
                rec_max = float(np.max(recorded))
                rec_range = rec_max - rec_min
                if rec_range > 0 and time_range > 0:
                    scaled = min_timestamp + ((recorded - rec_min) / rec_range) * time_range
                    return [float(x) for x in scaled]
            return [float(x) for x in recorded]
        if timestamp_source == "recorded":
            raise ValueError(
                "timestamp_source='recorded' requires valid recorded_request_timestamps_s for all requests."
            )

    # Legacy: scale provided time steps into the power window if present.
    if time_steps is not None:
        steps = np.asarray(list(time_steps), dtype=float)
        if steps.size == 0:
            return []
        max_relative = float(np.max(steps))
        if max_relative <= 0 or time_range <= 0:
            return [float(min_timestamp) for _ in range(len(steps))]
        scaled = min_timestamp + (steps / max_relative) * time_range
        return [float(x) for x in scaled]

    # Synthetic Poisson timestamps inside [min_timestamp, max_timestamp]
    if poisson_rate <= 0:
        raise ValueError("poisson_rate must be positive")
    num_requests = int(recorded.size) if recorded is not None else 0
    if num_requests <= 0:
        raise ValueError(
            "Unable to infer num_requests: provide recorded_request_timestamps_s or time_steps."
        )

    timestamps: List[float] = []
    current_time = min_timestamp
    for _ in range(num_requests):
        interarrival = float(np.random.exponential(1.0 / poisson_rate))
        current_time += interarrival
        if time_range > 0 and current_time > max_timestamp:
            current_time = min_timestamp + (current_time - min_timestamp) % time_range
        timestamps.append(float(current_time))

    timestamps.sort()
    return timestamps

