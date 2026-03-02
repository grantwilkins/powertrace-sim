"""
Shared utilities for Azure LLM inference trace processing.

This module provides common functions for loading and parsing Azure conversation traces
for use in datacenter power simulation studies.
"""

import csv
import os
import sys
from datetime import datetime
from typing import List

import numpy as np

# Add model path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../model"))

from simulators.arrival_simulator import ServeGenRequest  # noqa: E402


def load_and_parse_azure_csv(
    csv_path: str, scale_factor: float = 1.0, seed: int = 0
) -> List[ServeGenRequest]:
    """
    Load Azure conversation trace and convert to ServeGenRequest objects.

    This function reads an Azure LLM inference trace CSV file containing timestamps,
    context tokens, and generated tokens for each request. It normalizes timestamps
    to seconds from the start, optionally scales the arrival rate, and converts
    each row to a ServeGenRequest object.

    Args:
        csv_path: Path to Azure trace CSV with columns:
            - TIMESTAMP: ISO format with timezone (e.g., "2024-05-16 07:00:00.019229+00:00")
            - ContextTokens: Number of input tokens (int)
            - GeneratedTokens: Number of output tokens (int)
        scale_factor: Multiply arrival rate by this factor (default 1.0).
            Values > 1.0 compress time (increase rate).
            Values < 1.0 expand time (decrease rate).
            Example: scale_factor=50.0 makes trace run 50x faster
        seed: Random seed for reproducibility (currently unused, for future extensions)

    Returns:
        List of ServeGenRequest objects sorted by arrival_time in ascending order.
        Each request has:
            - request_id: Sequential index (0 to N-1)
            - arrival_time: Seconds from trace start (float)
            - input_tokens: Context tokens from CSV
            - output_tokens: Generated tokens from CSV

    Raises:
        FileNotFoundError: If csv_path doesn't exist
        ValueError: If CSV format is invalid (missing columns, unparseable timestamps)

    Example:
        >>> requests = load_and_parse_azure_csv(
        ...     "/path/to/trace.csv",
        ...     scale_factor=50.0,
        ...     seed=42
        ... )
        >>> print(f"Loaded {len(requests):,} requests")
        >>> print(f"Duration: {requests[-1].arrival_time / 3600:.1f} hours")
    """
    # Validate input
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Azure trace file not found: {csv_path}")

    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    # Set random seed for reproducibility (for potential future use)
    rng = np.random.RandomState(seed)

    # Load CSV
    rows = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            # Validate headers
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty or has no headers")

            required_cols = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
            missing_cols = required_cols - set(reader.fieldnames)
            if missing_cols:
                raise ValueError(
                    f"CSV missing required columns: {missing_cols}. "
                    f"Found columns: {reader.fieldnames}"
                )

            for row_idx, row in enumerate(reader):
                try:
                    rows.append(
                        {
                            "timestamp": row["TIMESTAMP"],
                            "context": int(row["ContextTokens"]),
                            "generated": int(row["GeneratedTokens"]),
                        }
                    )
                except (ValueError, KeyError) as e:
                    raise ValueError(
                        f"Error parsing row {row_idx + 2}: {e}. Row data: {row}"
                    )
    except Exception as e:
        raise ValueError(f"Failed to read Azure trace CSV: {e}")

    if not rows:
        raise ValueError(f"Azure trace CSV is empty: {csv_path}")

    # Parse timestamps
    timestamps = []
    failed_count = 0
    for i, row in enumerate(rows):
        ts_str = row["timestamp"]
        try:
            # Try with microseconds first
            if "." in ts_str:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f%z")
            else:
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S%z")
            timestamps.append(dt)
        except ValueError as e:
            failed_count += 1
            if failed_count <= 5:  # Only show first 5 errors
                print(
                    f"Warning: Failed to parse timestamp at row {i + 2}: '{ts_str}'. Error: {e}"
                )
            # Skip this row
            continue

    if failed_count > 0:
        print(f"Warning: Skipped {failed_count} rows due to timestamp parsing errors")
        # Remove failed rows
        rows = [rows[i] for i, _ in enumerate(timestamps) if i < len(timestamps)]

    if not timestamps:
        raise ValueError("No valid timestamps found in Azure trace")

    # Sort by time
    sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    timestamps = [timestamps[i] for i in sorted_indices]
    rows = [rows[i] for i in sorted_indices]

    # Convert to seconds from start
    start_time = timestamps[0]
    times_seconds = [(t - start_time).total_seconds() for t in timestamps]

    # Apply scaling factor by adjusting inter-arrival times
    if scale_factor != 1.0:
        # Compress/expand time intervals
        scaled_times = [t / scale_factor for t in times_seconds]
    else:
        scaled_times = times_seconds

    # Create ServeGenRequest objects
    requests = []
    for i, (t, row) in enumerate(zip(scaled_times, rows)):
        requests.append(
            ServeGenRequest(
                request_id=i,
                arrival_time=float(t),
                input_tokens=row["context"],
                output_tokens=row["generated"],
            )
        )

    return requests


def compute_trace_statistics(requests: List[ServeGenRequest]) -> dict:
    """
    Compute basic statistics for a request stream.

    Args:
        requests: List of ServeGenRequest objects (must be sorted by arrival_time)

    Returns:
        Dictionary with keys:
            - duration_s: Total duration in seconds (float)
            - num_requests: Total number of requests (int)
            - avg_rate_req_per_s: Average request rate (float)
            - avg_input_tokens: Mean input tokens per request (float)
            - avg_output_tokens: Mean output tokens per request (float)
            - p50_interarrival_ms: Median inter-arrival time in milliseconds (float)
            - p95_interarrival_ms: 95th percentile inter-arrival time in ms (float)
            - p99_interarrival_ms: 99th percentile inter-arrival time in ms (float)

    Example:
        >>> stats = compute_trace_statistics(requests)
        >>> print(f"Average rate: {stats['avg_rate_req_per_s']:.2f} req/s")
        >>> print(f"Mean input tokens: {stats['avg_input_tokens']:.0f}")
    """
    if not requests:
        return {
            "duration_s": 0.0,
            "num_requests": 0,
            "avg_rate_req_per_s": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "p50_interarrival_ms": 0.0,
            "p95_interarrival_ms": 0.0,
            "p99_interarrival_ms": 0.0,
        }

    arrival_times = [r.arrival_time for r in requests]
    input_tokens = [r.input_tokens for r in requests]
    output_tokens = [r.output_tokens for r in requests]

    duration_s = arrival_times[-1] - arrival_times[0]
    num_requests = len(requests)

    # Avoid division by zero
    avg_rate = num_requests / duration_s if duration_s > 0 else 0.0

    # Inter-arrival times (time between consecutive requests)
    interarrivals = []
    for i in range(1, len(arrival_times)):
        interarrivals.append((arrival_times[i] - arrival_times[i - 1]) * 1000)  # Convert to ms

    # Compute percentiles
    if interarrivals:
        p50 = float(np.percentile(interarrivals, 50))
        p95 = float(np.percentile(interarrivals, 95))
        p99 = float(np.percentile(interarrivals, 99))
    else:
        p50 = p95 = p99 = 0.0

    return {
        "duration_s": float(duration_s),
        "num_requests": num_requests,
        "avg_rate_req_per_s": float(avg_rate),
        "avg_input_tokens": float(np.mean(input_tokens)),
        "avg_output_tokens": float(np.mean(output_tokens)),
        "p50_interarrival_ms": p50,
        "p95_interarrival_ms": p95,
        "p99_interarrival_ms": p99,
    }
