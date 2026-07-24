"""Role-separated vLLM and NIXL telemetry for disaggregated serving."""
from __future__ import annotations

import argparse
import csv
import math
import signal
import time
import urllib.request
from pathlib import Path

from profiling.client.metrics_logger import (
    ENGINE_HEADER,
    available_columns,
    metrics_row,
    parse_prometheus_metrics,
)

REQUIRED_COLUMNS = (
    "num_requests_running",
    "num_requests_waiting",
    "gpu_cache_usage_perc",
    "prompt_tokens_total",
    "generation_tokens_total",
    "iteration_tokens_total_sum",
    "iteration_tokens_total_count",
    "num_preemptions_total",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "nixl_xfer_time_seconds_count",
    "nixl_num_failed_transfers",
    "nixl_num_failed_notifications",
    "nixl_num_kv_expired_reqs",
)

NIXL_COLUMNS = (
    "nixl_xfer_time_seconds_sum",
    "nixl_xfer_time_seconds_count",
    "nixl_post_time_seconds_sum",
    "nixl_post_time_seconds_count",
    "nixl_bytes_transferred_sum",
    "nixl_bytes_transferred_count",
    "nixl_num_descriptors_sum",
    "nixl_num_descriptors_count",
    "nixl_num_failed_transfers",
    "nixl_num_failed_notifications",
    "nixl_num_kv_expired_reqs",
)

_STOP = False


def _metric(parsed: dict[str, list[float]], column: str) -> float:
    values = parsed.get(f"vllm:{column}")
    if values is None:
        values = parsed.get(f"vllm:{column}_total")
    return float(sum(values)) if values is not None else float("nan")


def role_row(parsed: dict[str, list[float]], timestamp: float) -> list[float]:
    return metrics_row(parsed, timestamp) + [
        _metric(parsed, column) for column in NIXL_COLUMNS
    ]


def _fetch(url: str) -> dict[str, list[float]]:
    with urllib.request.urlopen(f"{url.rstrip('/')}/metrics", timeout=5) as response:
        return parse_prometheus_metrics(response.read().decode())


def missing_required(parsed: dict[str, list[float]]) -> list[str]:
    available = available_columns(parsed)
    available.update(
        column for column in NIXL_COLUMNS
        if math.isfinite(_metric(parsed, column))
    )
    return sorted(set(REQUIRED_COLUMNS) - available)


def _preflight(url: str) -> None:
    missing = missing_required(_fetch(url))
    if missing:
        raise ValueError(f"{url}/metrics lacks required evidence columns: {missing}")


def _stop(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def capture(prefill_url: str, decode_url: str, out_dir: Path, period_s: float) -> None:
    _preflight(prefill_url)
    _preflight(decode_url)
    signal.signal(signal.SIGTERM, _stop)
    paths = {
        "prefill": out_dir / "engine_prefill.csv",
        "decode": out_dir / "engine_decode.csv",
    }
    with paths["prefill"].open("w", newline="") as prefill_stream, paths[
        "decode"
    ].open("w", newline="") as decode_stream:
        writers = {
            "prefill": csv.writer(prefill_stream),
            "decode": csv.writer(decode_stream),
        }
        for writer in writers.values():
            writer.writerow(ENGINE_HEADER + list(NIXL_COLUMNS))
        while not _STOP:
            started = time.monotonic()
            timestamp = time.time()
            writers["prefill"].writerow(role_row(_fetch(prefill_url), timestamp))
            writers["decode"].writerow(role_row(_fetch(decode_url), timestamp))
            prefill_stream.flush()
            decode_stream.flush()
            remaining = period_s - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--period-s", type=float, default=0.25)
    args = parser.parse_args()
    capture(args.prefill_url, args.decode_url, args.out_dir, args.period_s)


if __name__ == "__main__":
    main()
