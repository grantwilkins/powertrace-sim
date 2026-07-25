"""Frozen plan and result gate for cache-disabled disaggregated confirmation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import statistics
from datetime import datetime
from pathlib import Path

DURATION_S = 300
SMOKE_DURATION_S = 60
IDLE_WINDOW_S = 30
INPUT_TOKENS = 8192
OUTPUT_TOKENS = 64
TOKEN_RANGE_RATIO = 0.25
MINIMUM_PREFILL_S = 0.25


def find_free_port_offset(start_offset: int) -> int:
    for attempt in range(5000):
        offset = (start_offset + 4 * attempt) % 20000
        sockets = []
        try:
            for port in (
                21000 + offset,
                21001 + offset,
                21002 + offset,
                31000 + offset,
                31001 + offset,
            ):
                listener = socket.socket()
                sockets.append(listener)
                listener.bind(("127.0.0.1", port))
            return offset
        except OSError:
            pass
        finally:
            for listener in sockets:
                listener.close()
    raise RuntimeError("no free disaggregated port group found")


def cells(smoke: bool = False) -> list[tuple[float, int, int, int, str]]:
    """Return rate, repeat, prompts, seed, and frozen evidence split."""
    if smoke:
        return [(2.0, 0, 2 * SMOKE_DURATION_S, 0, "smoke")]
    return [
        (2.0, 0, 2 * DURATION_S, 0, "calibration"),
        (1.0, 0, DURATION_S, 2, "evaluation"),
        (2.0, 1, 2 * DURATION_S, 1, "evaluation"),
        (1.0, 1, DURATION_S, 2, "evaluation"),
        (2.0, 2, 2 * DURATION_S, 1, "evaluation"),
    ]


def run_metadata(
    prefill_uuid: str, decode_uuid: str, image: str, smoke: bool = False,
) -> dict:
    if not prefill_uuid or not decode_uuid or prefill_uuid == decode_uuid:
        raise ValueError("prefill and decode require distinct GPU UUIDs")
    local_now = datetime.now().astimezone()
    plan = cells(smoke)
    return {
        "model": "openai/gpt-oss-20b",
        "hardware": "A100-80GB",
        "deployment": "disaggregated_prefill",
        "protocol": "prospective_cache_disabled_confirmation_v1",
        "mode": "smoke" if smoke else "campaign",
        "roles": {
            "prefill": {"tp": 1, "gpu_uuid": prefill_uuid},
            "decode": {"tp": 1, "gpu_uuid": decode_uuid},
        },
        "server": {
            "max_model_len": 131072,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.9,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "async_scheduling": True,
            "kv_connector": "NixlConnector",
            "kv_load_failure_policy": "fail",
        },
        "workload": {
            "dataset": "random",
            "input_tokens": INPUT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "token_range_ratio": TOKEN_RANGE_RATIO,
            "shared_prefix_tokens": 0,
            "ignore_eos": True,
            "arrival_process": "poisson",
            "nominal_duration_s": SMOKE_DURATION_S if smoke else DURATION_S,
            "idle_window_s": IDLE_WINDOW_S,
            "cells": [
                {
                    "rate": rate,
                    "repeat": repeat,
                    "prompts": prompts,
                    "seed": seed,
                    "split": split,
                }
                for rate, repeat, prompts, seed, split in plan
            ],
        },
        "power_measurement": {
            "source": "nvidia-smi power.draw",
            "interval_ms": 250,
            "profile": "core_timed",
            "sample_timestamp": "host query start/end midpoint",
            "raw_samples": True,
        },
        "analysis_protocol": {
            "calibration_cell": "rate-2-repeat-0",
            "calibration_parameters": [
                "prefill idle",
                "decode idle",
                "prefill nonnegative dynamic gain",
                "decode nonnegative dynamic gain",
            ],
            "heldout_cells": [
                f"rate-{rate:g}-repeat-{repeat}"
                for rate, repeat, _, _, split in plan
                if split == "evaluation"
            ],
            "fit_loss": "pointwise squared error in watts per role",
            "dynamic_gain": "max(0, x dot y / x dot x) above fixed role idle",
            "equal_parameter_null": (
                "source-scheduler role busy fraction plus the same fixed idle "
                "and one fitted nonnegative gain"
            ),
            "sample_mapping": (
                "evaluate the 250 ms source-ledger bin containing each query "
                "midpoint; retain every sample and never average duplicate bins"
            ),
            "temporal_alignment": "no fitted lag, interpolation, or warping",
            "plot_processing": "no smoothing, averaging, or interpolation",
            "heldout_role_acceptance": {
                "correlation_min": 0.8,
                "std_ratio_min": 0.8,
                "std_ratio_max": 1.25,
                "p95_error_pct_max": 10.0,
                "model_to_duty_null_loss_ratio_max": 0.95,
            },
            "measured_replay_acceptance": {
                "correlation_min": 0.8,
                "scope": "each role and load",
            },
            "uncertainty_unit": (
                "cell; report every held-out cell and paired replay range"
            ),
        },
        "container_image": image,
        "clock": {
            "local_utc_offset_s": float(local_now.utcoffset().total_seconds()),
            "power_timestamp_basis": "local query midpoint wall time",
            "engine_timestamp_basis": "unix_epoch",
            "request_timestamp_basis": "unix_epoch",
            "proxy_timestamp_basis": "unix_epoch_nanoseconds",
        },
    }


def validate_result(path: Path, expected: int) -> list[str]:
    result = json.loads(path.read_text())
    if result.get("completed") != expected:
        raise ValueError(
            f"{path} completed {result.get('completed')} of {expected} requests"
        )
    for key in (
        "input_lens", "output_lens", "ttfts", "itls", "request_timestamps",
        "request_ids",
    ):
        if len(result.get(key, [])) != expected:
            raise ValueError(f"{path} has {len(result.get(key, []))}/{expected} {key}")
    request_ids = result["request_ids"]
    if any(not value for value in request_ids) or len(set(request_ids)) != expected:
        raise ValueError(f"{path} request_ids must be nonempty and unique")
    return request_ids


def validate_workload(path: Path) -> None:
    result = json.loads(path.read_text())
    ranges = {
        "input_lens": (
            int(INPUT_TOKENS * (1.0 - TOKEN_RANGE_RATIO)),
            int(INPUT_TOKENS * (1.0 + TOKEN_RANGE_RATIO)),
        ),
        "output_lens": (
            int(OUTPUT_TOKENS * (1.0 - TOKEN_RANGE_RATIO)),
            int(OUTPUT_TOKENS * (1.0 + TOKEN_RANGE_RATIO)),
        ),
    }
    for field, (lower, upper) in ranges.items():
        if any(not lower <= int(value) <= upper for value in result[field]):
            raise ValueError(
                f"{path} has {field} outside the frozen [{lower}, {upper}] range"
            )


def validate_events(
    path: Path, start_line: int, request_ids: list[str],
) -> dict[str, list[dict]]:
    sequence = (
        "proxy_received", "prefill_sent", "prefill_completed", "decode_sent",
        "decode_first_byte", "decode_completed",
    )
    expected_ids = set(request_ids)
    rows = [
        row for row in (
            json.loads(line) for line in path.read_text().splitlines()[start_line:]
        ) if row["request_id"] in expected_ids
    ]
    requests: dict[str, list[dict]] = {}
    for row in rows:
        requests.setdefault(row["request_id"], []).append(row)
    if set(requests) != expected_ids:
        raise ValueError(
            f"{path} has {len(requests)}/{len(request_ids)} measured request timelines"
        )
    for request_id, events in requests.items():
        if tuple(row["event"] for row in events) != sequence:
            raise ValueError(f"request {request_id} has incomplete stage events")
        times = [int(row["wall_ns"]) for row in events]
        if times != sorted(times):
            raise ValueError(f"request {request_id} has nonmonotonic stage events")
    return requests


def validate_prefill_observability(
    events: dict[str, list[dict]], minimum_s: float = MINIMUM_PREFILL_S,
) -> None:
    durations = []
    for rows in events.values():
        times = {row["event"]: int(row["wall_ns"]) / 1e9 for row in rows}
        durations.append(times["prefill_completed"] - times["prefill_sent"])
    median = float(statistics.median(durations))
    if median < minimum_s:
        raise ValueError(
            f"median prefill duration {median:.3f}s is below {minimum_s:.3f}s"
        )


def _counter_delta(path: Path, field: str) -> float:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2 or field not in rows[0]:
        raise ValueError(f"{path} lacks two samples of {field}")
    values = [float(row[field]) for row in rows]
    if any(value != value for value in values):
        raise ValueError(f"{path} has non-finite {field}")
    delta = values[-1] - values[0]
    if delta < 0.0:
        raise ValueError(f"{path} counter {field} decreased")
    return delta


def validate_uncached_metrics(
    request_path: Path, prefill_path: Path, decode_path: Path,
) -> None:
    request = json.loads(request_path.read_text())
    hits = _counter_delta(prefill_path, "prefix_cache_hits_total")
    if hits != 0.0:
        raise ValueError(f"{prefill_path} recorded {hits:g} cached prompt tokens")
    transfers = _counter_delta(decode_path, "nixl_xfer_time_seconds_count")
    if transfers != float(request["completed"]):
        raise ValueError(
            f"{decode_path} recorded {transfers:g} transfers for "
            f"{request['completed']} measured requests"
        )
    observed_tokens = _counter_delta(prefill_path, "prompt_tokens_total")
    declared_tokens = float(sum(request["input_lens"]))
    relative_error = abs(observed_tokens - declared_tokens) / declared_tokens
    if relative_error > 0.01:
        raise ValueError(
            f"{prefill_path} prompt-token counter differs from requests by "
            f"{100.0 * relative_error:.2f}%"
        )
    for role_path in (prefill_path, decode_path):
        for field in (
            "nixl_num_failed_transfers",
            "nixl_num_failed_notifications",
            "nixl_num_kv_expired_reqs",
            "num_preemptions_total",
        ):
            value = _counter_delta(role_path, field)
            if value != 0.0:
                raise ValueError(f"{role_path} recorded {value:g} {field}")


def _power_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f")


def _median_field(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"power.csv has invalid {field}")
    return float(statistics.median(values))


def validate_power(
    path: Path,
    *,
    gpu_uuids: tuple[str, str],
    workload_start: float,
    workload_end: float,
) -> None:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "timestamp", "query.start", "query.end", "uuid", "power.draw [W]",
        "temperature.gpu",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} lacks timed two-GPU power fields")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["timestamp"], []).append(row)
    expected = set(gpu_uuids)
    sample_times = []
    for timestamp, group in groups.items():
        if len(group) != 2 or {row["uuid"] for row in group} != expected:
            raise ValueError(f"{path} has an incomplete GPU sample at {timestamp}")
        if len({row["query.start"] for row in group}) != 1 or len({
            row["query.end"] for row in group
        }) != 1:
            raise ValueError(f"{path} has inconsistent query bounds at {timestamp}")
        start = _power_time(group[0]["query.start"])
        midpoint = _power_time(timestamp)
        end = _power_time(group[0]["query.end"])
        duration = (end - start).total_seconds()
        if not start <= midpoint <= end or duration > 0.2:
            raise ValueError(f"{path} has invalid query timing at {timestamp}")
        _median_field(group, "power.draw [W]")
        sample_times.append(midpoint.timestamp())
    sample_times.sort()
    spacings = [
        right - left for left, right in zip(sample_times, sample_times[1:])
    ]
    if (
        len(sample_times) < 4
        or not 0.15 <= statistics.median(spacings) <= 0.5
        or max(spacings) > 0.75
    ):
        raise ValueError(f"{path} does not maintain the 250 ms sampling contract")
    if (
        sample_times[0] > workload_start - 25.0
        or sample_times[-1] < workload_end + 25.0
    ):
        raise ValueError(f"{path} does not cover both 30-second idle windows")
    for uuid in gpu_uuids:
        pre = [
            row for row in rows
            if row["uuid"] == uuid
            and _power_time(row["timestamp"]).timestamp() < workload_start
        ]
        post = [
            row for row in rows
            if row["uuid"] == uuid
            and _power_time(row["timestamp"]).timestamp() > workload_end
        ]
        if len(pre) < 80 or len(post) < 80:
            raise ValueError(f"{path} has insufficient idle samples for {uuid}")
        power_drift = abs(
            _median_field(pre, "power.draw [W]")
            - _median_field(post, "power.draw [W]")
        )
        temperature_drift = abs(
            _median_field(pre, "temperature.gpu")
            - _median_field(post, "temperature.gpu")
        )
        if power_drift > 5.0 or temperature_drift > 5.0:
            raise ValueError(
                f"{path} idle state did not restore for {uuid}: "
                f"{power_drift:.1f} W, {temperature_drift:.1f} C drift"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--smoke", action="store_true")
    ports = sub.add_parser("ports")
    ports.add_argument("--start-offset", type=int, required=True)
    check = sub.add_parser("check")
    check.add_argument("path", type=Path)
    check.add_argument("expected", type=int)
    check.add_argument("--events", type=Path)
    check.add_argument("--event-start-line", type=int, default=0)
    metadata = sub.add_parser("metadata")
    metadata.add_argument("path", type=Path)
    metadata.add_argument("--prefill-uuid", required=True)
    metadata.add_argument("--decode-uuid", required=True)
    metadata.add_argument("--image", required=True)
    metadata.add_argument("--smoke", action="store_true")
    check.add_argument("--prefill-metrics", type=Path, required=True)
    check.add_argument("--decode-metrics", type=Path, required=True)
    check.add_argument("--power", type=Path, required=True)
    check.add_argument("--prefill-uuid", required=True)
    check.add_argument("--decode-uuid", required=True)
    check.add_argument("--workload-start", type=Path, required=True)
    check.add_argument("--workload-end", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        for rate, repeat, prompts, seed, split in cells(args.smoke):
            print(f"{rate:g}\t{repeat}\t{prompts}\t{seed}\t{split}")
    elif args.command == "ports":
        print(find_free_port_offset(args.start_offset))
    elif args.command == "check":
        request_ids = validate_result(args.path, args.expected)
        validate_workload(args.path)
        if args.events:
            timelines = validate_events(
                args.events, args.event_start_line, request_ids
            )
            validate_prefill_observability(timelines)
        validate_uncached_metrics(
            args.path, args.prefill_metrics, args.decode_metrics
        )
        validate_power(
            args.power,
            gpu_uuids=(args.prefill_uuid, args.decode_uuid),
            workload_start=float(args.workload_start.read_text()),
            workload_end=float(args.workload_end.read_text()),
        )
    else:
        value = run_metadata(
            args.prefill_uuid, args.decode_uuid, args.image, args.smoke
        )
        args.path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
