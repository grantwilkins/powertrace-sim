"""Frozen campaign and evidence gates for disaggregated phase transfer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from profiling.disaggregated_prefill.campaign import (
    find_free_port_offset,
    validate_events,
    validate_power,
    validate_result,
    validate_uncached_metrics,
)
from profiling.disaggregated_prefill.planned_workload import validate_plan

INTERVAL_S = 0.25


def cells(smoke: bool = False) -> list[tuple[str, float, int, int, str, str]]:
    if smoke:
        return [("smoke", 0.2, 4, 97, "smoke", "smoke")]
    return [
        ("stage-probe", 0.2, 24, 7, "stage_probe", "probe"),
        ("rate-0p5-calibration", 0.5, 150, 11, "calibration", "calibration"),
        ("rate-0p5-heldout", 0.5, 150, 17, "evaluation", "heldout"),
        ("rate-0p5-replay", 0.5, 150, 17, "evaluation", "heldout"),
    ]


def run_metadata(
    prefill_uuid: str, decode_uuid: str, image: str, smoke: bool = False
) -> dict:
    if not prefill_uuid or not decode_uuid or prefill_uuid == decode_uuid:
        raise ValueError("prefill and decode require distinct GPU UUIDs")
    local_now = datetime.now().astimezone()
    return {
        "model": "openai/gpt-oss-20b",
        "hardware": "A100-80GB",
        "deployment": "disaggregated_prefill",
        "protocol": "cache_disabled_phase_transfer_v2",
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
            "dataset": "immutable_random_plan",
            "input_tokens": 8192,
            "output_tokens": 256,
            "token_range_ratio": 0.25,
            "arrival_process": "fixed_deadline",
            "ignore_eos": True,
            "idle_window_s": 30,
            "cells": [
                {
                    "name": name,
                    "rate": rate,
                    "prompts": prompts,
                    "seed": seed,
                    "split": split,
                    "plan_key": key,
                }
                for name, rate, prompts, seed, split, key in cells(smoke)
            ],
        },
        "power_measurement": {
            "source": "nvidia-smi power.draw",
            "interval_ms": 250,
            "profile": "core_timed_state",
            "sample_timestamp": "host query start/end midpoint",
            "raw_samples": True,
        },
        "analysis_protocol": {
            "calibration_cell": "rate-0p5-calibration",
            "target_scalar_count": 6,
            "calibration_parameters": [
                "prefill_idle_w",
                "decode_idle_w",
                "prefill_dynamic_gain",
                "decode_dynamic_gain",
                "queue_free_prefill_time_scale",
                "fixed_nixl_handoff_delay_s",
            ],
            "heldout_cells": ["rate-0p5-heldout", "rate-0p5-replay"],
            "fit_loss": "phase-separated soft-DTW at raw 250 ms resolution",
            "plot_processing": "no smoothing, averaging, or interpolation",
            "replay_contract": "identical prompt bytes, lengths, outputs, and offsets",
        },
        "container_image": image,
        "clock": {
            "local_utc_offset_s": float(local_now.utcoffset().total_seconds()),
            "power_timestamp_basis": "local query midpoint wall time",
            "engine_timestamp_basis": "unix_epoch",
            "request_timestamp_basis": "unix_epoch",
        },
    }


def phase_locked_origin(
    first_midpoint_epoch_s: float,
    now_epoch_s: float | None = None,
    minimum_lead_s: float = 30.0,
) -> float:
    now = time.time() if now_epoch_s is None else now_epoch_s
    steps = math.ceil((now + minimum_lead_s - first_midpoint_epoch_s) / INTERVAL_S)
    return first_midpoint_epoch_s + max(0, steps) * INTERVAL_S


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_meter_phase(timestamps: list[float], origin: float) -> None:
    if len(timestamps) < 4:
        raise ValueError("power trace lacks enough timestamps for a phase gate")
    errors = []
    for timestamp in timestamps:
        cycles = (timestamp - origin) / INTERVAL_S
        errors.append(abs(cycles - round(cycles)) * INTERVAL_S)
    median = statistics.median(errors)
    p95 = _percentile(errors, 0.95)
    if median > 0.025 or p95 > 0.050:
        raise ValueError(
            f"power-meter phase error is {median:.3f}s median/{p95:.3f}s p95"
        )


def validate_unsaturated(
    path: Path, *, traffic_start: float, traffic_end: float
) -> None:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    waiting = [
        float(row["num_requests_waiting"])
        for row in rows
        if traffic_start <= float(row["timestamp"]) <= traffic_end
    ]
    if not waiting:
        raise ValueError(f"{path} has no queue samples in the traffic window")
    median = statistics.median(waiting)
    p95 = _percentile(waiting, 0.95)
    if median != 0:
        raise ValueError(f"{path} waiting median is {median:g}, expected zero")
    if p95 > 1:
        raise ValueError(f"{path} waiting p95 is {p95:g}, expected <= 1")


def _validate_throughput_drift(
    path: Path, *, traffic_start: float, traffic_end: float
) -> None:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    third = (traffic_end - traffic_start) / 3.0
    rates = []
    for left, right in (
        (traffic_start, traffic_start + third),
        (traffic_end - third, traffic_end),
    ):
        window = [
            row for row in rows if left <= float(row["timestamp"]) <= right
        ]
        if len(window) < 2:
            raise ValueError(f"{path} lacks samples for throughput drift")
        elapsed = float(window[-1]["timestamp"]) - float(window[0]["timestamp"])
        tokens = float(window[-1]["prompt_tokens_total"]) - float(
            window[0]["prompt_tokens_total"]
        )
        rates.append(tokens / elapsed)
    if min(rates) <= 0 or abs(rates[1] - rates[0]) / rates[0] > 0.10:
        raise ValueError(
            f"{path} first/last-third prompt throughput drift exceeds 10%"
        )


def _validate_power_state(path: Path) -> None:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    limits: dict[str, set[float]] = {}
    state_fields = (
        "clocks_event_reasons.hw_slowdown",
        "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.sw_thermal_slowdown",
    )
    required = {"uuid", "power.limit [W]", *state_fields}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} lacks frozen GPU state fields")
    for row in rows:
        limits.setdefault(row["uuid"], set()).add(float(row["power.limit [W]"]))
        active = [
            field
            for field in state_fields
            if row[field].strip().lower() not in {"not active", "no", "0"}
        ]
        if active:
            raise ValueError(f"{path} recorded GPU slowdown state {active}")
    if any(len(values) != 1 for values in limits.values()):
        raise ValueError(f"{path} recorded a power-limit change")


def fit_nixl_handoff_delay(
    decode_first_byte_s: list[float], *, frozen_first_iteration_s: float
) -> float:
    if not decode_first_byte_s or frozen_first_iteration_s < 0:
        raise ValueError("NIXL fit requires positive observations and timing")
    return max(
        0.0, statistics.median(decode_first_byte_s) - frozen_first_iteration_s
    )


def _power_timestamps(path: Path) -> list[float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = {
        datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        for row in rows
    }
    return sorted(values)


def _first_power_timestamp(path: Path) -> float:
    timestamps = _power_timestamps(path)
    if not timestamps:
        raise ValueError(f"{path} has no power samples")
    return timestamps[0]


def _validate_planned_result(result_path: Path, plan_path: Path) -> None:
    result = json.loads(result_path.read_text())
    plan = json.loads(plan_path.read_text())
    digest = validate_plan(plan)
    if result.get("request_plan_sha256") != digest:
        raise ValueError("result does not identify the supplied request plan")
    expected_inputs = [row["prompt_len"] for row in plan["requests"]]
    expected_outputs = [row["output_len"] for row in plan["requests"]]
    if result["input_lens"] != expected_inputs:
        raise ValueError("served input lengths differ from the immutable plan")
    if result["output_lens"] != expected_outputs:
        raise ValueError("served output lengths differ from the immutable plan")
    if result.get("planned_offsets_s") != [
        row["offset_s"] for row in plan["requests"]
    ]:
        raise ValueError("result offsets differ from the immutable plan")
    errors = [
        abs(observed - result["traffic_start_epoch_s"] - planned)
        for observed, planned in zip(
            result["request_timestamps"], result["planned_offsets_s"]
        )
    ]
    if _percentile(errors, 0.95) > 0.050:
        raise ValueError("fixed-deadline request issue error exceeds 50 ms p95")


def check(args: argparse.Namespace) -> None:
    start = float(args.traffic_start.read_text())
    end = float(args.traffic_end.read_text())
    request_ids = validate_result(args.path, args.expected)
    _validate_planned_result(args.path, args.plan)
    timelines = validate_events(args.events, args.event_start_line, request_ids)
    first_proxy = min(
        int(row["wall_ns"]) / 1e9
        for rows in timelines.values()
        for row in rows
        if row["event"] == "proxy_received"
    )
    if abs(first_proxy - start) > 0.050:
        raise ValueError("first proxy receipt differs from planned origin by >50 ms")
    for role, begin, finish in (
        ("prefill", "prefill_sent", "prefill_completed"),
        ("decode", "decode_sent", "decode_completed"),
    ):
        visible = []
        for rows in timelines.values():
            values = {row["event"]: int(row["wall_ns"]) / 1e9 for row in rows}
            visible.append(values[finish] - values[begin])
        if statistics.median(visible) < 2 * INTERVAL_S:
            raise ValueError(f"median {role} phase is below two meter intervals")
    validate_uncached_metrics(args.path, args.prefill_metrics, args.decode_metrics)
    validate_unsaturated(args.prefill_metrics, traffic_start=start, traffic_end=end)
    validate_unsaturated(args.decode_metrics, traffic_start=start, traffic_end=end)
    if args.expected >= 100:
        _validate_throughput_drift(
            args.prefill_metrics, traffic_start=start, traffic_end=end
        )
    validate_power(
        args.power,
        gpu_uuids=(args.prefill_uuid, args.decode_uuid),
        workload_start=start,
        workload_end=end,
    )
    _validate_power_state(args.power)
    validate_meter_phase(_power_timestamps(args.power), start)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--smoke", action="store_true")
    ports = commands.add_parser("ports")
    ports.add_argument("--start-offset", type=int, required=True)
    metadata = commands.add_parser("metadata")
    metadata.add_argument("path", type=Path)
    metadata.add_argument("--prefill-uuid", required=True)
    metadata.add_argument("--decode-uuid", required=True)
    metadata.add_argument("--image", required=True)
    metadata.add_argument("--smoke", action="store_true")
    origin = commands.add_parser("origin")
    origin.add_argument("power", type=Path)
    origin.add_argument("--lead-s", type=float, default=30.0)
    gate = commands.add_parser("check")
    gate.add_argument("path", type=Path)
    gate.add_argument("expected", type=int)
    gate.add_argument("--plan", type=Path, required=True)
    gate.add_argument("--events", type=Path, required=True)
    gate.add_argument("--event-start-line", type=int, default=0)
    gate.add_argument("--prefill-metrics", type=Path, required=True)
    gate.add_argument("--decode-metrics", type=Path, required=True)
    gate.add_argument("--power", type=Path, required=True)
    gate.add_argument("--prefill-uuid", required=True)
    gate.add_argument("--decode-uuid", required=True)
    gate.add_argument("--traffic-start", type=Path, required=True)
    gate.add_argument("--traffic-end", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        for row in cells(args.smoke):
            print("\t".join(map(str, row)))
    elif args.command == "ports":
        print(find_free_port_offset(args.start_offset))
    elif args.command == "metadata":
        value = run_metadata(
            args.prefill_uuid, args.decode_uuid, args.image, args.smoke
        )
        args.path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    elif args.command == "origin":
        print(
            phase_locked_origin(
                _first_power_timestamp(args.power), minimum_lead_s=args.lead_s
            )
        )
    else:
        check(args)


if __name__ == "__main__":
    main()
