"""Frozen cell plan and result gate for the GPT-OSS disaggregated campaign."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

RATES = (0.25, 2.0, 4.0)
REPEATS = 3
DURATION_S = 600


def run_metadata(
    prefill_uuid: str, decode_uuid: str, image: str, dataset: str
) -> dict:
    if not prefill_uuid or not decode_uuid or prefill_uuid == decode_uuid:
        raise ValueError("prefill and decode require distinct GPU UUIDs")
    local_now = datetime.now().astimezone()
    return {
        "model": "openai/gpt-oss-20b",
        "hardware": "A100-80GB",
        "deployment": "disaggregated_prefill",
        "roles": {
            "prefill": {"tp": 1, "gpu_uuid": prefill_uuid},
            "decode": {"tp": 1, "gpu_uuid": decode_uuid},
        },
        "server": {
            "max_model_len": 131072,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.9,
            "enable_chunked_prefill": True,
            "async_scheduling": True,
            "kv_connector": "NixlConnector",
            "kv_load_failure_policy": "fail",
        },
        "workload": {
            "dataset": "sharegpt",
            "dataset_path": dataset,
            "arrival_process": "poisson",
            "seed": 0,
            "nominal_duration_s": DURATION_S,
            "rates": list(RATES),
            "repeats": REPEATS,
        },
        "container_image": image,
        "clock": {
            "local_utc_offset_s": float(local_now.utcoffset().total_seconds()),
            "power_timestamp_basis": "local_wall_time",
            "engine_timestamp_basis": "unix_epoch",
            "request_timestamp_basis": "unix_epoch",
            "proxy_timestamp_basis": "unix_epoch_nanoseconds",
        },
    }


def cells(smoke: bool = False) -> list[tuple[float, int, int]]:
    rates = (2.0,) if smoke else RATES
    repeats = 1 if smoke else REPEATS
    return [
        (rate, repeat, round(rate * DURATION_S))
        for rate in rates
        for repeat in range(repeats)
    ]


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


def validate_events(path: Path, start_line: int, request_ids: list[str]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--smoke", action="store_true")
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
    metadata.add_argument("--dataset", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        for rate, repeat, prompts in cells(args.smoke):
            print(f"{rate:g}\t{repeat}\t{prompts}")
    elif args.command == "check":
        request_ids = validate_result(args.path, args.expected)
        if args.events:
            validate_events(args.events, args.event_start_line, request_ids)
    else:
        value = run_metadata(
            args.prefill_uuid, args.decode_uuid, args.image, args.dataset
        )
        args.path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
