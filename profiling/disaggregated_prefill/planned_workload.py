"""Create and execute immutable disaggregated-inference request plans."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "powertrace-fixed-request-plan-v1"
CLIENT_DIR = Path(__file__).resolve().parents[1] / "client"


def plan_sha256(plan: dict[str, Any]) -> str:
    payload = {key: value for key, value in plan.items() if key != "sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_plan(
    requests: list[dict[str, Any]],
    *,
    interval_s: float,
    model: str,
    seed: int,
) -> dict[str, Any]:
    if interval_s <= 0:
        raise ValueError("interval must be positive")
    rows = [
        {
            "prompt": request["prompt"],
            "prompt_len": int(request["prompt_len"]),
            "output_len": int(request["output_len"]),
            "offset_s": index * float(interval_s),
        }
        for index, request in enumerate(requests)
    ]
    plan = {
        "schema": SCHEMA,
        "model": model,
        "seed": int(seed),
        "interval_s": float(interval_s),
        "requests": rows,
    }
    plan["sha256"] = plan_sha256(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> str:
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported request-plan schema")
    rows = plan.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("request plan must contain requests")
    offsets = []
    for row in rows:
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            raise ValueError("request plan has an empty prompt")
        if int(row.get("prompt_len", 0)) <= 0 or int(row.get("output_len", 0)) <= 0:
            raise ValueError("request plan has nonpositive token lengths")
        offsets.append(float(row.get("offset_s", -1)))
    if offsets[0] != 0.0 or any(
        right <= left for left, right in zip(offsets, offsets[1:])
    ):
        raise ValueError("request-plan offsets must start at zero and increase")
    digest = plan_sha256(plan)
    if plan.get("sha256") != digest:
        raise ValueError("request-plan hash does not match its contents")
    return digest


def deadline_delay(
    start_monotonic: float, offset_s: float, *, now_monotonic: float | None = None
) -> float:
    now = time.perf_counter() if now_monotonic is None else now_monotonic
    return max(0.0, start_monotonic + offset_s - now)


def _client_imports():
    sys.path.insert(0, str(CLIENT_DIR))
    from benchmark_dataset import RandomDataset, SampleRequest
    from benchmark_serving import benchmark, get_tokenizer

    return RandomDataset, SampleRequest, benchmark, get_tokenizer


def generate(args: argparse.Namespace) -> None:
    import numpy as np

    RandomDataset, _, _, get_tokenizer = _client_imports()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = get_tokenizer(args.model)
    sampled = RandomDataset(random_seed=args.seed).sample(
        tokenizer=tokenizer,
        num_requests=args.num_requests,
        prefix_len=0,
        input_len=8192,
        output_len=256,
        range_ratio=0.25,
    )
    requests = [
        {
            "prompt": row.prompt,
            "prompt_len": row.prompt_len,
            "output_len": row.expected_output_len,
        }
        for row in sampled
    ]
    plan = make_plan(
        requests,
        interval_s=1.0 / args.request_rate,
        model=args.model,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(plan, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
    _, SampleRequest, benchmark, get_tokenizer = _client_imports()
    plan = json.loads(args.request_plan.read_text())
    digest = validate_plan(plan)
    traffic_start = float(args.traffic_start.read_text())
    requests = [
        SampleRequest(
            prompt=row["prompt"],
            prompt_len=int(row["prompt_len"]),
            expected_output_len=int(row["output_len"]),
        )
        for row in plan["requests"]
    ]
    tokenizer = get_tokenizer(plan["model"])
    result = asyncio.run(
        benchmark(
            backend="vllm",
            api_url=f"{args.base_url}/v1/completions",
            base_url=args.base_url,
            model_id=plan["model"],
            model_name=None,
            tokenizer=tokenizer,
            input_requests=requests,
            logprobs=None,
            request_rate=1.0 / float(plan["interval_s"]),
            burstiness=1.0,
            disable_tqdm=True,
            profile=False,
            selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"],
            selected_percentiles=[50.0, 90.0, 99.0],
            ignore_eos=True,
            goodput_config_dict={},
            max_concurrency=None,
            lora_modules=None,
            extra_body={"temperature": 0.0},
            skip_test_prompt=True,
            request_offsets_s=[
                float(row["offset_s"]) for row in plan["requests"]
            ],
            traffic_start_epoch_s=traffic_start,
        )
    )
    result.update(
        {
            "model_id": plan["model"],
            "num_prompts": len(requests),
            "request_rate": 1.0 / float(plan["interval_s"]),
            "request_plan_sha256": digest,
            "planned_offsets_s": [
                float(row["offset_s"]) for row in plan["requests"]
            ],
            "traffic_start_epoch_s": traffic_start,
        }
    )
    args.output.write_text(json.dumps(result) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("generate")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--request-rate", type=float, required=True)
    create.add_argument("--num-requests", type=int, required=True)
    create.add_argument("--seed", type=int, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--request-plan", type=Path, required=True)
    execute.add_argument("--traffic-start", type=Path, required=True)
    execute.add_argument("--base-url", required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args) if args.command == "generate" else run(args)


if __name__ == "__main__":
    main()
