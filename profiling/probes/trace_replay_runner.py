"""Live runner for canonical exact-arrival/direct-token trace plans."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CLIENT = _HERE.parents[0] / "client"
sys.path.insert(0, str(_CLIENT))

import probe_runner  # noqa: E402
import trace_replay_driver  # noqa: E402


def round_release_epoch(start_epoch: float, ready_s: float,
                        closed_loop_ready: float) -> float:
    """A round needs both its exogenous release and prior tool work to finish."""
    return max(start_epoch + ready_s, closed_loop_ready)


async def drive_plan(
    http, base_url, model, plan, *, vocab_size, concurrency, prefix_cache,
    start_epoch, cache_block_tokens=16,
) -> tuple[list[dict], list[dict]]:
    """Honor release times before capacity acquisition and preserve session order."""
    semaphore = asyncio.Semaphore(concurrency)

    async def drive_session(session_id, rows):
        token_session = trace_replay_driver.TokenSession(
            session_id, vocab_size, plan.seed
        )
        records = []
        session_start = None
        closed_loop_ready = start_epoch
        for row in rows:
            planned_ready = start_epoch + row.ready_s
            release = round_release_epoch(
                start_epoch, row.ready_s, closed_loop_ready
            )
            await asyncio.sleep(max(0.0, release - time.time()))
            prompt = token_session.prompt(row)
            forced_output_token_id = token_session.forced_output_token(row)
            request_seed = token_session.request_seed(row)
            async with semaphore:
                record = await trace_replay_driver.send_round(
                    http, base_url, model, row, prompt, prefix_cache,
                    forced_output_token_id, request_seed, cache_block_tokens,
                )
            record["planned_ready_epoch"] = planned_ready
            record["arrival_delay_s"] = record["request_timestamp"] - planned_ready
            output_token_ids = record.pop("_output_token_ids")
            session_start = session_start or record["request_timestamp"]
            records.append(record)
            token_session.commit(row, prompt, output_token_ids)
            closed_loop_ready = time.time() + row.tool_wait_s
        window = {
            "session_id": session_id,
            "planned_start_epoch": start_epoch + rows[0].ready_s,
            "actual_start_epoch": session_start,
            "end_epoch": time.time(),
            "completed_turns": len(records),
        }
        return window, records

    results = await asyncio.gather(*(
        drive_session(session_id, rows)
        for session_id, rows in plan.by_session().items()
    ))
    windows = [window for window, _ in results]
    records = sorted(
        (record for _, rows in results for record in rows),
        key=lambda record: record["request_timestamp"],
    )
    return windows, records


def run(
    plan, *, model, hardware, tp, gpus_per_node, server_cfg, out_root,
    base_url="http://localhost:8000/v1", weight_footprint_bytes=None,
    embedding_bytes_per_param=None, fp8_flop_frac=None,
    dtype_hint=None, n_active_override=None, run_id=None, concurrency=64,
    prefix_cache=False, evidence_profile="core", power_profile="core",
    cache_block_tokens=16, pre_idle_s=0.0, validation_role="development",
):
    import aiohttp
    import transformers

    plan.validate(server_cfg.get("max_model_len"))
    run_manifest = probe_runner._client_mod("run_manifest")
    arch_extract = probe_runner._client_mod("arch_extract")
    run_id = run_id or f"{hardware.lower()}_trace_replay_tp{tp}_{int(time.time())}"
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    arch = arch_extract.extract_arch(
        arch_extract.load_config(model), dtype_hint=dtype_hint,
        weight_footprint_bytes=weight_footprint_bytes,
        embedding_bytes_per_param=embedding_bytes_per_param,
        fp8_flop_frac=fp8_flop_frac,
        n_active_override=n_active_override,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    vocab_size = int(tokenizer.vocab_size)
    concurrency = min(int(concurrency), len(plan.by_session()))
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if pre_idle_s < 0.0:
        raise ValueError("pre_idle_s must be non-negative")

    async def drive(start_epoch):
        async with aiohttp.ClientSession() as http:
            return await drive_plan(
                http, base_url, model, plan, vocab_size=vocab_size,
                concurrency=concurrency, prefix_cache=prefix_cache,
                start_epoch=start_epoch, cache_block_tokens=cache_block_tokens,
            )

    window_start = time.time()
    with probe_runner.logging_session(
        run_dir, base_url, evidence_profile=evidence_profile,
        gpus_per_node=gpus_per_node, power_profile=power_profile,
    ) as capture:
        idle_start = time.time()
        time.sleep(float(pre_idle_s))
        idle_end = time.time()
        replay_start = time.time()
        session_windows, records = asyncio.run(drive(replay_start))
        if plan.horizon_s is not None:
            time.sleep(max(0.0, replay_start + plan.horizon_s - time.time()))
    window_end = time.time()
    (run_dir / "requests.json").write_text(
        json.dumps(trace_replay_driver.build_requests_json(records))
    )
    manifest = run_manifest.build_manifest(
        run_id=run_id,
        probe={
            "type": "trace_replay",
            "source": plan.source,
            "source_revision": plan.revision,
            "trace_plan_sha256": plan.sha256,
            "seed": plan.seed,
            "planned_horizon_s": plan.horizon_s,
            "prefix_cache": bool(prefix_cache),
            "concurrency": concurrency,
            "cache_block_tokens": int(cache_block_tokens),
            "token_content": "deterministic_seeded_ids",
            "decode_constraint": (
                trace_replay_driver.DETERMINISTIC_DECODE_PROTOCOL
            ),
            "window": {"start_epoch": window_start, "end_epoch": window_end},
            "idle_window": {
                "start_epoch": idle_start,
                "end_epoch": idle_end,
                "requested_s": float(pre_idle_s),
            },
            "sessions": session_windows,
        },
        model=model, arch=arch, hardware=hardware, tp=tp,
        gpus_per_node=gpus_per_node,
        server=dict(server_cfg, enable_prefix_caching=bool(prefix_cache)),
        versions=run_manifest.collect_versions(), clock=capture["clock"],
        instrumentation=capture["instrumentation"],
        evidence_profile=evidence_profile, validation_role=validation_role,
    )
    run_manifest.write_manifest(str(run_dir / "manifest.json"), manifest)
    return run_dir
