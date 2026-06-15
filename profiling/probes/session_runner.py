"""Agentic session runner: drive multi-turn sessions, emit a §2 bundle (§5-D).

Reuses ``probe_runner.logging_session`` (continuous power + /metrics logs) and the
manifest emitter, so an agentic bundle is interchangeable with a probe bundle —
only ``probe.type`` ("agentic"), the per-session windows, and the richer
``requests.json`` (per-turn extensions) differ. The session HTTP sending is the
live layer; manifest/window assembly is pure and testable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CLIENT = _HERE.parents[0] / "client"
sys.path.insert(0, str(_CLIENT))

import probe_runner  # noqa: E402  (logging_session, _client_mod)
import session_driver  # noqa: E402


def build_session_window(session, t_start_epoch, t_end_epoch, n_records) -> dict:
    """Manifest entry locating one conversation in absolute time (pure)."""
    return {
        "session_id": session.session_id,
        "n_turns": len(session.turns),
        "prefix_tokens": session.prefix_tokens,
        "t_start_epoch": float(t_start_epoch),
        "t_end_epoch": float(t_end_epoch),
        "completed_turns": int(n_records),
        "context_lengths": session.context_lengths(),
    }


def run(plan, *, model, hardware, tp, gpus_per_node, server_cfg, out_root,
        base_url="http://localhost:8000/v1", weight_footprint_bytes=None,
        dtype_hint=None, n_active_override=None, run_id=None):
    """Execute an AgenticPlan and write the bundle. Returns the run directory."""
    run_manifest = probe_runner._client_mod("run_manifest")
    arch_extract = probe_runner._client_mod("arch_extract")
    import aiohttp
    import asyncio
    import transformers

    run_id = run_id or f"{hardware.lower()}_agentic_tp{tp}_{int(time.time())}"
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    arch = arch_extract.extract_arch(
        arch_extract.load_config(model), dtype_hint=dtype_hint,
        weight_footprint_bytes=weight_footprint_bytes,
        n_active_override=n_active_override)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    window_start = time.time()
    all_records, session_windows = [], []

    async def _drive():
        async with aiohttp.ClientSession() as http:
            for sess in plan.sessions:
                t0 = time.time()
                recs = await session_driver.send_session(
                    http, base_url, model, sess, plan.prefix_cache, tokenizer)
                t1 = time.time()
                all_records.extend(recs)
                session_windows.append(build_session_window(sess, t0, t1, len(recs)))

    with probe_runner.logging_session(run_dir, base_url) as clock:
        asyncio.run(_drive())
    window_end = time.time()

    (run_dir / "requests.json").write_text(
        json.dumps(session_driver.build_requests_json(all_records)))

    server = dict(server_cfg, enable_prefix_caching=bool(plan.prefix_cache))
    manifest = run_manifest.build_manifest(
        run_id=run_id,
        probe={"type": "agentic", "label": plan.label,
               "prefix_cache": bool(plan.prefix_cache),
               "window": {"start_epoch": window_start, "end_epoch": window_end},
               "sessions": session_windows},
        model=model, arch=arch, hardware=hardware, tp=tp,
        gpus_per_node=gpus_per_node, server=server,
        versions=run_manifest.collect_versions(), clock=clock)
    run_manifest.write_manifest(str(run_dir / "manifest.json"), manifest)
    return run_dir
