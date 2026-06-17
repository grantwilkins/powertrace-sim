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


_HBM_DEFAULT_GIB = {"A100": 80, "H100": 80}  # fallback if nvidia-smi is unavailable


def gpu_hbm_bytes(hardware: str) -> int:
    """Total HBM per GPU in bytes, from nvidia-smi (fallback: hardware default)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        return int(out.stdout.split("\n")[0]) * 1024 * 1024  # MiB -> bytes
    except Exception:
        return _HBM_DEFAULT_GIB.get(hardware, 80) * 1024**3


def auto_concurrency(*, arch, avg_context, tp, hbm_bytes_per_gpu, max_num_seqs,
                     kv_cache_dtype="auto", target_fill=0.7, overhead_frac=0.1) -> int:
    """Sessions that fit the KV-cache budget at a target fill (CAMPAIGN.md §4).

    Decode batch is bounded by KV cache, not parameter count: the budget is the
    aggregate HBM left after weights, divided by the per-token KV cost times the
    typical context. TP adds budget (more GPUs); weights and context spend it. All
    inputs come from ``manifest.arch`` + the launched HBM/TP, so it's portable across
    model sizes. Clamped to ``[1, max_num_seqs]``.
    """
    kv_dtype_bytes = 1 if "fp8" in str(kv_cache_dtype).lower() else 2
    kv_per_token = 2 * arch["n_layers"] * arch["n_kv"] * arch["head_dim"] * kv_dtype_bytes
    total_hbm = tp * hbm_bytes_per_gpu
    kv_budget = max(total_hbm * (1 - overhead_frac) - arch["w_bytes"], 0.0) * target_fill
    fit = int(kv_budget // max(kv_per_token * avg_context, 1))
    return max(1, min(fit, max_num_seqs))


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
        dtype_hint=None, n_active_override=None, run_id=None, max_concurrency=None):
    """Execute an AgenticPlan and write the bundle. Returns the run directory.

    Sessions run concurrently — turns *within* a session stay ordered because its
    context grows turn by turn. ``max_concurrency`` caps how many are in flight:
    ``"auto"`` sizes it to the KV-cache budget (default), an int fixes it, and
    ``None`` runs all sessions at once.
    """
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
    if max_concurrency == "auto":
        ctx = [c for s in plan.sessions for c in s.context_lengths()]
        avg_context = sum(ctx) / len(ctx) if ctx else 1
        concurrency = auto_concurrency(
            arch=arch, avg_context=avg_context, tp=tp,
            hbm_bytes_per_gpu=gpu_hbm_bytes(hardware),
            max_num_seqs=server_cfg["max_num_seqs"],
            kv_cache_dtype=server_cfg.get("kv_cache_dtype", "auto"))
        print(f"[agentic] auto concurrency={concurrency} "
              f"(avg_context={avg_context:.0f} tok, tp={tp})")
    else:
        concurrency = max_concurrency or len(plan.sessions)
    concurrency = min(concurrency, len(plan.sessions))

    async def _drive():
        sem = asyncio.Semaphore(concurrency)

        async def _one(http, sess):
            async with sem:
                t0 = time.time()
                recs = await session_driver.send_session(
                    http, base_url, model, sess, plan.prefix_cache, tokenizer)
                return build_session_window(sess, t0, time.time(), len(recs)), recs

        async with aiohttp.ClientSession() as http:
            return await asyncio.gather(*(_one(http, s) for s in plan.sessions))

    with probe_runner.logging_session(run_dir, base_url) as clock:
        results = asyncio.run(_drive())
    window_end = time.time()

    session_windows = [w for w, _ in results]
    all_records = [r for _, recs in results for r in recs]

    (run_dir / "requests.json").write_text(
        json.dumps(session_driver.build_requests_json(all_records)))

    server = dict(server_cfg, enable_prefix_caching=bool(plan.prefix_cache))
    manifest = run_manifest.build_manifest(
        run_id=run_id,
        probe={"type": "agentic", "label": plan.label,
               "prefix_cache": bool(plan.prefix_cache),
               "concurrency": concurrency,
               "window": {"start_epoch": window_start, "end_epoch": window_end},
               "sessions": session_windows},
        model=model, arch=arch, hardware=hardware, tp=tp,
        gpus_per_node=gpus_per_node, server=server,
        versions=run_manifest.collect_versions(), clock=clock)
    run_manifest.write_manifest(str(run_dir / "manifest.json"), manifest)
    return run_dir
