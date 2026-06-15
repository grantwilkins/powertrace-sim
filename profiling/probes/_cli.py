"""Shared CLI plumbing for the probe drivers (thin live layer).

Common arguments + the ``execute`` entry that hands a built ProbeSchedule to the
runner. Heavy imports stay inside ``execute`` so the wrappers import cleanly.
"""

from __future__ import annotations

import argparse


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True, choices=["A100", "H100"])
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--out-root", default="data/runs")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--hold-s", type=float, default=45.0)
    p.add_argument("--dtype-hint", default=None)
    p.add_argument("--weight-footprint-bytes", type=float, default=None)
    p.add_argument("--n-active-override", type=float, default=None,
                   help="vendor-stated active param count (MoE; overrides analytic)")
    # server knobs recorded into every manifest
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--enable-chunked-prefill", action="store_true", default=True)
    p.add_argument("--enable-prefix-caching", action="store_true", default=False)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--max-model-len", type=int, default=131072)
    return p


def server_cfg(args) -> dict:
    return {
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": args.enable_prefix_caching,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_model_len": args.max_model_len,
    }


def execute(schedule, args):
    """Run the schedule against the live endpoint and write the bundle."""
    import probe_runner

    return probe_runner.run(
        schedule,
        model=args.model,
        hardware=args.hardware,
        tp=args.tp,
        gpus_per_node=args.gpus_per_node,
        server_cfg=server_cfg(args),
        out_root=args.out_root,
        base_url=args.base_url,
        weight_footprint_bytes=args.weight_footprint_bytes,
        dtype_hint=args.dtype_hint,
        n_active_override=args.n_active_override,
    )
