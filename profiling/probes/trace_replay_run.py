"""Replay a canonical trace plan with exact arrivals, prefixes, and token counts."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "agentic_traces"))

from _cli import base_parser, server_cfg  # noqa: E402
from trace_plan import load_plan  # noqa: E402
import trace_replay_runner  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--trace-plan", required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--cache-block-tokens", type=int, default=16)
    args = parser.parse_args()
    plan = load_plan(args.trace_plan, args.max_model_len)
    print(trace_replay_runner.run(
        plan, model=args.model, hardware=args.hardware, tp=args.tp,
        gpus_per_node=args.gpus_per_node, server_cfg=server_cfg(args),
        out_root=args.out_root, base_url=args.base_url,
        weight_footprint_bytes=args.weight_footprint_bytes,
        dtype_hint=args.dtype_hint, n_active_override=args.n_active_override,
        concurrency=args.concurrency, prefix_cache=args.prefix_cache,
        evidence_profile=args.evidence_profile,
        power_profile=args.power_profile,
        cache_block_tokens=args.cache_block_tokens,
    ))


if __name__ == "__main__":
    main()
