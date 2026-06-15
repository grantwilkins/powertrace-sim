"""CLI: agentic multi-turn session workload (Tier-3, §5-D).

Run with --prefix-cache and without (and a matching server --enable-prefix-caching)
to cover both production regimes.
"""

from _cli import base_parser, server_cfg
from agentic import build_synthetic_sessions
import session_runner


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-sessions", type=int, default=8)
    p.add_argument("--min-turns", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--user-tokens-mean", type=int, default=256)
    p.add_argument("--assistant-tokens-mean", type=int, default=256)
    p.add_argument("--prefix-tokens", type=int, default=512)
    p.add_argument("--gap-mean-s", type=float, default=3.0)
    p.add_argument("--gap-sigma", type=float, default=0.8)
    p.add_argument("--prefix-cache", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    plan = build_synthetic_sessions(
        n_sessions=args.n_sessions, min_turns=args.min_turns, max_turns=args.max_turns,
        user_tokens_mean=args.user_tokens_mean,
        assistant_tokens_mean=args.assistant_tokens_mean,
        prefix_tokens=args.prefix_tokens, gap_mean_s=args.gap_mean_s,
        gap_sigma=args.gap_sigma, prefix_cache=args.prefix_cache, seed=args.seed)

    print(session_runner.run(
        plan, model=args.model, hardware=args.hardware, tp=args.tp,
        gpus_per_node=args.gpus_per_node, server_cfg=server_cfg(args),
        out_root=args.out_root, base_url=args.base_url,
        weight_footprint_bytes=args.weight_footprint_bytes,
        dtype_hint=args.dtype_hint, n_active_override=args.n_active_override))


if __name__ == "__main__":
    main()
