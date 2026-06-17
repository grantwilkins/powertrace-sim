"""CLI: agentic multi-turn session workload (Tier-3, §5-D).

Two modes:
  --replay   real SWE-agent traces (real text + per-tool gap model)   [Gap 1]
  (default)  synthetic sessions (lognormal token/gap draws)

Run with and without --prefix-cache (the orchestrator relaunches the server with a
matching --enable-prefix-caching) to cover both production regimes.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                          # _cli, agentic, session_runner
sys.path.insert(0, str(_HERE.parents[0] / "client"))   # loggers / manifest (via runner)
sys.path.insert(0, str(_HERE.parents[0] / "agentic_traces"))  # replay_loader

from _cli import base_parser, server_cfg  # noqa: E402
import session_runner  # noqa: E402


def _build_plan(args):
    if args.replay:
        import transformers
        from replay_loader import build_replay_plan
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        return build_replay_plan(
            corpus=args.corpus, n_sessions=args.n_sessions, seed=args.seed,
            tokenizer=tokenizer, prefix_cache=args.prefix_cache,
            gap_params=args.gap_params, max_model_len=args.max_model_len)

    from agentic import build_synthetic_sessions
    return build_synthetic_sessions(
        n_sessions=args.n_sessions, min_turns=args.min_turns, max_turns=args.max_turns,
        user_tokens_mean=args.user_tokens_mean,
        assistant_tokens_mean=args.assistant_tokens_mean,
        prefix_tokens=args.prefix_tokens, gap_mean_s=args.gap_mean_s,
        gap_sigma=args.gap_sigma, prefix_cache=args.prefix_cache, seed=args.seed)


def main():
    p = base_parser(__doc__)
    p.add_argument("--prefix-cache", action="store_true", default=False)
    p.add_argument("--n-sessions", type=int, default=8)
    p.add_argument("--concurrency", default="auto",
                   help="'auto' (KV-budget), an int (fixed), or 0 (all sessions)")
    p.add_argument("--seed", type=int, default=0)
    # replay mode
    p.add_argument("--replay", action="store_true", default=False)
    p.add_argument("--corpus", default="swe_smith")
    p.add_argument("--gap-params",
                   default=str(_HERE.parents[0] / "agentic_traces" / "gap_params.json"))
    # synthetic mode
    p.add_argument("--min-turns", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--user-tokens-mean", type=int, default=256)
    p.add_argument("--assistant-tokens-mean", type=int, default=256)
    p.add_argument("--prefix-tokens", type=int, default=512)
    p.add_argument("--gap-mean-s", type=float, default=3.0)
    p.add_argument("--gap-sigma", type=float, default=0.8)
    args = p.parse_args()

    print(session_runner.run(
        _build_plan(args), model=args.model, hardware=args.hardware, tp=args.tp,
        gpus_per_node=args.gpus_per_node, server_cfg=server_cfg(args),
        out_root=args.out_root, base_url=args.base_url,
        weight_footprint_bytes=args.weight_footprint_bytes,
        dtype_hint=args.dtype_hint, n_active_override=args.n_active_override,
        max_concurrency=_concurrency(args.concurrency)))


def _concurrency(value):
    """'auto' -> auto; '0' -> None (all sessions); 'N' -> N."""
    return value if value == "auto" else (int(value) or None)


if __name__ == "__main__":
    main()
