"""Normalize a released trace into the exact replay-plan JSON.

Examples:
  python profiling/agentic_traces/build_trace_plan.py tracelab.csv plan.json \
    --format tracelab --revision <dataset-commit>
"""

from __future__ import annotations

import argparse

from trace_plan import (  # noqa: E402
    assign_poisson_session_arrivals, load_burstgpt_csv, load_canonical_csv,
    load_tracelab_csv, load_tracelab_jsonl, select_context_bands,
    select_densest_arrival_window, select_sessions, write_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_json")
    parser.add_argument(
        "--format",
        choices=("canonical", "tracelab", "tracelab-jsonl", "burstgpt"),
        required=True,
    )
    parser.add_argument("--source")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, required=True)
    parser.add_argument("--max-rounds-per-session", type=int, default=16)
    parser.add_argument("--min-rounds-per-session", type=int, default=1)
    parser.add_argument("--min-max-context", type=int, default=0)
    parser.add_argument("--max-max-context", type=int)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--window-duration-s", type=float, default=600.0)
    parser.add_argument(
        "--context-band", action="append", default=[],
        metavar="MIN:MAX:COUNT",
        help="repeat for deterministic stratification; MAX is exclusive",
    )
    args = parser.parse_args()
    if args.format == "tracelab":
        plan = load_tracelab_csv(
            args.input_csv, revision=args.revision, seed=args.seed
        )
    elif args.format == "tracelab-jsonl":
        plan = load_tracelab_jsonl(
            args.input_csv, revision=args.revision, seed=args.seed
        )
    elif args.format == "burstgpt":
        plan = load_burstgpt_csv(
            args.input_csv, revision=args.revision, seed=args.seed
        )
    else:
        if not args.source:
            parser.error("--source is required for --format canonical")
        plan = load_canonical_csv(
            args.input_csv, source=args.source, revision=args.revision, seed=args.seed
        )
    if args.format == "burstgpt":
        plan = select_densest_arrival_window(
            plan, duration_s=args.window_duration_s,
            max_requests=args.max_sessions,
        )
    elif args.context_band:
        bands = [tuple(int(value) for value in item.split(":"))
                 for item in args.context_band]
        if any(len(band) != 3 for band in bands):
            parser.error("--context-band must be MIN:MAX:COUNT")
        if sum(band[2] for band in bands) > args.max_sessions:
            parser.error("context-band counts exceed --max-sessions")
        plan = select_context_bands(
            plan, bands, max_rounds_per_session=args.max_rounds_per_session,
            min_rounds_per_session=args.min_rounds_per_session,
        )
    else:
        plan = select_sessions(
            plan, max_sessions=args.max_sessions,
            max_rounds_per_session=args.max_rounds_per_session,
            min_max_context=args.min_max_context,
            max_max_context=args.max_max_context,
            min_rounds_per_session=args.min_rounds_per_session,
        )
    if args.format == "tracelab-jsonl":
        plan = assign_poisson_session_arrivals(
            plan, rate_rps=args.arrival_rate, seed=args.seed + 1
        )
    write_plan(plan, args.output_json)
    print(f"{args.output_json} {plan.sha256}")


if __name__ == "__main__":
    main()
