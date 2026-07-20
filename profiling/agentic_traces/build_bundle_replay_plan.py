"""Build an exact release/length replay plan from a canonical bundle."""

from __future__ import annotations

import argparse

from trace_plan import load_bundle_requests_json, write_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests_json")
    parser.add_argument("output_json")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    plan = load_bundle_requests_json(
        args.requests_json, time_scale=args.time_scale, seed=args.seed
    )
    write_plan(plan, args.output_json)
    print(f"{args.output_json} {plan.sha256}")


if __name__ == "__main__":
    main()
