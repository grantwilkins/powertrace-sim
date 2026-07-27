"""Build the four additional BurstGPT plans for the transfer interval."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from trace_plan import load_plan, load_stratified_burstgpt_csv, write_plan


WINDOW_RE = re.compile(r";window:([0-9.]+)-([0-9.]+);")


def existing_windows(
    paths: list[Path], min_requests: int, revision: str
) -> tuple[tuple[float, float], ...]:
    windows = []
    for path in paths:
        plan = load_plan(path)
        if not plan.revision.startswith(f"{revision};"):
            raise ValueError(f"{path} does not match source revision {revision}")
        match = WINDOW_RE.search(plan.revision)
        if match and len(plan.rounds) >= min_requests:
            windows.append((float(match.group(1)), float(match.group(2))))
    return tuple(windows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("burstgpt_csv")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out-dir", default="data/trace_plans")
    parser.add_argument("--duration-s", type=float, default=900.0)
    parser.add_argument("--min-requests", type=int, default=100)
    args = parser.parse_args()

    prior = existing_windows(
        sorted(Path(args.out_dir).glob("burstgpt_15min_fano*.json")),
        args.min_requests,
        args.revision,
    )
    if len(prior) != 2:
        raise ValueError(f"expected two reusable BurstGPT windows, found {len(prior)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        plan = load_stratified_burstgpt_csv(
            args.burstgpt_csv,
            revision=args.revision,
            duration_s=args.duration_s,
            window_index=index,
            window_count=4,
            min_requests=args.min_requests,
            excluded_windows=prior,
        )
        output = out_dir / f"burstgpt_transfer_ci_{index}.json"
        write_plan(plan, output)
        print(f"{output} {len(plan.rounds)} requests {plan.sha256}")


if __name__ == "__main__":
    main()
