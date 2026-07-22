#!/usr/bin/env python3
from __future__ import annotations

import argparse

from model.release import DEFAULT_ARTIFACT
from model.simulation import simulate_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict request timing and deterministic GPU power."
    )
    parser.add_argument("--requests", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-unsupported", action="store_true")
    args = parser.parse_args()
    outputs = simulate_file(
        args.requests, deployment=args.deployment, out_dir=args.out_dir,
        artifact_path=args.artifact, seed=args.seed,
        allow_unsupported=args.allow_unsupported,
    )
    print(f"[infer] wrote {outputs['power']}, {outputs['requests']}, and {outputs['manifest']}")


if __name__ == "__main__":
    main()

