#!/usr/bin/env python3
from __future__ import annotations

import argparse

from model.pipeline.physics_inference import run_physics_inference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic physics power from an arrivals schedule."
    )
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--requests-json", required=True)
    parser.add_argument(
        "--physics-artifact",
        default="feature-test/results/physics_artifact_v1.json",
    )
    parser.add_argument("--throughput-db", default="model/throughput_database.json")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--T", type=int, default=None)
    args = parser.parse_args()
    result = run_physics_inference(
        config_id=args.config_id,
        requests_json=args.requests_json,
        physics_artifact=args.physics_artifact,
        throughput_db=args.throughput_db,
        out_csv=args.out_csv,
        dt=args.dt,
        T=args.T,
    )
    print(f"[infer_physics] wrote {result['out_csv']} ({result['T']} bins)")


if __name__ == "__main__":
    main()
