#!/usr/bin/env python3
from __future__ import annotations

import argparse

from model.evaluation import evaluate_power_csv, write_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Score aligned PowerTrace CSVs.")
    parser.add_argument("--measured", required=True)
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--measured-column", default="node_gpu_power_w")
    parser.add_argument("--predicted-column", default="node_gpu_power_w")
    args = parser.parse_args()
    metrics = evaluate_power_csv(
        args.measured, args.predicted,
        measured_column=args.measured_column,
        predicted_column=args.predicted_column,
    )
    write_evaluation(metrics, args.out)
    print(f"[evaluate] wrote {args.out}")


if __name__ == "__main__":
    main()

