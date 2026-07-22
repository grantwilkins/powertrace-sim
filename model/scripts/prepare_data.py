#!/usr/bin/env python3
from __future__ import annotations

import argparse

from model.preparation import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and hash-bind the selected-model prepared dataset."
    )
    parser.add_argument("--timing-dataset", default="timing-test/timing_dataset.npz")
    parser.add_argument("--run-index", default="timing-test/timing_dataset.runs.json")
    parser.add_argument("--split-manifest", default="timing-test/split_manifest_fp8.json")
    parser.add_argument("--base-split-manifest", default="timing-test/split_manifest.json")
    parser.add_argument("--probe-calibration", default="timing-test/probe_calibration.json")
    parser.add_argument(
        "--power-cache",
        default="power-test/sim_ledger_power_uniform_current_250ms.npz",
    )
    parser.add_argument(
        "--out-manifest", default="results/clean_model/prepared_dataset.json"
    )
    args = parser.parse_args()
    result = prepare_dataset(
        timing_dataset=args.timing_dataset,
        run_index=args.run_index,
        split_manifest=args.split_manifest,
        base_split_manifest=args.base_split_manifest,
        probe_calibration=args.probe_calibration,
        power_cache=args.power_cache,
        out_manifest=args.out_manifest,
    )
    print(
        f"[prepare_data] bound {result['run_count']} runs and "
        f"{result['request_count']} requests in {args.out_manifest}"
    )


if __name__ == "__main__":
    main()
