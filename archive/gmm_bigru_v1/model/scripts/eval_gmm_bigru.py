#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from model.utils.runtime import configure_threading_env


def _generation_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode != "iid":
        raise argparse.ArgumentTypeError(
            f"--generation-mode must be 'iid'; AR(1) generation modes "
            f"('ar1', 'ar1_thresholded') were removed. Got: {value!r}"
        )
    return mode


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate continuous v1 GMM+BiGRU models on test traces.")
    parser.add_argument("--run-manifest", default="results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json")
    parser.add_argument("--experimental-manifest", default="results/experimental_continuous_v1/manifest.json")
    parser.add_argument(
        "--throughput-db",
        default="model/throughput_database.json",
        help="Retained for CLI compatibility; trained GMM artifacts use bound throughput.",
    )
    parser.add_argument(
        "--pair-manifest-csv",
        default="results/stage0/pair_manifest.csv",
        help="Retained for CLI compatibility; evaluation resolves request sources via lineage.",
    )
    parser.add_argument("--out-dir", default="results/continuous_v1_gmm_bigru/k10_f2/eval_metrics")
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--acf-max-lag", type=int, default=50)
    parser.add_argument(
        "--generation-mode",
        type=_generation_mode,
        default="iid",
        help="Trace generation mode; only 'iid' is supported (AR(1) generation removed).",
    )
    parser.add_argument("--decode-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    configure_threading_env()
    from model.pipeline.evaluation import evaluate_from_artifacts

    args = build_arg_parser().parse_args()
    run = evaluate_from_artifacts(
        run_manifest=args.run_manifest,
        experimental_manifest=args.experimental_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        out_dir=args.out_dir,
        config_ids=args.config_id,
        num_seeds=args.num_seeds,
        base_seed=args.base_seed,
        device=args.device,
        acf_max_lag=args.acf_max_lag,
        generation_mode=args.generation_mode,
        decode_mode=args.decode_mode,
        median_filter_window=args.median_filter_window,
        plots=(not args.no_plots),
    )

    print("[eval_gmm_bigru] Done")
    for k, v in run.get("summary", {}).items():
        print(f"  {k}: {v}")
    artifacts = run.get("artifacts", {})
    print(f"  per_seed_metrics : {artifacts.get('per_seed_metrics_csv', '')}")
    print(f"  per_trace_metrics: {artifacts.get('per_trace_metrics_csv', '')}")
    print(f"  config_summary   : {artifacts.get('config_summary_csv', '')}")
    print(f"  run_manifest     : {os.path.join(args.out_dir, 'run_manifest.json')}")
    summary = run.get("summary", {})
    if int(summary.get("num_evaluated_configs", 0)) == 0 or int(
        summary.get("num_failed_configs", 0)
    ) > 0:
        raise SystemExit(1)
    if args.config_id and int(summary.get("num_evaluated_configs", 0)) != len(args.config_id):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

