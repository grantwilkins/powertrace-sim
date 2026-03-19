#!/usr/bin/env python3
from __future__ import annotations

import argparse

from model.utils.runtime import configure_threading_env


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate traces from continuous v1 GMM+BiGRU artifacts.")
    parser.add_argument("--run-manifest", default="results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json")
    parser.add_argument("--throughput-db", default="model/config/throughput_database.json")
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--requests-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--decode-mode", choices=["stochastic", "argmax"], default="stochastic")
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gmm-params", default=None)
    parser.add_argument("--norm-params", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--feature-set", choices=["f2"], default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    return parser


def main() -> None:
    configure_threading_env()
    from model.pipeline.inference import run_inference_from_artifacts

    args = build_arg_parser().parse_args()
    result = run_inference_from_artifacts(
        config_id=args.config_id,
        requests_json=args.requests_json,
        out_csv=args.out_csv,
        run_manifest=args.run_manifest,
        throughput_db=args.throughput_db,
        device=args.device,
        seed=args.seed,
        dt=args.dt,
        T=args.T,
        p0=args.p0,
        decode_mode=args.decode_mode,
        median_filter_window=args.median_filter_window,
        checkpoint=args.checkpoint,
        gmm_params=args.gmm_params,
        norm_params=args.norm_params,
        k=args.k,
        feature_set=args.feature_set,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    print("[infer_gmm_bigru] Inference complete")
    print(f"  config_id: {result['config_id']}")
    print(f"  dt: {result['dt']}")
    print(f"  T: {result['T']}")
    print(f"  decode_mode: {result['decode_mode']}")
    print(f"  out_csv: {result['out_csv']}")


if __name__ == "__main__":
    main()
