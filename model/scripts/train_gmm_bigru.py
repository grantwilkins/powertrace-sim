#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from model.pipeline.gmm_fitting import _parse_k_candidates
from model.utils.runtime import configure_threading_env


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train continuous v1 GMM+BiGRU models from experimental manifest."
    )
    parser.add_argument(
        "--manifest", default="results/experimental_continuous_v1/manifest.json"
    )
    parser.add_argument("--out-root", default="results/continuous_v1_gmm_bigru")
    parser.add_argument("--config-id", action="append", default=[])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--feature-set", choices=["f2"], default="f2")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--scheduler-patience", type=int, default=20)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--bic-candidates", default="6,8,10,12,14,16,18,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Automatically select K per config based on BIC minimum (up to --max-k).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=20,
        help="Maximum K to consider when using --auto-k (default: 20).",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Disable phased default scoping and target all written configs.",
    )
    return parser


def main() -> None:
    configure_threading_env()
    from model.pipeline.training import run_training_from_manifest

    args = build_arg_parser().parse_args()
    run_manifest = run_training_from_manifest(
        manifest_path=args.manifest,
        out_root=args.out_root,
        config_ids=args.config_id,
        k=args.k,
        feature_set=args.feature_set,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        bic_candidates=_parse_k_candidates(args.bic_candidates),
        seed=args.seed,
        device=args.device,
        force_all_configs=bool(args.all_configs),
        auto_k=bool(args.auto_k),
        max_k=args.max_k,
    )
    print("[train_gmm_bigru] Summary:")
    for k, v in run_manifest.get("summary", {}).items():
        print(f"  {k}: {v}")
    print(
        f"  run_manifest: {os.path.join(run_manifest['defaults']['out_dir'], 'run_manifest.json')}"
    )
    print(
        f"  run_summary : {os.path.join(run_manifest['defaults']['out_dir'], 'run_summary.csv')}"
    )


if __name__ == "__main__":
    main()
