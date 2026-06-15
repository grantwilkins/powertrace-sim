"""Sweep request rate and plot steady-state average power for one config.

For each rps point, generates Poisson request schedules (token lengths sampled
from the config's measured benchmark runs), runs the trained GMM-BiGRU power
model, and averages the generated power over a steady-state window. Produces a
summary CSV and an rps-vs-average-power figure with a mean +/- std band.

Example:
    uv run -m scripts.eval.rps_power_sweep \
        --config-id llama-3-70b_H100_tp8 \
        --rps 0.25 0.5 1 2 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.pipeline.inference import run_inference_from_artifacts
from model.utils.io import resolve_existing_path
from scripts.eval.run_baselines_facility import (
    _extract_token_pools,
    _generate_poisson_requests,
)

DEFAULT_RPS = [
    0.0625, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
    6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0,
]


def _config_json_paths(pair_manifest_csv: str, config_id: str) -> List[str]:
    """Collect benchmark JSON paths matching the config's model/hardware/TP."""
    model_name, hardware, tp_part = config_id.rsplit("_", 2)
    tp = tp_part.removeprefix("tp")
    base_dir = str(Path(pair_manifest_csv).resolve().parent)
    paths: List[str] = []
    with open(pair_manifest_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if (
                str(row.get("status", "")).strip() != "matched"
                or str(row.get("model_name", "")).strip() != model_name
                or str(row.get("hardware", "")).strip() != hardware
                or str(row.get("tensor_parallelism", "")).strip() != tp
            ):
                continue
            json_path = resolve_existing_path(str(row.get("json_path", "")).strip(), base_dir)
            if json_path is not None:
                paths.append(json_path)
    return paths


def run_sweep(
    *,
    config_id: str,
    rps_values: List[float],
    duration_s: float,
    window_start_s: float,
    window_s: float,
    num_seeds: int,
    pair_manifest_csv: str,
    throughput_db: str,
    work_dir: str,
) -> pd.DataFrame:
    json_paths = _config_json_paths(pair_manifest_csv, config_id)
    if len(json_paths) == 0:
        raise ValueError(f"No matched benchmark JSONs for '{config_id}' in {pair_manifest_csv}")
    input_pool, output_pool = _extract_token_pools(json_paths)
    print(
        f"Token pools from {len(json_paths)} benchmark runs: "
        f"{input_pool.size} input lens, {output_pool.size} output lens"
    )

    work = Path(work_dir)
    (work / "requests").mkdir(parents=True, exist_ok=True)
    (work / "traces").mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for rps in rps_values:
        for seed in range(num_seeds):
            rng = np.random.default_rng(seed)
            requests = _generate_poisson_requests(
                duration_s=duration_s,
                rate_per_s=rps,
                input_pool=input_pool,
                output_pool=output_pool,
                rng=rng,
            )
            if len(requests) == 0:
                print(f"rps={rps} seed={seed}: no arrivals drawn, skipping")
                continue
            tag = f"rps{rps:g}_seed{seed}"
            requests_json = work / "requests" / f"{tag}.json"
            with open(requests_json, "w") as f:
                json.dump({"requests": requests}, f)
            out_csv = work / "traces" / f"{tag}.csv"
            meta = run_inference_from_artifacts(
                config_id=config_id,
                requests_json=str(requests_json),
                out_csv=str(out_csv),
                throughput_db=throughput_db,
                seed=seed,
            )
            dt = float(meta["dt"])
            trace = pd.read_csv(out_csv)
            lo = int(round(window_start_s / dt))
            hi = int(round((window_start_s + window_s) / dt))
            window = trace["power_w"].to_numpy()[lo:hi]
            if window.size == 0:
                print(f"rps={rps} seed={seed}: trace shorter than window, skipping")
                continue
            avg_w = float(np.mean(window))
            rows.append(
                {
                    "rps": float(rps),
                    "seed": int(seed),
                    "num_requests": int(len(requests)),
                    "avg_power_w": avg_w,
                }
            )
            print(f"rps={rps:>6g} seed={seed}: {len(requests):>5d} requests, avg power {avg_w:8.1f} W")

    return pd.DataFrame(rows)


def plot_sweep(per_run: pd.DataFrame, summary: pd.DataFrame, config_id: str, window_s: float, out_paths: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(summary["rps"], summary["mean_power_w"], marker="o", color="tab:blue")
    ax.fill_between(
        summary["rps"],
        summary["mean_power_w"] - summary["std_power_w"],
        summary["mean_power_w"] + summary["std_power_w"],
        alpha=0.25,
        color="tab:blue",
        label="$\\pm 1$ std",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel(f"Average power over {window_s:g} s window (W)")
    ax.set_title(config_id)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    for path in out_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", default="llama-3-70b_H100_tp8")
    parser.add_argument("--rps", type=float, nargs="+", default=DEFAULT_RPS)
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--window-start-s", type=float, default=180.0)
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    parser.add_argument("--throughput-db", default="model/throughput_database.json")
    parser.add_argument("--work-dir", default="results/rps_sweep")
    parser.add_argument("--out-csv", default="results/rps_sweep/rps_vs_avg_power.csv")
    parser.add_argument(
        "--out-fig",
        nargs="+",
        default=[
            "figures/rps_vs_avg_power_llama70b_tp8.pdf",
            "figures/rps_vs_avg_power_llama70b_tp8.png",
        ],
    )
    args = parser.parse_args()

    per_run = run_sweep(
        config_id=args.config_id,
        rps_values=list(args.rps),
        duration_s=args.duration_s,
        window_start_s=args.window_start_s,
        window_s=args.window_s,
        num_seeds=args.num_seeds,
        pair_manifest_csv=args.pair_manifest_csv,
        throughput_db=args.throughput_db,
        work_dir=args.work_dir,
    )
    if per_run.empty:
        raise SystemExit("No successful runs; nothing to plot.")

    summary = (
        per_run.groupby("rps")["avg_power_w"]
        .agg(mean_power_w="mean", std_power_w="std")
        .fillna(0.0)
        .reset_index()
        .sort_values("rps")
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(summary.to_string(index=False))

    max_train_rate = 4.0
    extrap = [r for r in summary["rps"] if r > max_train_rate]
    if extrap:
        print(
            f"NOTE: training data for this config covers rates up to {max_train_rate:g} req/s; "
            f"points {extrap} are extrapolation and may saturate at the model's power ceiling."
        )

    plot_sweep(per_run, summary, args.config_id, args.window_s, list(args.out_fig))


if __name__ == "__main__":
    main()
