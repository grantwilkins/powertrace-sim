#!/usr/bin/env python3
"""
Experiment 3a: Variance reduction analysis under aggregation.

Computes coefficient of variation (CoV = sigma / mu) for random node subsets
and compares empirical collapse against a 1/sqrt(N) independence reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUBSET_SIZES: Tuple[int, ...] = (1, 4, 16, 60, 120, 240)
CONF_BAND_Q_LOW = 5.0
CONF_BAND_Q_HIGH = 95.0


@dataclass(frozen=True)
class FacilityLayout:
    rows: int = 10
    racks_per_row: int = 6
    nodes_per_rack: int = 4

    @property
    def n_nodes(self) -> int:
        return int(self.rows) * int(self.racks_per_row) * int(self.nodes_per_rack)

    def iter_nodes(self) -> Iterable[Tuple[int, int, int]]:
        for row in range(int(self.rows)):
            for rack in range(int(self.racks_per_row)):
                for node in range(int(self.nodes_per_rack)):
                    yield int(row), int(rack), int(node)


def _ensure_dir_for_file(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Union[str, Path], payload: Mapping[str, object]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "node_trace_dir": str(repo_root / "results" / "azure_facility" / "node_traces"),
        "out_plot": str(repo_root / "figures" / "azure_aggregation_cov_vs_nodes.pdf"),
        "out_csv": str(repo_root / "results" / "eval_paper" / "azure_aggregation_cov_vs_nodes.csv"),
        "out_json": str(repo_root / "results" / "eval_paper" / "azure_aggregation_cov_vs_nodes.json"),
    }


def _parse_subset_sizes(raw: str) -> Tuple[int, ...]:
    out: List[int] = []
    for token in str(raw).split(","):
        t = token.strip()
        if t == "":
            continue
        try:
            value = int(t)
        except Exception as exc:
            raise ValueError(f"Invalid subset size token: '{t}'") from exc
        out.append(int(value))
    if len(out) == 0:
        raise ValueError("subset sizes cannot be empty")
    if len(set(out)) != len(out):
        raise ValueError(f"subset sizes must be unique, got: {out}")
    if any(int(x) <= 0 for x in out):
        raise ValueError(f"subset sizes must be positive, got: {out}")
    return tuple(int(x) for x in out)


def _compute_cov(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if x.size <= 0:
        raise ValueError("Cannot compute CoV for empty array")
    mu = float(np.mean(x))
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"Non-positive or non-finite mean in CoV computation: {mu}")
    sigma = float(np.std(x, ddof=0))
    return float(sigma / mu)


def _load_node_matrix(
    *,
    node_trace_dir: str,
    layout: FacilityLayout,
) -> Dict[str, object]:
    expected_paths: List[Tuple[int, int, int, str]] = []
    for row, rack, node in layout.iter_nodes():
        p = os.path.join(node_trace_dir, f"node_{row}_{rack}_{node}.npy")
        expected_paths.append((row, rack, node, p))

    missing = [p for _, _, _, p in expected_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} node traces in {node_trace_dir}; first missing: {missing[0]}"
        )

    first = np.asarray(np.load(expected_paths[0][3]), dtype=np.float64).reshape(-1)
    if first.size <= 0:
        raise ValueError(f"Empty node trace: {expected_paths[0][3]}")
    t_horizon = int(first.size)

    matrix = np.zeros((int(layout.n_nodes), int(t_horizon)), dtype=np.float32)
    node_ids: List[str] = []

    idx = 0
    for row, rack, node, path in expected_paths:
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if int(arr.size) != int(t_horizon):
            raise ValueError(
                f"Trace length mismatch for {path}: got {arr.size}, expected {t_horizon}"
            )
        matrix[idx, :] = np.asarray(arr, dtype=np.float32)
        node_ids.append(f"{row}_{rack}_{node}")
        idx += 1

    return {
        "matrix": matrix,
        "node_ids": node_ids,
        "timesteps": int(t_horizon),
    }


def compute_aggregation_variance(
    *,
    node_trace_dir: str,
    out_plot: str,
    out_csv: str,
    out_json: str,
    subset_sizes: Sequence[int] = DEFAULT_SUBSET_SIZES,
    repeats: int = 50,
    seed: int = 42,
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
) -> Dict[str, object]:
    if int(repeats) <= 0:
        raise ValueError("repeats must be >= 1")

    layout = FacilityLayout(
        rows=int(rows),
        racks_per_row=int(racks_per_row),
        nodes_per_rack=int(nodes_per_rack),
    )
    if layout.n_nodes <= 0:
        raise ValueError("layout yields zero nodes")

    subset_int = tuple(int(x) for x in subset_sizes)
    if len(subset_int) == 0:
        raise ValueError("subset_sizes must be non-empty")
    if len(set(subset_int)) != len(subset_int):
        raise ValueError(f"subset_sizes must be unique, got {subset_int}")
    if any(int(x) <= 0 for x in subset_int):
        raise ValueError(f"subset_sizes must be positive, got {subset_int}")
    if any(int(x) > int(layout.n_nodes) for x in subset_int):
        raise ValueError(
            f"subset_sizes must be <= n_nodes={layout.n_nodes}, got {subset_int}"
        )
    if 1 not in set(subset_int):
        raise ValueError("subset_sizes must include 1 to build 1/sqrt(N) reference")

    loaded = _load_node_matrix(node_trace_dir=node_trace_dir, layout=layout)
    matrix = np.asarray(loaded["matrix"], dtype=np.float32)
    timesteps = int(loaded["timesteps"])

    node_cov_all = np.asarray(
        [_compute_cov(matrix[i, :]) for i in range(int(layout.n_nodes))],
        dtype=np.float64,
    )

    rng = np.random.default_rng(int(seed))
    all_rows: List[Dict[str, object]] = []
    summary: Dict[int, Dict[str, float]] = {}

    for n in subset_int:
        covs: List[float] = []
        sampled_ids: List[List[int]] = []
        for r in range(int(repeats)):
            idx = np.asarray(
                rng.choice(int(layout.n_nodes), size=int(n), replace=False),
                dtype=np.int64,
            )
            agg = np.sum(matrix[idx, :], axis=0, dtype=np.float64)
            cov = _compute_cov(agg)
            covs.append(float(cov))
            sampled_ids.append([int(i) for i in idx.tolist()])

        cov_arr = np.asarray(covs, dtype=np.float64)
        summary[int(n)] = {
            "cov_mean": float(np.mean(cov_arr)),
            "cov_p05": float(np.percentile(cov_arr, CONF_BAND_Q_LOW)),
            "cov_p95": float(np.percentile(cov_arr, CONF_BAND_Q_HIGH)),
        }

        for r in range(int(repeats)):
            all_rows.append(
                {
                    "subset_size": int(n),
                    "repeat_idx": int(r),
                    "cov": float(cov_arr[r]),
                    "sampled_node_ids": ";".join(str(x) for x in sampled_ids[r]),
                }
            )

    cov_ref = float(summary[1]["cov_mean"])
    for n in subset_int:
        summary[int(n)]["cov_theory"] = float(cov_ref / math.sqrt(float(n)))

    x = np.asarray([float(n) for n in subset_int], dtype=np.float64)
    y = np.asarray([float(summary[int(n)]["cov_mean"]) for n in subset_int], dtype=np.float64)
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    y_hat = np.exp(float(intercept) + (float(slope) * np.log(x)))
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0.0 else float("nan")

    _ensure_dir_for_file(out_csv)
    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "subset_size",
            "repeat_idx",
            "cov",
            "sampled_node_ids",
            "cov_mean",
            "cov_p05",
            "cov_p95",
            "cov_theory",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            n = int(row["subset_size"])
            writer.writerow(
                {
                    "subset_size": int(row["subset_size"]),
                    "repeat_idx": int(row["repeat_idx"]),
                    "cov": float(row["cov"]),
                    "sampled_node_ids": str(row["sampled_node_ids"]),
                    "cov_mean": float(summary[n]["cov_mean"]),
                    "cov_p05": float(summary[n]["cov_p05"]),
                    "cov_p95": float(summary[n]["cov_p95"]),
                    "cov_theory": float(summary[n]["cov_theory"]),
                }
            )

    x_plot = np.asarray([float(n) for n in subset_int], dtype=np.float64)
    y_mean = np.asarray([float(summary[int(n)]["cov_mean"]) for n in subset_int], dtype=np.float64)
    y_lo = np.asarray([float(summary[int(n)]["cov_p05"]) for n in subset_int], dtype=np.float64)
    y_hi = np.asarray([float(summary[int(n)]["cov_p95"]) for n in subset_int], dtype=np.float64)
    y_theory = np.asarray([float(summary[int(n)]["cov_theory"]) for n in subset_int], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(
        x_plot,
        y_mean,
        color="#2c3e50",
        marker="o",
        linewidth=2.0,
        label="Empirical mean CoV",
    )
    ax.fill_between(
        x_plot,
        y_lo,
        y_hi,
        color="#2c3e50",
        alpha=0.20,
        label="Empirical p5-p95",
    )
    ax.plot(
        x_plot,
        y_theory,
        linestyle="--",
        color="#e67e22",
        linewidth=2.0,
        label="Theory 1/sqrt(N)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N_nodes")
    ax.set_ylabel("CoV (sigma/mu)")
    ax.set_title("Aggregation Variance Reduction (Azure 240-node traces)")
    ax.grid(True, which="both", alpha=0.35, linestyle=":")
    ax.legend(loc="best")
    _ensure_dir_for_file(out_plot)
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "status": "ok",
        "node_trace_dir": str(node_trace_dir),
        "layout": {
            "rows": int(layout.rows),
            "racks_per_row": int(layout.racks_per_row),
            "nodes_per_rack": int(layout.nodes_per_rack),
            "n_nodes": int(layout.n_nodes),
        },
        "timesteps": int(timesteps),
        "sampling": {
            "subset_sizes": [int(x) for x in subset_int],
            "repeats": int(repeats),
            "seed": int(seed),
            "confidence_band_percentiles": [float(CONF_BAND_Q_LOW), float(CONF_BAND_Q_HIGH)],
        },
        "single_node_cov": {
            "count": int(node_cov_all.size),
            "mean": float(np.mean(node_cov_all)),
            "median": float(np.median(node_cov_all)),
            "min": float(np.min(node_cov_all)),
            "max": float(np.max(node_cov_all)),
        },
        "fit_loglog_cov_vs_n": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
        },
        "summary_by_subset": {
            str(int(n)): {
                "cov_mean": float(summary[int(n)]["cov_mean"]),
                "cov_p05": float(summary[int(n)]["cov_p05"]),
                "cov_p95": float(summary[int(n)]["cov_p95"]),
                "cov_theory": float(summary[int(n)]["cov_theory"]),
            }
            for n in subset_int
        },
        "outputs": {
            "plot_pdf": str(out_plot),
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    _write_json(out_json, payload)
    return payload


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Experiment 3a: CoV variance reduction vs aggregation size for Azure node traces."
    )
    parser.add_argument("--node-trace-dir", default=defaults["node_trace_dir"])
    parser.add_argument("--out-plot", default=defaults["out_plot"])
    parser.add_argument("--out-csv", default=defaults["out_csv"])
    parser.add_argument("--out-json", default=defaults["out_json"])
    parser.add_argument(
        "--subset-sizes",
        default="1,4,16,60,120,240",
        help="Comma-separated node subset sizes.",
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--nodes-per-rack", type=int, default=4)
    args = parser.parse_args()

    subset_sizes = _parse_subset_sizes(str(args.subset_sizes))
    result = compute_aggregation_variance(
        node_trace_dir=str(args.node_trace_dir),
        out_plot=str(args.out_plot),
        out_csv=str(args.out_csv),
        out_json=str(args.out_json),
        subset_sizes=subset_sizes,
        repeats=int(args.repeats),
        seed=int(args.seed),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        nodes_per_rack=int(args.nodes_per_rack),
    )

    fit = result["fit_loglog_cov_vs_n"]
    print("=" * 72)
    print("Experiment 3a: Aggregation Variance")
    print("=" * 72)
    print(f"Node traces : {result['node_trace_dir']}")
    print(f"Subsets     : {result['sampling']['subset_sizes']}")
    print(f"Repeats     : {result['sampling']['repeats']}")
    print(f"Seed        : {result['sampling']['seed']}")
    print(f"log-log slope (empirical): {float(fit['slope']):.4f}")
    print(f"R^2                    : {float(fit['r2']):.4f}")
    print(f"Figure     : {result['outputs']['plot_pdf']}")
    print(f"CSV        : {result['outputs']['csv']}")
    print(f"JSON       : {result['outputs']['json']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
