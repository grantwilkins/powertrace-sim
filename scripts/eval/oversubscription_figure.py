#!/usr/bin/env python3
"""
Azure oversubscription capacity analysis and figures.

Produces:
  1) capacity/uncertainty scatter + envelopes vs rack count
  2) step-2(e)-style time-series line comparison (oversubscribed vs TDP-safe)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

import seaborn as sns

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir_for_file(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Union[str, Path], payload: Mapping[str, object]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "aggregated_dir": str(repo_root / "results" / "azure_facility" / "aggregated"),
        "out_capacity_plot": str(
            repo_root / "figures" / "azure_oversubscription_capacity.pdf"
        ),
        "out_lines_plot": str(
            repo_root / "figures" / "azure_oversubscription_lines.pdf"
        ),
        "out_csv": str(
            repo_root / "results" / "eval_paper" / "azure_oversubscription_capacity.csv"
        ),
        "out_json": str(
            repo_root
            / "results"
            / "eval_paper"
            / "azure_oversubscription_capacity.json"
        ),
    }


def _load_rack_matrix_kw(aggregated_dir: str) -> np.ndarray:
    paths = sorted(Path(aggregated_dir).glob("rack_*_*.npy"))
    if len(paths) <= 0:
        raise FileNotFoundError(
            f"No rack traces found in {aggregated_dir} (expected rack_*_*.npy)"
        )

    traces: List[np.ndarray] = []
    for path in paths:
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if arr.size <= 0:
            raise ValueError(f"Empty rack trace: {path}")
        traces.append(arr / 1000.0)  # kW

    lengths = sorted({int(x.size) for x in traces})
    if len(lengths) != 1:
        raise ValueError(f"Rack trace length mismatch: {lengths}")
    return np.stack(traces, axis=0).astype(np.float64)


def _sample_subset_indices(
    *,
    n_total: int,
    n_select: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_total_i = int(n_total)
    n_select_i = int(n_select)
    n_samples_i = int(n_samples)
    if n_total_i <= 0:
        raise ValueError("n_total must be >= 1")
    if n_select_i <= 0 or n_select_i > n_total_i:
        raise ValueError(f"n_select must be in [1,{n_total_i}], got {n_select_i}")
    if n_samples_i <= 0:
        raise ValueError("n_samples must be >= 1")

    if n_select_i == n_total_i:
        return np.tile(np.arange(n_total_i, dtype=np.int64), (1, 1))

    out = np.zeros((n_samples_i, n_select_i), dtype=np.int64)
    for i in range(n_samples_i):
        out[i, :] = np.asarray(
            rng.choice(n_total_i, size=n_select_i, replace=False),
            dtype=np.int64,
        )
    return out


def _aggregate_peaks_for_count(
    *,
    rack_matrix_kw: np.ndarray,
    n_select: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    idx = _sample_subset_indices(
        n_total=int(rack_matrix_kw.shape[0]),
        n_select=int(n_select),
        n_samples=int(n_samples),
        rng=rng,
    )
    peaks = np.zeros((int(idx.shape[0]),), dtype=np.float64)
    for i in range(int(idx.shape[0])):
        trace = np.sum(rack_matrix_kw[idx[i, :], :], axis=0, dtype=np.float64)
        peaks[i] = float(np.max(trace))
    return {
        "peaks_kw": peaks,
        "samples": int(idx.shape[0]),
    }


def _sample_traces_for_count(
    *,
    rack_matrix_kw: np.ndarray,
    n_select: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = _sample_subset_indices(
        n_total=int(rack_matrix_kw.shape[0]),
        n_select=int(n_select),
        n_samples=int(n_samples),
        rng=rng,
    )
    traces = np.zeros(
        (int(idx.shape[0]), int(rack_matrix_kw.shape[1])), dtype=np.float64
    )
    for i in range(int(idx.shape[0])):
        traces[i, :] = np.sum(rack_matrix_kw[idx[i, :], :], axis=0, dtype=np.float64)
    return traces


def _validate_positive(name: str, value: float) -> float:
    v = float(value)
    if (not np.isfinite(v)) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return float(v)


def compute_oversubscription_capacity(
    *,
    aggregated_dir: str,
    out_capacity_plot: str,
    out_lines_plot: str,
    out_csv: str,
    out_json: str,
    row_limit_kw: float = 600.0,
    rack_tdp_kw: float = 26.0,
    risk_percentile: float = 95.0,
    seed: int = 42,
    samples_per_count: int = 200,
    trace_samples: int = 80,
    tdp_racks: Optional[int] = None,
    oversub_racks: Optional[int] = None,
) -> Dict[str, object]:
    row_limit = _validate_positive("row_limit_kw", float(row_limit_kw))
    rack_tdp = _validate_positive("rack_tdp_kw", float(rack_tdp_kw))
    risk_p = float(risk_percentile)
    if (not np.isfinite(risk_p)) or risk_p <= 0.0 or risk_p > 100.0:
        raise ValueError(f"risk_percentile must be in (0,100], got {risk_percentile}")
    if int(samples_per_count) <= 0:
        raise ValueError("samples_per_count must be >= 1")
    if int(trace_samples) <= 0:
        raise ValueError("trace_samples must be >= 1")

    racks_kw = _load_rack_matrix_kw(aggregated_dir)
    n_racks = int(racks_kw.shape[0])
    timesteps = int(racks_kw.shape[1])
    if timesteps <= 0:
        raise ValueError("rack traces have zero timesteps")

    rng = np.random.default_rng(int(seed))

    rows: List[Dict[str, object]] = []
    peak_samples_by_count: Dict[int, np.ndarray] = {}
    for n in range(1, n_racks + 1):
        sample = _aggregate_peaks_for_count(
            rack_matrix_kw=racks_kw,
            n_select=int(n),
            n_samples=int(samples_per_count),
            rng=rng,
        )
        peaks_kw = np.asarray(sample["peaks_kw"], dtype=np.float64)
        peak_samples_by_count[int(n)] = peaks_kw
        row = {
            "n_racks": int(n),
            "n_samples": int(peaks_kw.size),
            "peak_mean_kw": float(np.mean(peaks_kw)),
            "peak_p05_kw": float(np.percentile(peaks_kw, 5)),
            "peak_p50_kw": float(np.percentile(peaks_kw, 50)),
            "peak_p95_kw": float(np.percentile(peaks_kw, 95)),
            "peak_prisk_kw": float(np.percentile(peaks_kw, risk_p)),
            "peak_max_kw": float(np.max(peaks_kw)),
            "exceed_prob": float(np.mean(peaks_kw > row_limit)),
        }
        rows.append(row)

    if tdp_racks is None:
        tdp_count = int(np.floor(row_limit / rack_tdp))
        tdp_count = int(max(1, min(n_racks, tdp_count)))
    else:
        tdp_count = int(tdp_racks)
        if tdp_count <= 0 or tdp_count > n_racks:
            raise ValueError(f"tdp_racks must be in [1,{n_racks}], got {tdp_racks}")

    valid_counts = [
        int(r["n_racks"]) for r in rows if float(r["peak_prisk_kw"]) <= float(row_limit)
    ]
    if oversub_racks is None:
        if len(valid_counts) > 0:
            oversub_count = int(max(valid_counts))
            oversub_note = "auto_from_risk_rule"
        else:
            oversub_count = 1
            oversub_note = "no_count_met_risk_rule_clamped_to_1"
    else:
        oversub_count = int(oversub_racks)
        if oversub_count <= 0 or oversub_count > n_racks:
            raise ValueError(
                f"oversub_racks must be in [1,{n_racks}], got {oversub_racks}"
            )
        oversub_note = "manual_override"

    # Figure A: capacity scatter + envelopes.
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    fig_a, ax_a = plt.subplots(figsize=(8, 4))
    for n in range(1, n_racks + 1):
        peaks = peak_samples_by_count[int(n)]
        ax_a.scatter(
            np.full(peaks.shape, float(n), dtype=np.float64),
            peaks,
            s=8,
            color="#bdbdbd",
            alpha=0.40,
            edgecolors="none",
            zorder=1,
        )
    x = np.asarray([float(r["n_racks"]) for r in rows], dtype=np.float64)
    y_p50 = np.asarray([float(r["peak_p50_kw"]) for r in rows], dtype=np.float64)
    y_p95 = np.asarray([float(r["peak_p95_kw"]) for r in rows], dtype=np.float64)
    ax_a.plot(x, y_p50, color="#2c3e50", linewidth=2.0, label="Median peak", zorder=3)
    ax_a.plot(
        x,
        y_p95,
        color="#34495e",
        linewidth=1.8,
        linestyle="--",
        label="P95 peak",
        zorder=3,
    )
    ax_a.axhline(
        row_limit,
        color="#000000",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label=f"Row limit ({row_limit:.0f} kW)",
    )
    ax_a.axvline(
        tdp_count,
        color="#e67e22",
        linewidth=1.4,
        linestyle="-.",
        alpha=0.85,
        label=f"TDP-safe N={tdp_count}",
    )
    ax_a.axvline(
        oversub_count,
        color="#2980b9",
        linewidth=1.4,
        linestyle="-",
        alpha=0.9,
        label=f"Oversub N={oversub_count}",
    )
    ax_a.set_xlabel("Rack count (N)")
    ax_a.set_ylabel("Peak Row Power (kW)")
    ax_a.grid(True, alpha=0.25, linestyle=":")
    ax_a.set_xlim(1.0, float(n_racks))
    ax_a.legend(loc="best")
    fig_a.tight_layout()
    _ensure_dir_for_file(out_capacity_plot)
    fig_a.savefig(out_capacity_plot, bbox_inches="tight")
    plt.close(fig_a)

    # Figure B: line comparison with uncertainty bands over time.
    traces_oversub = _sample_traces_for_count(
        rack_matrix_kw=racks_kw,
        n_select=int(oversub_count),
        n_samples=int(trace_samples),
        rng=rng,
    )
    traces_tdp = _sample_traces_for_count(
        rack_matrix_kw=racks_kw,
        n_select=int(tdp_count),
        n_samples=int(trace_samples),
        rng=rng,
    )

    oversub_med = np.percentile(traces_oversub, 50, axis=0)
    oversub_p05 = np.percentile(traces_oversub, 5, axis=0)
    oversub_p95 = np.percentile(traces_oversub, 95, axis=0)
    tdp_med = np.percentile(traces_tdp, 50, axis=0)
    tdp_p05 = np.percentile(traces_tdp, 5, axis=0)
    tdp_p95 = np.percentile(traces_tdp, 95, axis=0)
    hours = np.arange(timesteps, dtype=np.float64) / 3600.0

    fig_b, ax_b = plt.subplots(figsize=(8, 5))
    ax_b.fill_between(
        hours, oversub_p05, oversub_p95, color="#95a5a6", alpha=0.25, linewidth=0.0
    )
    ax_b.fill_between(
        hours, tdp_p05, tdp_p95, color="#d0d3d4", alpha=0.30, linewidth=0.0
    )
    ax_b.plot(
        hours,
        oversub_med,
        color="#2980b9",
        linewidth=1.8,
        label=f"{oversub_count} Racks (Oversubscribed)",
    )
    ax_b.plot(
        hours,
        tdp_med,
        color="#e67e22",
        linewidth=1.8,
        label=f"{tdp_count} Racks (TDP-safe)",
    )
    ax_b.axhline(
        row_limit,
        color="#000000",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label="Row limit",
    )
    ax_b.set_xlabel("Time (hours)")
    ax_b.set_ylabel("Row Power (kW)")
    ax_b.grid(True, alpha=0.25, linestyle=":")
    ax_b.set_xlim(0.0, float(hours[-1]) if hours.size > 1 else 1.0)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=2)
    fig_b.tight_layout()
    _ensure_dir_for_file(out_lines_plot)
    fig_b.savefig(out_lines_plot, bbox_inches="tight")
    plt.close(fig_b)

    _ensure_dir_for_file(out_csv)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_racks",
                "n_samples",
                "peak_mean_kw",
                "peak_p05_kw",
                "peak_p50_kw",
                "peak_p95_kw",
                "peak_prisk_kw",
                "peak_max_kw",
                "exceed_prob",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "ok",
        "aggregated_dir": str(aggregated_dir),
        "inputs": {
            "row_limit_kw": float(row_limit),
            "rack_tdp_kw": float(rack_tdp),
            "risk_percentile": float(risk_p),
            "seed": int(seed),
            "samples_per_count": int(samples_per_count),
            "trace_samples": int(trace_samples),
        },
        "dataset": {
            "n_racks_available": int(n_racks),
            "timesteps_1s": int(timesteps),
        },
        "selection": {
            "tdp_racks": int(tdp_count),
            "oversub_racks": int(oversub_count),
            "oversub_note": str(oversub_note),
            "valid_counts_under_risk_rule": [int(x) for x in valid_counts],
        },
        "summary_rows": rows,
        "outputs": {
            "capacity_plot_pdf": str(out_capacity_plot),
            "lines_plot_pdf": str(out_lines_plot),
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    _write_json(out_json, payload)
    return payload


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Azure oversubscription capacity analysis from aggregated rack traces."
    )
    parser.add_argument("--aggregated-dir", default=defaults["aggregated_dir"])
    parser.add_argument("--row-limit-kw", type=float, default=600.0)
    parser.add_argument("--rack-tdp-kw", type=float, default=26.0)
    parser.add_argument("--risk-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-count", type=int, default=200)
    parser.add_argument("--trace-samples", type=int, default=80)
    parser.add_argument("--tdp-racks", type=int, default=None)
    parser.add_argument("--oversub-racks", type=int, default=None)
    parser.add_argument("--out-capacity-plot", default=defaults["out_capacity_plot"])
    parser.add_argument("--out-lines-plot", default=defaults["out_lines_plot"])
    parser.add_argument("--out-csv", default=defaults["out_csv"])
    parser.add_argument("--out-json", default=defaults["out_json"])
    args = parser.parse_args()

    result = compute_oversubscription_capacity(
        aggregated_dir=str(args.aggregated_dir),
        out_capacity_plot=str(args.out_capacity_plot),
        out_lines_plot=str(args.out_lines_plot),
        out_csv=str(args.out_csv),
        out_json=str(args.out_json),
        row_limit_kw=float(args.row_limit_kw),
        rack_tdp_kw=float(args.rack_tdp_kw),
        risk_percentile=float(args.risk_percentile),
        seed=int(args.seed),
        samples_per_count=int(args.samples_per_count),
        trace_samples=int(args.trace_samples),
        tdp_racks=args.tdp_racks,
        oversub_racks=args.oversub_racks,
    )

    print("=" * 72)
    print("Azure Oversubscription Capacity")
    print("=" * 72)
    print(f"Aggregated dir : {result['aggregated_dir']}")
    print(f"Available racks: {result['dataset']['n_racks_available']}")
    print(f"TDP-safe N     : {result['selection']['tdp_racks']}")
    print(f"Oversub N      : {result['selection']['oversub_racks']}")
    print(f"Capacity plot  : {result['outputs']['capacity_plot_pdf']}")
    print(f"Lines plot     : {result['outputs']['lines_plot_pdf']}")
    print(f"CSV            : {result['outputs']['csv']}")
    print(f"JSON           : {result['outputs']['json']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
