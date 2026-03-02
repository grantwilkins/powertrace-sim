#!/usr/bin/env python3
"""
Experiment 3c: Azure hierarchy figure(s) (server -> rack -> row -> site).

Supports:
  - combined 2x2 layout (legacy-compatible)
  - separate panel-ready files (raw + trend only, no text boxes)
  - both outputs in one run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PANEL_ORDER = ("server", "rack", "row", "site")


def _ensure_dir_for_file(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Union[str, Path], payload: Mapping[str, object]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    fig_dir = repo_root / "figures"
    return {
        "node_trace_dir": str(repo_root / "results" / "azure_facility" / "node_traces"),
        "aggregated_dir": str(repo_root / "results" / "azure_facility" / "aggregated"),
        "out_plot": str(fig_dir / "azure_hierarchy_figure.pdf"),
        "out_csv": str(
            repo_root / "results" / "eval_paper" / "azure_hierarchy_figure.csv"
        ),
        "out_json": str(
            repo_root / "results" / "eval_paper" / "azure_hierarchy_figure.json"
        ),
        "panel_out_dir": str(fig_dir),
    }


def _parse_node_id(raw: str) -> Tuple[int, int, int]:
    toks = [x.strip() for x in str(raw).split("_")]
    if len(toks) != 3:
        raise ValueError(f"node-id must be '<row>_<rack>_<node>', got: {raw}")
    try:
        row = int(toks[0])
        rack = int(toks[1])
        node = int(toks[2])
    except Exception as exc:
        raise ValueError(f"Invalid node-id: {raw}") from exc
    if row < 0 or rack < 0 or node < 0:
        raise ValueError(f"node-id values must be non-negative, got: {raw}")
    return int(row), int(rack), int(node)


def _parse_rack_id(raw: str) -> Tuple[int, int]:
    toks = [x.strip() for x in str(raw).split("_")]
    if len(toks) != 2:
        raise ValueError(f"rack-id must be '<row>_<rack>', got: {raw}")
    try:
        row = int(toks[0])
        rack = int(toks[1])
    except Exception as exc:
        raise ValueError(f"Invalid rack-id: {raw}") from exc
    if row < 0 or rack < 0:
        raise ValueError(f"rack-id values must be non-negative, got: {raw}")
    return int(row), int(rack)


def _parse_row_id(raw: str) -> int:
    try:
        row = int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid row-id: {raw}") from exc
    if row < 0:
        raise ValueError(f"row-id must be non-negative, got: {raw}")
    return int(row)


def _load_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required array not found: {path}")
    arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        raise ValueError(f"Empty array: {path}")
    return arr


def _downsample_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    f = int(factor)
    if f <= 0:
        raise ValueError("downsample factor must be >= 1")
    if x.size % f != 0:
        raise ValueError(f"Array length {x.size} not divisible by factor {f}")
    return np.mean(x.reshape(-1, f), axis=1).astype(np.float64)


def _moving_mean(arr: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    w = int(window)
    if w <= 0:
        raise ValueError("moving window must be >= 1")
    if w > x.size:
        raise ValueError(f"moving window {w} exceeds array length {x.size}")
    kernel = np.ones((w,), dtype=np.float64) / float(w)
    return np.convolve(x, kernel, mode="same").astype(np.float64)


def _compute_level_metrics(arr_1s_w: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr_1s_w, dtype=np.float64).reshape(-1)
    if x.size <= 0:
        raise ValueError("Cannot compute metrics on empty array")
    mean = float(np.mean(x))
    if mean <= 0.0 or (not np.isfinite(mean)):
        raise ValueError(f"Mean power must be positive and finite, got {mean}")
    std = float(np.std(x, ddof=0))
    cov = float(std / mean)
    par_1s = float(np.max(x) / mean)
    x_15m = _downsample_mean(x, 900)
    mean_15m = float(np.mean(x_15m))
    par_15m = float(np.max(x_15m) / mean_15m)
    return {
        "mean_kw": float(mean / 1000.0),
        "std_kw": float(std / 1000.0),
        "cov": float(cov),
        "par_1s": float(par_1s),
        "par_15min": float(par_15m),
        "n_samples_1s": int(x.size),
        "n_samples_15min": int(x_15m.size),
    }


def _prepare_hierarchy_bundle(
    *,
    node_trace_dir: str,
    aggregated_dir: str,
    node_id: str,
    rack_id: str,
    row_id: str,
    server_downsample_factor: int,
    trend_window_s: int,
) -> Dict[str, object]:
    if int(server_downsample_factor) <= 0:
        raise ValueError("server_downsample_factor must be >= 1")
    if int(trend_window_s) <= 0:
        raise ValueError("trend_window_s must be >= 1")

    node_row, node_rack, node_node = _parse_node_id(node_id)
    rack_row, rack_col = _parse_rack_id(rack_id)
    row_idx = _parse_row_id(row_id)

    node_path = os.path.join(
        node_trace_dir, f"node_{node_row}_{node_rack}_{node_node}.npy"
    )
    rack_path = os.path.join(aggregated_dir, f"rack_{rack_row}_{rack_col}.npy")
    row_path = os.path.join(aggregated_dir, f"row_{row_idx}.npy")
    site_path = os.path.join(aggregated_dir, "site_it_1s.npy")

    server_250ms = _load_array(node_path)
    server_1s = _downsample_mean(server_250ms, int(server_downsample_factor))
    rack_1s = _load_array(rack_path)
    row_1s = _load_array(row_path)
    site_1s = _load_array(site_path)

    n_1s = int(server_1s.size)
    if (
        int(rack_1s.size) != n_1s
        or int(row_1s.size) != n_1s
        or int(site_1s.size) != n_1s
    ):
        raise ValueError(
            "Hierarchy traces must have matching 1s lengths after server downsample: "
            f"server={server_1s.size}, rack={rack_1s.size}, row={row_1s.size}, site={site_1s.size}"
        )
    if n_1s % 900 != 0:
        raise ValueError(
            f"1s trace length {n_1s} is not divisible by 900 for PAR(15min)"
        )
    if int(trend_window_s) > n_1s:
        raise ValueError(
            f"trend_window_s ({trend_window_s}) exceeds 1s trace length ({n_1s})"
        )

    levels = [
        ("server", "Server", server_1s, f"node_{node_row}_{node_rack}_{node_node}"),
        ("rack", "Rack", rack_1s, f"rack_{rack_row}_{rack_col}"),
        ("row", "Row", row_1s, f"row_{row_idx}"),
        ("site", "Site", site_1s, "site_it_1s"),
    ]

    hours = np.arange(n_1s, dtype=np.float64) / 3600.0

    csv_rows: List[Dict[str, object]] = []
    metrics_by_level: Dict[str, Dict[str, float]] = {}
    for level_key, _title, arr_1s, series_id in levels:
        m = _compute_level_metrics(np.asarray(arr_1s, dtype=np.float64))
        metrics_by_level[level_key] = dict(m)
        csv_rows.append(
            {
                "level": str(level_key),
                "series_id": str(series_id),
                "mean_kw": float(m["mean_kw"]),
                "std_kw": float(m["std_kw"]),
                "cov": float(m["cov"]),
                "par_1s": float(m["par_1s"]),
                "par_15min": float(m["par_15min"]),
                "n_samples_1s": int(m["n_samples_1s"]),
                "n_samples_15min": int(m["n_samples_15min"]),
            }
        )

    return {
        "inputs": {
            "node_trace_dir": str(node_trace_dir),
            "aggregated_dir": str(aggregated_dir),
            "node_trace_path": str(node_path),
            "rack_trace_path": str(rack_path),
            "row_trace_path": str(row_path),
            "site_trace_path": str(site_path),
        },
        "selected_ids": {
            "node_id": f"{node_row}_{node_rack}_{node_node}",
            "rack_id": f"{rack_row}_{rack_col}",
            "row_id": int(row_idx),
        },
        "timing": {
            "n_samples_1s": int(n_1s),
            "server_downsample_factor": int(server_downsample_factor),
            "trend_window_s": int(trend_window_s),
        },
        "levels": levels,
        "hours": hours,
        "csv_rows": csv_rows,
        "metrics_by_level": metrics_by_level,
    }


def _write_summary_csv(path: str, csv_rows: List[Dict[str, object]]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "series_id",
                "mean_kw",
                "std_kw",
                "cov",
                "par_1s",
                "par_15min",
                "n_samples_1s",
                "n_samples_15min",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)


def _plot_combined(
    *,
    out_plot: str,
    levels: List[Tuple[str, str, np.ndarray, str]],
    hours: np.ndarray,
    trend_window_s: int,
    metrics_by_level: Mapping[str, Mapping[str, float]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), sharex=True)
    axes_flat = list(axes.reshape(-1))
    trend_window = int(trend_window_s)

    for ax, (level_key, title, arr_1s, _series_id) in zip(axes_flat, levels):
        raw_kw = np.asarray(arr_1s, dtype=np.float64) / 1000.0
        trend_kw = (
            _moving_mean(np.asarray(arr_1s, dtype=np.float64), trend_window) / 1000.0
        )

        ax.plot(hours, raw_kw, color="#bdc3c7", linewidth=0.7, label="Raw (1s)")
        ax.plot(
            hours,
            trend_kw,
            color="#2c3e50",
            linewidth=1.7,
            label=f"Trend ({trend_window_s}s)",
        )
        ax.set_title(title)
        ax.set_ylabel("Power (kW)")
        ax.set_xlim(0.0, 24.0)
        ax.grid(True, alpha=0.25, linestyle=":")

        m = metrics_by_level[level_key]
        ax.text(
            0.02,
            0.98,
            f"CoV={float(m['cov']):.3f}\nPAR(1s)={float(m['par_1s']):.3f}\nPAR(15m)={float(m['par_15min']):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.85},
        )

    axes_flat[0].legend(loc="best", fontsize=8)
    axes_flat[2].set_xlabel("Time (hours)")
    axes_flat[3].set_xlabel("Time (hours)")
    fig.suptitle("Azure Hierarchy Figure: Server -> Rack -> Row -> Site", y=1.02)
    fig.tight_layout()
    _ensure_dir_for_file(out_plot)
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)


def _plot_separate_panels(
    *,
    panel_out_dir: str,
    levels: List[Tuple[str, str, np.ndarray, str]],
    hours: np.ndarray,
    trend_window_s: int,
) -> Dict[str, str]:
    Path(panel_out_dir).mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}
    trend_window = int(trend_window_s)

    for level_key, _title, arr_1s, _series_id in levels:
        raw_kw = np.asarray(arr_1s, dtype=np.float64) / 1000.0
        trend_kw = (
            _moving_mean(np.asarray(arr_1s, dtype=np.float64), trend_window) / 1000.0
        )
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.2)
        fig, ax = plt.subplots(figsize=(7.0, 4))
        ax.plot(hours, raw_kw, color="#bdc3c7", linewidth=0.7, label="Raw (1s)")
        ax.plot(
            hours,
            trend_kw,
            color="#2c3e50",
            linewidth=1.7,
            label=f"Trend ({trend_window_s}s)",
        )
        ax.set_xlim(0.0, 24.0)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Power (kW)")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = Path(panel_out_dir) / f"azure_hierarchy_{level_key}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        out[level_key] = str(out_path)

    return out


def generate_hierarchy_figure(
    *,
    node_trace_dir: str,
    aggregated_dir: str,
    out_plot: str,
    out_csv: str,
    out_json: str,
    node_id: str = "0_0_0",
    rack_id: str = "0_0",
    row_id: str = "0",
    server_downsample_factor: int = 4,
    trend_window_s: int = 900,
) -> Dict[str, object]:
    bundle = _prepare_hierarchy_bundle(
        node_trace_dir=node_trace_dir,
        aggregated_dir=aggregated_dir,
        node_id=node_id,
        rack_id=rack_id,
        row_id=row_id,
        server_downsample_factor=int(server_downsample_factor),
        trend_window_s=int(trend_window_s),
    )
    _plot_combined(
        out_plot=str(out_plot),
        levels=list(bundle["levels"]),
        hours=np.asarray(bundle["hours"], dtype=np.float64),
        trend_window_s=int(trend_window_s),
        metrics_by_level=bundle["metrics_by_level"],
    )
    _write_summary_csv(str(out_csv), list(bundle["csv_rows"]))

    payload = {
        "status": "ok",
        "output_mode": "combined",
        "inputs": dict(bundle["inputs"]),
        "selected_ids": dict(bundle["selected_ids"]),
        "timing": dict(bundle["timing"]),
        "metrics_by_level": dict(bundle["metrics_by_level"]),
        "outputs": {
            "plot_pdf": str(out_plot),
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    _write_json(out_json, payload)
    return payload


def generate_hierarchy_panels(
    *,
    node_trace_dir: str,
    aggregated_dir: str,
    panel_out_dir: str,
    out_csv: str,
    out_json: str,
    node_id: str = "0_0_0",
    rack_id: str = "0_0",
    row_id: str = "0",
    server_downsample_factor: int = 4,
    trend_window_s: int = 900,
) -> Dict[str, object]:
    bundle = _prepare_hierarchy_bundle(
        node_trace_dir=node_trace_dir,
        aggregated_dir=aggregated_dir,
        node_id=node_id,
        rack_id=rack_id,
        row_id=row_id,
        server_downsample_factor=int(server_downsample_factor),
        trend_window_s=int(trend_window_s),
    )
    panel_files = _plot_separate_panels(
        panel_out_dir=str(panel_out_dir),
        levels=list(bundle["levels"]),
        hours=np.asarray(bundle["hours"], dtype=np.float64),
        trend_window_s=int(trend_window_s),
    )
    _write_summary_csv(str(out_csv), list(bundle["csv_rows"]))
    payload = {
        "status": "ok",
        "output_mode": "separate",
        "inputs": dict(bundle["inputs"]),
        "selected_ids": dict(bundle["selected_ids"]),
        "timing": dict(bundle["timing"]),
        "metrics_by_level": dict(bundle["metrics_by_level"]),
        "panel_files": {k: str(v) for k, v in panel_files.items()},
        "outputs": {
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    _write_json(out_json, payload)
    return payload


def generate_hierarchy_outputs(
    *,
    node_trace_dir: str,
    aggregated_dir: str,
    out_plot: str,
    panel_out_dir: str,
    out_csv: str,
    out_json: str,
    output_mode: str = "separate",
    node_id: str = "0_0_0",
    rack_id: str = "0_0",
    row_id: str = "0",
    server_downsample_factor: int = 4,
    trend_window_s: int = 900,
) -> Dict[str, object]:
    mode = str(output_mode).strip().lower()
    if mode not in {"combined", "separate", "both"}:
        raise ValueError(
            f"output_mode must be one of combined,separate,both; got {output_mode}"
        )

    if mode == "combined":
        return generate_hierarchy_figure(
            node_trace_dir=node_trace_dir,
            aggregated_dir=aggregated_dir,
            out_plot=out_plot,
            out_csv=out_csv,
            out_json=out_json,
            node_id=node_id,
            rack_id=rack_id,
            row_id=row_id,
            server_downsample_factor=int(server_downsample_factor),
            trend_window_s=int(trend_window_s),
        )
    if mode == "separate":
        return generate_hierarchy_panels(
            node_trace_dir=node_trace_dir,
            aggregated_dir=aggregated_dir,
            panel_out_dir=panel_out_dir,
            out_csv=out_csv,
            out_json=out_json,
            node_id=node_id,
            rack_id=rack_id,
            row_id=row_id,
            server_downsample_factor=int(server_downsample_factor),
            trend_window_s=int(trend_window_s),
        )

    combined = generate_hierarchy_figure(
        node_trace_dir=node_trace_dir,
        aggregated_dir=aggregated_dir,
        out_plot=out_plot,
        out_csv=out_csv,
        out_json=out_json,
        node_id=node_id,
        rack_id=rack_id,
        row_id=row_id,
        server_downsample_factor=int(server_downsample_factor),
        trend_window_s=int(trend_window_s),
    )
    separate = generate_hierarchy_panels(
        node_trace_dir=node_trace_dir,
        aggregated_dir=aggregated_dir,
        panel_out_dir=panel_out_dir,
        out_csv=out_csv,
        out_json=out_json,
        node_id=node_id,
        rack_id=rack_id,
        row_id=row_id,
        server_downsample_factor=int(server_downsample_factor),
        trend_window_s=int(trend_window_s),
    )
    payload = {
        "status": "ok",
        "output_mode": "both",
        "inputs": dict(separate["inputs"]),
        "selected_ids": dict(separate["selected_ids"]),
        "timing": dict(separate["timing"]),
        "metrics_by_level": dict(separate["metrics_by_level"]),
        "panel_files": dict(separate["panel_files"]),
        "outputs": {
            "combined_plot_pdf": str(combined["outputs"]["plot_pdf"]),
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    _write_json(out_json, payload)
    return payload


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Experiment 3c: Azure hierarchy figure outputs (combined and/or separate panels)."
    )
    parser.add_argument("--node-trace-dir", default=defaults["node_trace_dir"])
    parser.add_argument("--aggregated-dir", default=defaults["aggregated_dir"])
    parser.add_argument("--out-plot", default=defaults["out_plot"])
    parser.add_argument("--panel-out-dir", default=defaults["panel_out_dir"])
    parser.add_argument("--out-csv", default=defaults["out_csv"])
    parser.add_argument("--out-json", default=defaults["out_json"])
    parser.add_argument(
        "--output-mode", choices=["combined", "separate", "both"], default="separate"
    )
    parser.add_argument("--node-id", default="0_0_0")
    parser.add_argument("--rack-id", default="0_0")
    parser.add_argument("--row-id", default="0")
    parser.add_argument("--server-downsample-factor", type=int, default=4)
    parser.add_argument("--trend-window-s", type=int, default=900)
    args = parser.parse_args()

    result = generate_hierarchy_outputs(
        node_trace_dir=str(args.node_trace_dir),
        aggregated_dir=str(args.aggregated_dir),
        out_plot=str(args.out_plot),
        panel_out_dir=str(args.panel_out_dir),
        out_csv=str(args.out_csv),
        out_json=str(args.out_json),
        output_mode=str(args.output_mode),
        node_id=str(args.node_id),
        rack_id=str(args.rack_id),
        row_id=str(args.row_id),
        server_downsample_factor=int(args.server_downsample_factor),
        trend_window_s=int(args.trend_window_s),
    )

    print("=" * 72)
    print("Experiment 3c: Azure Hierarchy Figure Outputs")
    print("=" * 72)
    print(f"Mode                : {result['output_mode']}")
    print(f"Node source         : {result['inputs']['node_trace_path']}")
    print(f"Rack source         : {result['inputs']['rack_trace_path']}")
    print(f"Row source          : {result['inputs']['row_trace_path']}")
    print(f"Site source         : {result['inputs']['site_trace_path']}")
    if "panel_files" in result:
        for level in PANEL_ORDER:
            if level in result["panel_files"]:
                print(f"Panel [{level:>6}]      : {result['panel_files'][level]}")
    if "combined_plot_pdf" in result.get("outputs", {}):
        print(f"Combined figure      : {result['outputs']['combined_plot_pdf']}")
    elif "plot_pdf" in result.get("outputs", {}):
        print(f"Combined figure      : {result['outputs']['plot_pdf']}")
    print(f"CSV                 : {result['outputs']['csv']}")
    print(f"JSON                : {result['outputs']['json']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
