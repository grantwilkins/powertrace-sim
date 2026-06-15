#!/usr/bin/env python3
"""
Experiment 2e: Generate Azure facility-scale figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.azure_defaults import (
    DEFAULT_TRACE_KINDS,
    build_default_paths,
    parse_csv_list,
    safe_float,
    write_json,
)

COLOR_DARK = "#2c3e50"
COLOR_RED = "#e74c3c"
COLOR_ORANGE = "#e67e22"
COLOR_TEAL = "#17becf"
COLOR_LIGHT_GRAY = "#bdc3c7"

TRACE_KIND_LABEL = {
    "ours": "Ours",
    "tdp_baseline": "TDP",
    "mean_baseline": "Mean",
    "splitwise_strict": "Splitwise",
}
TRACE_KIND_COLOR = {
    "ours": COLOR_DARK,
    "tdp_baseline": COLOR_RED,
    "mean_baseline": COLOR_ORANGE,
    "splitwise_strict": COLOR_TEAL,
}
TRACE_KIND_LINESTYLE = {
    "ours": "-",
    "tdp_baseline": "--",
    "mean_baseline": "--",
    "splitwise_strict": ":",
}


def apply_publication_style() -> None:
    plt.rcParams.update({"axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def save_pdf(fig: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _normalize_trace_kinds(trace_kinds: Sequence[str] | str) -> List[str]:
    values = (
        parse_csv_list(trace_kinds)
        if isinstance(trace_kinds, str)
        else [str(x).strip() for x in trace_kinds]
    )
    out: List[str] = []
    allowed = set(DEFAULT_TRACE_KINDS)
    for value in values:
        if not value:
            continue
        if value not in allowed:
            raise ValueError(
                f"Unsupported trace_kind '{value}'. Allowed: {sorted(allowed)}"
            )
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("No trace kinds selected")
    if "ours" not in out:
        raise ValueError("trace_kinds must include 'ours' for figure overlays")
    return out


def _downsample_mean(values: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    f = int(factor)
    if f <= 0:
        raise ValueError("downsample factor must be >= 1")
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if arr.size % f != 0:
        raise ValueError(f"Array length {arr.size} is not divisible by factor {f}")
    return np.mean(arr.reshape(-1, f), axis=1).astype(np.float64)


def _load_arrival_rate_binned(
    parsed_requests_csv: str,
    bin_seconds: int = 300,
    day_seconds: int = 86400,
) -> Dict[str, np.ndarray]:
    if not os.path.exists(parsed_requests_csv):
        raise FileNotFoundError(f"Parsed requests CSV not found: {parsed_requests_csv}")
    if int(bin_seconds) <= 0:
        raise ValueError("bin_seconds must be > 0")
    if int(day_seconds) <= 0:
        raise ValueError("day_seconds must be > 0")
    if int(day_seconds) % int(bin_seconds) != 0:
        raise ValueError("day_seconds must be divisible by bin_seconds")

    n_bins = int(day_seconds // bin_seconds)
    counts = np.zeros((n_bins,), dtype=np.int64)
    required = {"arrival_time", "n_in", "n_out"}

    n_rows = 0
    min_arrival = float("inf")
    max_arrival = float("-inf")
    with open(parsed_requests_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Parsed requests CSV missing header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Parsed requests CSV missing required columns: {missing}")

        for row_idx, row in enumerate(reader, start=2):
            arrival = safe_float(row.get("arrival_time", ""), "arrival_time")
            if arrival < 0.0:
                raise ValueError(f"Negative arrival_time at row {row_idx}: {arrival}")
            bucket = int(np.floor(arrival / float(bin_seconds)))
            if bucket < 0 or bucket >= n_bins:
                raise ValueError(
                    f"arrival_time out of [0, {day_seconds}) at row {row_idx}: {arrival}"
                )
            counts[bucket] += 1
            n_rows += 1
            min_arrival = min(min_arrival, arrival)
            max_arrival = max(max_arrival, arrival)

    if n_rows <= 0:
        raise ValueError("Parsed requests CSV has no rows")
    if max_arrival < float(day_seconds - 300):
        raise ValueError(
            f"Parsed request horizon appears too short for a full day: max arrival={max_arrival:.3f}s"
        )

    rate_req_per_s = counts.astype(np.float64) / float(bin_seconds)
    hours = (np.arange(n_bins, dtype=np.float64) + 0.5) * (float(bin_seconds) / 3600.0)
    return {
        "hours": hours.astype(np.float64),
        "rate_req_per_s": rate_req_per_s.astype(np.float64),
        "counts": counts.astype(np.int64),
        "num_requests": np.asarray([n_rows], dtype=np.int64),
        "min_arrival": np.asarray([min_arrival], dtype=np.float64),
        "max_arrival": np.asarray([max_arrival], dtype=np.float64),
    }


def _load_metrics_15min_rows(
    metrics_csv: str, trace_kinds: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    wanted = set(trace_kinds)
    out: Dict[str, Dict[str, float]] = {}
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Metrics CSV missing header")
        required_cols = {
            "trace_kind",
            "resolution_s",
            "peak_kw",
            "avg_kw",
            "par",
            "ramp_max_up_kw_per_step",
            "ramp_max_down_kw_per_step",
            "ldc_p99_kw",
        }
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Metrics CSV missing required columns: {missing}")

        for row in reader:
            try:
                resolution_s = float(row["resolution_s"])
            except Exception:
                continue
            if not math.isclose(resolution_s, 900.0, rel_tol=0.0, abs_tol=1e-9):
                continue
            trace_kind = str(row["trace_kind"]).strip()
            if trace_kind not in wanted:
                continue
            out[trace_kind] = {
                "peak_kw": safe_float(row["peak_kw"], "peak_kw"),
                "avg_kw": safe_float(row["avg_kw"], "avg_kw"),
                "par": safe_float(row["par"], "par"),
                "ramp_max_up_kw_per_step": safe_float(
                    row["ramp_max_up_kw_per_step"], "ramp_max_up_kw_per_step"
                ),
                "ramp_max_down_kw_per_step": safe_float(
                    row["ramp_max_down_kw_per_step"], "ramp_max_down_kw_per_step"
                ),
                "ldc_p99_kw": safe_float(row["ldc_p99_kw"], "ldc_p99_kw"),
            }

    missing_kinds = [kind for kind in trace_kinds if kind not in out]
    if missing_kinds:
        raise ValueError(f"Missing metrics rows for 15-min resolution: {missing_kinds}")
    return out


def _load_ldc_15min(ldc_csv: str, trace_kinds: Sequence[str]) -> Dict[str, np.ndarray]:
    if not os.path.exists(ldc_csv):
        raise FileNotFoundError(f"LDC CSV not found: {ldc_csv}")

    grouped: Dict[str, List[tuple[int, float, float]]] = {
        kind: [] for kind in trace_kinds
    }
    with open(ldc_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("LDC CSV missing header")
        for row in reader:
            try:
                resolution_s = float(row["resolution_s"])
            except Exception:
                continue
            if not math.isclose(resolution_s, 900.0, rel_tol=0.0, abs_tol=1e-9):
                continue
            trace_kind = str(row["trace_kind"]).strip()
            if trace_kind not in grouped:
                continue
            rank = int(float(row["rank"]))
            fraction_exceeded = safe_float(
                row["fraction_exceeded"], "fraction_exceeded"
            )
            power_kw = safe_float(row["power_kw"], "power_kw")
            grouped[trace_kind].append((rank, fraction_exceeded, power_kw))

    out: Dict[str, np.ndarray] = {}
    for trace_kind in trace_kinds:
        rows = grouped[trace_kind]
        if len(rows) == 0:
            raise ValueError(f"Missing LDC rows for trace_kind='{trace_kind}'")
        rows_sorted = sorted(rows, key=lambda x: int(x[0]))
        out[f"{trace_kind}_fraction"] = np.asarray(
            [row[1] for row in rows_sorted], dtype=np.float64
        )
        out[f"{trace_kind}_power_mw"] = np.asarray(
            [row[2] / 1000.0 for row in rows_sorted], dtype=np.float64
        )
    return out


def _load_site_traces_15min(
    site_traces_15min_csv: str, trace_kinds: Sequence[str]
) -> Dict[str, np.ndarray]:
    if not os.path.exists(site_traces_15min_csv):
        raise FileNotFoundError(f"Site traces CSV not found: {site_traces_15min_csv}")

    grouped: Dict[str, List[tuple[int, float, float]]] = {
        kind: [] for kind in trace_kinds
    }
    with open(site_traces_15min_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Site traces CSV missing header")
        required = {"trace_kind", "bin_idx", "hour", "power_mw"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Site traces CSV missing required columns: {missing}")
        for row in reader:
            kind = str(row.get("trace_kind", "")).strip()
            if kind not in grouped:
                continue
            idx = int(float(row["bin_idx"]))
            hour = safe_float(row["hour"], "hour")
            power_mw = safe_float(row["power_mw"], "power_mw")
            grouped[kind].append((idx, hour, power_mw))

    out: Dict[str, np.ndarray] = {}
    for kind in trace_kinds:
        rows = grouped[kind]
        if len(rows) == 0:
            raise ValueError(f"Missing site 15-min rows for trace_kind='{kind}'")
        rows_sorted = sorted(rows, key=lambda x: int(x[0]))
        out[f"{kind}_hours"] = np.asarray(
            [row[1] for row in rows_sorted], dtype=np.float64
        )
        out[f"{kind}_power_mw"] = np.asarray(
            [row[2] for row in rows_sorted], dtype=np.float64
        )
    return out


def _load_rack_matrix_15min_kw(
    aggregated_root: str,
    rows: int = 10,
    racks_per_row: int = 6,
    heatmap_downsample_seconds: int = 900,
    method: str = "ours",
) -> Dict[str, np.ndarray]:
    method_dir = os.path.join(aggregated_root, method)
    if int(rows) <= 0 or int(racks_per_row) <= 0:
        raise ValueError("rows and racks_per_row must be > 0")
    factor = int(heatmap_downsample_seconds)
    if factor <= 0:
        raise ValueError("heatmap_downsample_seconds must be > 0")

    rack_series: List[np.ndarray] = []
    paths: List[str] = []
    for row_i in range(int(rows)):
        for rack_j in range(int(racks_per_row)):
            path = os.path.join(method_dir, f"rack_{row_i}_{rack_j}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing rack file: {path}")
            arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
            if arr.size <= 0:
                raise ValueError(f"Empty rack trace: {path}")
            rack_series.append(arr)
            paths.append(path)

    lengths = {int(arr.size) for arr in rack_series}
    if len(lengths) != 1:
        raise ValueError(f"Rack traces have mismatched lengths: {sorted(lengths)}")
    n_1s = int(rack_series[0].size)
    if n_1s % factor != 0:
        raise ValueError(
            f"Rack 1s length {n_1s} is not divisible by downsample factor={factor}"
        )

    n_bins = int(n_1s // factor)
    matrix = np.zeros((len(rack_series), n_bins), dtype=np.float64)
    for idx, arr in enumerate(rack_series):
        matrix[idx] = _downsample_mean(arr, factor=factor) / 1000.0

    hours = (np.arange(n_bins, dtype=np.float64) + 0.5) * (float(factor) / 3600.0)
    return {
        "matrix_kw": matrix.astype(np.float64),
        "hours": hours.astype(np.float64),
        "paths": np.asarray(paths, dtype=object),
    }


def _compute_baseline_comparison_stats(
    metrics_rows_15min: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    ours_peak = float(metrics_rows_15min["ours"]["peak_kw"])
    ours_avg = float(metrics_rows_15min["ours"]["avg_kw"])
    tdp_peak = float(metrics_rows_15min["tdp_baseline"]["peak_kw"])
    tdp_avg = float(metrics_rows_15min["tdp_baseline"]["avg_kw"])
    if ours_peak <= 0.0 or ours_avg <= 0.0:
        raise ValueError("Ours peak/avg must be > 0 for overestimation stats")
    return {
        "tdp_over_peak_pct": float((tdp_peak - ours_peak) / ours_peak * 100.0),
        "tdp_over_avg_pct": float((tdp_avg - ours_avg) / ours_avg * 100.0),
    }


def _compute_splitwise_vs_ours_stats(
    metrics_rows_15min: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    ours_peak = float(metrics_rows_15min["ours"]["peak_kw"])
    ours_avg = float(metrics_rows_15min["ours"]["avg_kw"])
    out: Dict[str, float] = {}
    if "splitwise_strict" in metrics_rows_15min:
        out["splitwise_strict_over_peak_pct_vs_ours"] = float(
            (float(metrics_rows_15min["splitwise_strict"]["peak_kw"]) - ours_peak)
            / ours_peak
            * 100.0
        )
        out["splitwise_strict_over_avg_pct_vs_ours"] = float(
            (float(metrics_rows_15min["splitwise_strict"]["avg_kw"]) - ours_avg)
            / ours_avg
            * 100.0
        )
    return out


def _compute_sizing_metrics_from_rows(
    metrics_rows_15min: Mapping[str, Mapping[str, float]],
    trace_kinds: Sequence[str],
    power_resolution_seconds: int = 900,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    scale = 3600.0 / float(power_resolution_seconds)
    for trace_kind in trace_kinds:
        row = metrics_rows_15min[trace_kind]
        ramp_up = float(row["ramp_max_up_kw_per_step"])
        ramp_down = float(row["ramp_max_down_kw_per_step"])
        max_abs_kw_step = float(max(abs(ramp_up), abs(ramp_down)))
        out[trace_kind] = {
            "peak_mw": float(row["peak_kw"]) / 1000.0,
            "par": float(row["par"]),
            "max_ramp_mw_per_hr": (max_abs_kw_step / 1000.0) * scale,
            "max_abs_kw_per_step": max_abs_kw_step,
        }
    return out


def _plot_figure_1_diurnal_overlay(
    *,
    out_path: str,
    power_hours: np.ndarray,
    site_mw: np.ndarray,
    arrival_hours: np.ndarray,
    arrival_rate_req_per_s: np.ndarray,
) -> Dict[str, object]:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.0)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    line1 = ax1.plot(
        power_hours, site_mw, color="#B1040E", linewidth=1.8, label="Site Power"
    )[0]
    ax1.set_xlim(0.0, 24.0)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Site Power (MW)", color="#B1040E")
    ax1.tick_params(axis="y", colors="#B1040E")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        arrival_hours,
        arrival_rate_req_per_s,
        color=COLOR_DARK,
        linewidth=1.6,
        alpha=0.95,
        label="Arrival Rate",
    )[0]
    ax2.set_ylabel("Arrival Rate (req/s)", color=COLOR_DARK)
    ax2.tick_params(axis="y", colors=COLOR_DARK)
    ax2.set_ylim(0.0, 100.0)
    ax2.grid(False)
    ax1.legend(
        [line1, line2],
        [line1.get_label(), line2.get_label()],
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
    )
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "24-hour Site Power with Arrival Overlay",
        "n_points": {
            "site_power": int(site_mw.size),
            "arrival_rate": int(arrival_rate_req_per_s.size),
        },
        "stats": {
            "site_peak_mw": float(np.max(site_mw)),
            "site_avg_mw": float(np.mean(site_mw)),
            "arrival_peak_req_per_s": float(np.max(arrival_rate_req_per_s)),
            "arrival_avg_req_per_s": float(np.mean(arrival_rate_req_per_s)),
        },
        "notes": "Ours only for the diurnal overlay.",
    }


def _plot_figure_2_baseline_comparison(
    *,
    out_path: str,
    trace_kinds: Sequence[str],
    site_traces: Mapping[str, np.ndarray],
    baseline_stats: Mapping[str, float],
    splitwise_vs_ours: Mapping[str, float],
) -> Dict[str, object]:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 3.9))

    ours_hours = np.asarray(site_traces["ours_hours"], dtype=np.float64)
    ours_mw = np.asarray(site_traces["ours_power_mw"], dtype=np.float64)

    for kind in trace_kinds:
        ax.plot(
            np.asarray(site_traces[f"{kind}_hours"], dtype=np.float64),
            np.asarray(site_traces[f"{kind}_power_mw"], dtype=np.float64),
            color=TRACE_KIND_COLOR[kind],
            linestyle=TRACE_KIND_LINESTYLE[kind],
            linewidth=1.7,
            label=TRACE_KIND_LABEL[kind],
        )

    if "tdp_baseline" in trace_kinds:
        tdp_mw = np.asarray(site_traces["tdp_baseline_power_mw"], dtype=np.float64)
        shade_mask = tdp_mw > ours_mw
        ax.fill_between(
            ours_hours,
            ours_mw,
            tdp_mw,
            where=shade_mask,
            color=COLOR_RED,
            alpha=0.10,
            interpolate=True,
            label="TDP Overprovisioned",
        )

    ann_lines = [
        f"TDP over peak: {float(baseline_stats['tdp_over_peak_pct']):.1f}%",
        f"TDP over avg: {float(baseline_stats['tdp_over_avg_pct']):.1f}%",
    ]
    if "splitwise_strict_over_peak_pct_vs_ours" in splitwise_vs_ours:
        ann_lines.append(
            f"Splitwise vs Ours peak: {float(splitwise_vs_ours['splitwise_strict_over_peak_pct_vs_ours']):.1f}%"
        )

    ax.text(
        0.015,
        0.985,
        "\n".join(ann_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    ax.set_xlim(0.0, 24.0)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Site Power (MW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "15-minute Baseline Comparison",
        "n_points": int(ours_mw.size),
        "notes": "All selected trace kinds plotted.",
    }


def _plot_figure_3_ldc(
    *,
    out_path: str,
    trace_kinds: Sequence[str],
    ldc_data: Mapping[str, np.ndarray],
    p99_mw_by_method: Mapping[str, float],
) -> Dict[str, object]:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for trace_kind in trace_kinds:
        fraction = np.asarray(ldc_data[f"{trace_kind}_fraction"], dtype=np.float64)
        power_mw = np.asarray(ldc_data[f"{trace_kind}_power_mw"], dtype=np.float64)
        hours_exceeded = fraction * 24.0
        ax.plot(
            hours_exceeded,
            power_mw,
            color=TRACE_KIND_COLOR[trace_kind],
            linestyle=TRACE_KIND_LINESTYLE[trace_kind],
            linewidth=1.7,
            label=TRACE_KIND_LABEL[trace_kind],
        )
    ax.set_xlim(0.0, 24.0)
    ax.set_xlabel("Hours Exceeded per Day")
    ax.set_ylabel("Site Power (MW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

    p99_text = "\n".join(
        [
            f"{TRACE_KIND_LABEL[k]} P99: {float(p99_mw_by_method[k]):.3f} MW"
            for k in trace_kinds
        ]
    )
    ax.text(
        0.015,
        0.985,
        p99_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "Load Duration Curve (15-min)",
        "n_points_per_method": int(
            np.asarray(ldc_data[f"{trace_kinds[0]}_power_mw"]).size
        ),
        "stats": {"p99_mw": {k: float(v) for k, v in p99_mw_by_method.items()}},
    }


def _plot_figure_4_heatmap(
    *,
    out_path: str,
    rack_matrix_kw: np.ndarray,
    hours: np.ndarray,
    peak_window_hours: float = 3.0,
) -> Dict[str, object]:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.1)
    matrix = np.asarray(rack_matrix_kw, dtype=np.float64)
    hours_arr = np.asarray(hours, dtype=np.float64).reshape(-1)

    site_kw = np.sum(matrix, axis=0, dtype=np.float64)
    peak_idx = int(np.argmax(site_kw))
    dt_hours = (
        float(np.median(np.diff(hours_arr)))
        if hours_arr.size > 1
        else 24.0 / float(matrix.shape[1])
    )
    half_window_bins = max(1, int(np.ceil((float(peak_window_hours) * 0.5) / dt_hours)))
    idx_lo = max(0, peak_idx - half_window_bins)
    idx_hi = min(int(matrix.shape[1]), peak_idx + half_window_bins + 1)
    peak_window_kw = np.asarray(matrix[:, idx_lo:idx_hi], dtype=np.float64)
    peak_window_hours_rel = np.asarray(hours_arr[idx_lo:idx_hi], dtype=np.float64)
    peak_window_hours_rel = peak_window_hours_rel - float(peak_window_hours_rel[0])

    vmin = float(np.percentile(peak_window_kw, 5))
    vmax = float(np.percentile(peak_window_kw, 95))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.max(peak_window_kw))
        vmin = float(np.min(peak_window_kw))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(vmin + 1.0)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(
        peak_window_kw,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            0.0,
            float(peak_window_hours_rel[-1] + dt_hours),
            0.0,
            float(peak_window_kw.shape[0]),
        ],
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rack Power (kW)")
    ax.set_xlabel("Time in Peak Window (hours)")
    ax.set_ylabel("Rack ID")
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "Rack Power Heatmap (Ours peak window)",
        "n_points": {
            "racks": int(matrix.shape[0]),
            "time_bins": int(matrix.shape[1]),
            "peak_window_bins": int(peak_window_kw.shape[1]),
        },
    }


def _plot_figure_5_sizing_bars(
    *,
    out_path: str,
    trace_kinds: Sequence[str],
    sizing_metrics: Mapping[str, Mapping[str, float]],
) -> Dict[str, object]:
    categories = ["Peak (MW)", "PAR", "Max Ramp (MW/hr)"]
    x = np.arange(len(categories), dtype=np.float64)
    width = 0.8 / max(1, len(trace_kinds))
    offset_start = -0.4 + (width / 2.0)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for idx, method in enumerate(trace_kinds):
        values = np.asarray(
            [
                float(sizing_metrics[method]["peak_mw"]),
                float(sizing_metrics[method]["par"]),
                float(sizing_metrics[method]["max_ramp_mw_per_hr"]),
            ],
            dtype=np.float64,
        )
        xpos = x + (offset_start + idx * width)
        bars = ax.bar(
            xpos,
            values,
            width=width,
            color=TRACE_KIND_COLOR[method],
            alpha=0.9,
            label=TRACE_KIND_LABEL[method],
        )
        ypad = 0.02 * max(1.0, float(np.max(values)))
        for bar in bars:
            height = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + ypad,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Metric Value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "Infrastructure Sizing Metrics",
        "n_points": int(len(categories) * len(trace_kinds)),
    }


def generate_azure_figures(
    *,
    parsed_requests_csv: str,
    aggregated_root: str,
    metrics_csv: str,
    ldc_csv: str,
    site_traces_15min_csv: str,
    out_dir: str,
    trace_kinds: Sequence[str] | str = ",".join(DEFAULT_TRACE_KINDS),
    arrival_bin_seconds: int = 300,
    power_resolution_seconds: int = 900,
    heatmap_downsample_seconds: int = 300,
    peak_window_hours: float = 3.0,
    rows: int = 10,
    racks_per_row: int = 6,
    dry_run: bool = False,
) -> Dict[str, object]:
    if int(power_resolution_seconds) != 900:
        raise ValueError(
            "power_resolution_seconds is fixed to 900 for figure generation"
        )

    trace_order = _normalize_trace_kinds(trace_kinds)
    apply_publication_style()
    os.makedirs(out_dir, exist_ok=True)

    arrivals = _load_arrival_rate_binned(
        parsed_requests_csv=parsed_requests_csv,
        bin_seconds=int(arrival_bin_seconds),
        day_seconds=86400,
    )
    metrics_rows = _load_metrics_15min_rows(metrics_csv, trace_order)
    ldc_data = _load_ldc_15min(ldc_csv, trace_order)
    site_traces = _load_site_traces_15min(site_traces_15min_csv, trace_order)
    racks = _load_rack_matrix_15min_kw(
        aggregated_root=aggregated_root,
        rows=int(rows),
        racks_per_row=int(racks_per_row),
        heatmap_downsample_seconds=int(heatmap_downsample_seconds),
        method="ours",
    )

    n_power = int(np.asarray(site_traces["ours_power_mw"]).size)
    n_ldc = int(np.asarray(ldc_data[f"{trace_order[0]}_power_mw"]).size)
    if n_power != n_ldc:
        raise ValueError(
            f"LDC rows ({n_ldc}) do not match site 15-min points ({n_power})"
        )

    baseline_stats = _compute_baseline_comparison_stats(metrics_rows)
    splitwise_vs_ours = _compute_splitwise_vs_ours_stats(metrics_rows)
    sizing_metrics = _compute_sizing_metrics_from_rows(
        metrics_rows,
        trace_order,
        power_resolution_seconds=900,
    )
    p99_mw_by_method = {
        kind: float(metrics_rows[kind]["ldc_p99_kw"]) / 1000.0 for kind in trace_order
    }

    paths = {
        "figure_1": str(Path(out_dir) / "azure_figure_1_diurnal_profile.pdf"),
        "figure_2": str(Path(out_dir) / "azure_figure_2_baseline_comparison_15min.pdf"),
        "figure_3": str(Path(out_dir) / "azure_figure_3_load_duration_curve.pdf"),
        "figure_4": str(Path(out_dir) / "azure_figure_4_rack_heatmap.pdf"),
        "figure_5": str(Path(out_dir) / "azure_figure_5_sizing_metrics.pdf"),
        "manifest": str(Path(out_dir) / "azure_figure_manifest.json"),
    }

    if dry_run:
        fig1_meta = {
            "file": paths["figure_1"],
            "title": "24-hour Site Power with Arrival Overlay",
            "notes": "Dry-run",
        }
        fig2_meta = {
            "file": paths["figure_2"],
            "title": "15-minute Baseline Comparison",
            "notes": "Dry-run",
        }
        fig3_meta = {
            "file": paths["figure_3"],
            "title": "Load Duration Curve (15-min)",
            "notes": "Dry-run",
        }
        fig4_meta = {
            "file": paths["figure_4"],
            "title": "Rack Power Heatmap",
            "notes": "Dry-run",
        }
        fig5_meta = {
            "file": paths["figure_5"],
            "title": "Infrastructure Sizing Metrics",
            "notes": "Dry-run",
        }
    else:
        fig1_meta = _plot_figure_1_diurnal_overlay(
            out_path=paths["figure_1"],
            power_hours=np.asarray(site_traces["ours_hours"], dtype=np.float64),
            site_mw=np.asarray(site_traces["ours_power_mw"], dtype=np.float64),
            arrival_hours=np.asarray(arrivals["hours"], dtype=np.float64),
            arrival_rate_req_per_s=np.asarray(
                arrivals["rate_req_per_s"], dtype=np.float64
            ),
        )
        fig2_meta = _plot_figure_2_baseline_comparison(
            out_path=paths["figure_2"],
            trace_kinds=trace_order,
            site_traces=site_traces,
            baseline_stats=baseline_stats,
            splitwise_vs_ours=splitwise_vs_ours,
        )
        fig3_meta = _plot_figure_3_ldc(
            out_path=paths["figure_3"],
            trace_kinds=trace_order,
            ldc_data=ldc_data,
            p99_mw_by_method=p99_mw_by_method,
        )
        fig4_meta = _plot_figure_4_heatmap(
            out_path=paths["figure_4"],
            rack_matrix_kw=np.asarray(racks["matrix_kw"], dtype=np.float64),
            hours=np.asarray(racks["hours"], dtype=np.float64),
            peak_window_hours=float(peak_window_hours),
        )
        fig5_meta = _plot_figure_5_sizing_bars(
            out_path=paths["figure_5"],
            trace_kinds=trace_order,
            sizing_metrics=sizing_metrics,
        )

    manifest: Dict[str, object] = {
        "schema_version": "azure-figures-v2",
        "inputs": {
            "parsed_requests_csv": str(Path(parsed_requests_csv).resolve()),
            "aggregated_root": str(Path(aggregated_root).resolve()),
            "metrics_csv": str(Path(metrics_csv).resolve()),
            "ldc_csv": str(Path(ldc_csv).resolve()),
            "site_traces_15min_csv": str(Path(site_traces_15min_csv).resolve()),
        },
        "config": {
            "trace_kinds": list(trace_order),
            "arrival_bin_seconds": int(arrival_bin_seconds),
            "power_resolution_seconds": int(power_resolution_seconds),
            "heatmap_downsample_seconds": int(heatmap_downsample_seconds),
            "peak_window_hours": float(peak_window_hours),
            "rows": int(rows),
            "racks_per_row": int(racks_per_row),
            "dry_run": bool(dry_run),
        },
        "figures": {
            "figure_1_diurnal_profile": fig1_meta,
            "figure_2_baseline_comparison_15min": fig2_meta,
            "figure_3_load_duration_curve": fig3_meta,
            "figure_4_rack_heatmap": fig4_meta,
            "figure_5_sizing_metrics": fig5_meta,
        },
        "derived_metrics": {
            "tdp_over_peak_pct": float(baseline_stats["tdp_over_peak_pct"]),
            "tdp_over_avg_pct": float(baseline_stats["tdp_over_avg_pct"]),
            "splitwise_vs_ours": {
                key: float(value) for key, value in splitwise_vs_ours.items()
            },
            "p99_mw_by_method": {
                key: float(value) for key, value in p99_mw_by_method.items()
            },
            "sizing_metrics": {
                method: {
                    metric: float(value)
                    for metric, value in sizing_metrics[method].items()
                }
                for method in trace_order
            },
        },
        "output_paths": {
            "figure_1_diurnal_profile": paths["figure_1"],
            "figure_2_baseline_comparison_15min": paths["figure_2"],
            "figure_3_load_duration_curve": paths["figure_3"],
            "figure_4_rack_heatmap": paths["figure_4"],
            "figure_5_sizing_metrics": paths["figure_5"],
            "manifest": paths["manifest"],
        },
    }
    write_json(paths["manifest"], manifest)
    return manifest


def main() -> None:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(
        description="Generate Azure facility figures from top-level outputs."
    )
    parser.add_argument(
        "--parsed-requests-csv", default=defaults["parsed_requests_csv"]
    )
    parser.add_argument("--aggregated-root", default=defaults["aggregated_root"])
    parser.add_argument("--metrics-csv", default=defaults["metrics_csv"])
    parser.add_argument("--ldc-csv", default=defaults["ldc_csv"])
    parser.add_argument(
        "--site-traces-15min-csv", default=defaults["site_traces_15min_csv"]
    )
    parser.add_argument("--out-dir", default=defaults["figures_out_dir"])
    parser.add_argument("--trace-kinds", default=",".join(DEFAULT_TRACE_KINDS))
    parser.add_argument("--arrival-bin-seconds", type=int, default=300)
    parser.add_argument("--power-resolution-seconds", type=int, default=900)
    parser.add_argument("--heatmap-downsample-seconds", type=int, default=300)
    parser.add_argument("--peak-window-hours", type=float, default=3.0)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run = generate_azure_figures(
        parsed_requests_csv=str(args.parsed_requests_csv),
        aggregated_root=str(args.aggregated_root),
        metrics_csv=str(args.metrics_csv),
        ldc_csv=str(args.ldc_csv),
        site_traces_15min_csv=str(args.site_traces_15min_csv),
        out_dir=str(args.out_dir),
        trace_kinds=str(args.trace_kinds),
        arrival_bin_seconds=int(args.arrival_bin_seconds),
        power_resolution_seconds=int(args.power_resolution_seconds),
        heatmap_downsample_seconds=int(args.heatmap_downsample_seconds),
        peak_window_hours=float(args.peak_window_hours),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        dry_run=bool(args.dry_run),
    )
    print(
        "[azure_figures] Dry run complete"
        if bool(args.dry_run)
        else "[azure_figures] Done"
    )
    print(f"  manifest: {run['output_paths']['manifest']}")


if __name__ == "__main__":
    main()
