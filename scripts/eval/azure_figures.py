#!/usr/bin/env python3
"""
Experiment 2e: Generate Azure facility-scale figures.

This script generates single-panel publication figures (no subplot grids) from
existing Experiment 2 outputs and writes a manifest with derived annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COLOR_DARK = "#2c3e50"
COLOR_RED = "#e74c3c"
COLOR_ORANGE = "#e67e22"
COLOR_LIGHT_GRAY = "#bdc3c7"
COLOR_BLUE = "#1f77b4"
COLOR_MEDIUM_GRAY = "#7f8c8d"

B2_DEFAULT_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
B2_CONFIG_ID = "deepseek-r1-distill-70b_H100_tp4"
B2_DURATION_S = 86400.0
B2_DT = 0.25
B2_NON_GPU_OVERHEAD_W = 1000.0
B2_PUE = 1.3
B2_NODES_PER_RACK = 4

TRACE_KIND_ORDER = ["ours", "tdp_baseline", "mean_baseline"]
TRACE_KIND_LABEL = {
    "ours": "Ours",
    "tdp_baseline": "TDP",
    "mean_baseline": "Mean",
}
TRACE_KIND_COLOR = {
    "ours": COLOR_DARK,
    "tdp_baseline": COLOR_RED,
    "mean_baseline": COLOR_ORANGE,
}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_pdf(fig: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _safe_float(value: object, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Unable to parse float for '{field_name}': {value}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Non-finite float for '{field_name}': {value}")
    return out


def _parse_seed_csv(seed_csv: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in str(seed_csv).split(",") if p.strip() != ""]
    if len(parts) == 0:
        raise ValueError("b2 seed list is empty; provide at least one integer seed.")
    out: List[int] = []
    seen = set()
    for part in parts:
        try:
            seed = int(part)
        except Exception as exc:
            raise ValueError(f"Invalid seed '{part}' in b2 seed list: {seed_csv}") from exc
        if seed < 0:
            raise ValueError(f"Seed values must be >= 0; got {seed}.")
        if seed not in seen:
            seen.add(seed)
            out.append(seed)
    return tuple(out)


def _build_b2_generation_defaults() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "run_manifest": str(
            repo_root
            / "results"
            / "continuous_v1_gmm_bigru_sharegpt_all"
            / "kauto_max12_f2"
            / "run_manifest.json"
        ),
        "experimental_manifest": str(
            repo_root
            / "results"
            / "experimental_continuous_v1_gru_all"
            / "manifest.json"
        ),
        "throughput_db": str(repo_root / "model" / "config" / "throughput_database.json"),
        "ar1_params_dir": str(
            repo_root
            / "results"
            / "continuous_v1_gmm_bigru_sharegpt_all"
            / "kauto_max12_f2_ar1_thresh"
            / "ar1_params"
        ),
        "node_stream_dir": str(repo_root / "data" / "azure_facility" / "node_streams"),
    }


def _validate_seed_site_15min(seed: int, site_mw: np.ndarray, site_path: Path) -> np.ndarray:
    arr = np.asarray(site_mw, dtype=np.float64).reshape(-1)
    if arr.size != 96:
        raise ValueError(
            f"Expected 96 points in seed={int(seed)} cache '{site_path}', found {arr.size}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Non-finite values detected in seed={int(seed)} cache '{site_path}'."
        )
    return arr


def _generate_b2_seed_cache(
    *,
    seed: int,
    seed_dir: Path,
    rows: int,
    racks_per_row: int,
) -> None:
    # Lazy imports avoid torch-heavy module import overhead unless B2 generation is requested.
    from scripts.eval.azure_aggregate import aggregate_facility_traces
    from scripts.eval.azure_generate_traces import generate_node_traces

    defaults = _build_b2_generation_defaults()
    node_trace_dir = seed_dir / "node_traces"
    aggregated_dir = seed_dir / "aggregated"

    generate_node_traces(
        run_manifest=defaults["run_manifest"],
        experimental_manifest=defaults["experimental_manifest"],
        throughput_db=defaults["throughput_db"],
        ar1_params_dir=defaults["ar1_params_dir"],
        node_stream_dir=defaults["node_stream_dir"],
        out_dir=str(node_trace_dir),
        config_id=B2_CONFIG_ID,
        duration_s=float(B2_DURATION_S),
        dt=float(B2_DT),
        rows=int(rows),
        racks_per_row=int(racks_per_row),
        nodes_per_rack=int(B2_NODES_PER_RACK),
        base_seed=int(seed),
        device="auto",
        decode_mode="stochastic",
        median_filter_window=1,
        ours_std_scale=1.0,
        ours_logit_temperature=1.0,
    )
    aggregate_facility_traces(
        node_trace_dir=str(node_trace_dir),
        out_dir=str(aggregated_dir),
        dt=float(B2_DT),
        rows=int(rows),
        racks_per_row=int(racks_per_row),
        nodes_per_rack=int(B2_NODES_PER_RACK),
        non_gpu_overhead_w=float(B2_NON_GPU_OVERHEAD_W),
        pue=float(B2_PUE),
    )


def _load_or_generate_b2_seed_site_mw(
    *,
    seeds: Sequence[int],
    seed_cache_dir: str,
    rows: int,
    racks_per_row: int,
) -> Dict[str, object]:
    seed_series: List[np.ndarray] = []
    seed_sources: List[Dict[str, object]] = []
    cache_root = Path(seed_cache_dir)
    for seed in [int(x) for x in seeds]:
        if seed < 0:
            raise ValueError(f"Seed values must be >= 0; got {seed}.")
        seed_dir = cache_root / f"seed_{seed}"
        aggregated_dir = seed_dir / "aggregated"
        site_path = aggregated_dir / "site_15min.npy"
        source = "cache"
        if not site_path.exists():
            source = "generated"
            try:
                _generate_b2_seed_cache(
                    seed=int(seed),
                    seed_dir=seed_dir,
                    rows=int(rows),
                    racks_per_row=int(racks_per_row),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate seed cache for seed={int(seed)} at '{seed_dir}'. "
                    f"Either pre-populate '{site_path}' or ensure generation prerequisites "
                    f"(run manifest, throughput DB, and node streams) are available. "
                    f"Underlying error: {type(exc).__name__}: {exc}"
                ) from exc
        try:
            site = _load_site_15min_mw(
                aggregated_dir=str(aggregated_dir),
                power_resolution_seconds=900,
                day_seconds=86400,
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to load B2 seed cache for seed={int(seed)} from '{site_path}': {exc}"
            ) from exc
        series = _validate_seed_site_15min(
            int(seed), np.asarray(site["site_mw"], dtype=np.float64), site_path
        )
        seed_series.append(series)
        seed_sources.append(
            {
                "seed": int(seed),
                "source": source,
                "seed_dir": str(seed_dir),
                "aggregated_dir": str(aggregated_dir),
                "site_15min_path": str(site_path),
            }
        )
    if len(seed_series) == 0:
        raise ValueError("No seed series available for B2.")
    return {
        "site_mw_by_seed": np.stack(seed_series, axis=0).astype(np.float64),
        "seed_sources": seed_sources,
    }


def _compute_ldc_band_from_seed_site_mw(
    site_mw_by_seed: np.ndarray,
) -> Dict[str, np.ndarray | float | int]:
    values = np.asarray(site_mw_by_seed, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"site_mw_by_seed must be rank-2 [n_seeds, n_points], got shape {values.shape}"
        )
    if values.size == 0 or int(values.shape[1]) <= 0:
        raise ValueError("site_mw_by_seed is empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("site_mw_by_seed contains non-finite values.")

    sorted_desc = np.sort(values, axis=1)[:, ::-1]
    n_points = int(sorted_desc.shape[1])
    fraction_exceeded = (
        np.arange(n_points, dtype=np.float64) / float(max(1, n_points))
    ).astype(np.float64)
    per_seed_mean = np.mean(values, axis=1, dtype=np.float64)
    per_seed_peak = np.max(values, axis=1)
    mean_mw = float(np.median(per_seed_mean))
    peak_mw = float(np.median(per_seed_peak))
    load_factor = float(mean_mw / peak_mw) if peak_mw > 0.0 else float("nan")

    return {
        "fraction_exceeded": fraction_exceeded,
        "ldc_median_mw": np.median(sorted_desc, axis=0).astype(np.float64),
        "ldc_min_mw": np.min(sorted_desc, axis=0).astype(np.float64),
        "ldc_max_mw": np.max(sorted_desc, axis=0).astype(np.float64),
        "per_seed_mean_mw": per_seed_mean.astype(np.float64),
        "per_seed_peak_mw": per_seed_peak.astype(np.float64),
        "mean_mw": float(mean_mw),
        "peak_mw": float(peak_mw),
        "load_factor": float(load_factor),
        "n_seeds": int(sorted_desc.shape[0]),
        "n_points": int(n_points),
    }


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


def _get_site_path(aggregated_dir: str, resolution_seconds: int) -> str:
    mapping = {
        1: "site_1s.npy",
        60: "site_1min.npy",
        900: "site_15min.npy",
    }
    if int(resolution_seconds) not in mapping:
        raise ValueError(f"Unsupported site resolution: {resolution_seconds}")
    return os.path.join(aggregated_dir, mapping[int(resolution_seconds)])


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
            raise ValueError("Parsed requests CSV missing header.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Parsed requests CSV missing required columns: {missing}")

        for row_idx, row in enumerate(reader, start=2):
            t = _safe_float(row.get("arrival_time", ""), "arrival_time")
            if t < 0.0:
                raise ValueError(f"Negative arrival_time at row {row_idx}: {t}")
            b = int(np.floor(t / float(bin_seconds)))
            if b < 0 or b >= n_bins:
                # Keep strict day-window validation.
                raise ValueError(
                    f"arrival_time out of [0, {day_seconds}) at row {row_idx}: {t}"
                )
            counts[b] += 1
            n_rows += 1
            min_arrival = min(min_arrival, t)
            max_arrival = max(max_arrival, t)

    if n_rows <= 0:
        raise ValueError("Parsed requests CSV has no rows.")

    # "Near full day" validation.
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


def _load_site_15min_mw(
    aggregated_dir: str, power_resolution_seconds: int = 900, day_seconds: int = 86400
) -> Dict[str, np.ndarray]:
    if int(power_resolution_seconds) <= 0:
        raise ValueError("power_resolution_seconds must be > 0")
    if int(day_seconds) % int(power_resolution_seconds) != 0:
        raise ValueError("day_seconds must be divisible by power_resolution_seconds")
    expected_n = int(day_seconds // power_resolution_seconds)

    site_path = _get_site_path(
        aggregated_dir=aggregated_dir, resolution_seconds=power_resolution_seconds
    )
    if not os.path.exists(site_path):
        raise FileNotFoundError(f"Site trace file not found: {site_path}")
    site_w = np.asarray(np.load(site_path), dtype=np.float64).reshape(-1)
    if site_w.size != expected_n:
        raise ValueError(
            f"Expected {expected_n} points in {site_path}, found {site_w.size}"
        )
    site_mw = site_w / 1_000_000.0
    hours = (np.arange(expected_n, dtype=np.float64) + 0.5) * (
        float(power_resolution_seconds) / 3600.0
    )
    return {
        "hours": hours.astype(np.float64),
        "site_mw": site_mw.astype(np.float64),
        "site_w": site_w.astype(np.float64),
    }


def _load_metrics_15min_rows(
    metrics_csv: str,
    power_resolution_seconds: int = 900,
) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
    out: Dict[str, Dict[str, float]] = {}
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Metrics CSV missing header.")
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
            if not math.isclose(
                resolution_s,
                float(power_resolution_seconds),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                continue
            trace_kind = str(row["trace_kind"]).strip()
            if trace_kind not in TRACE_KIND_ORDER:
                continue
            out[trace_kind] = {
                "peak_kw": _safe_float(row["peak_kw"], "peak_kw"),
                "avg_kw": _safe_float(row["avg_kw"], "avg_kw"),
                "par": _safe_float(row["par"], "par"),
                "ramp_max_up_kw_per_step": _safe_float(
                    row["ramp_max_up_kw_per_step"], "ramp_max_up_kw_per_step"
                ),
                "ramp_max_down_kw_per_step": _safe_float(
                    row["ramp_max_down_kw_per_step"], "ramp_max_down_kw_per_step"
                ),
                "ldc_p99_kw": _safe_float(row["ldc_p99_kw"], "ldc_p99_kw"),
            }

    missing_kinds = [k for k in TRACE_KIND_ORDER if k not in out]
    if missing_kinds:
        raise ValueError(
            f"Missing metrics rows for resolution={power_resolution_seconds}: {missing_kinds}"
        )
    return out


def _load_ldc_15min(
    ldc_csv: str, power_resolution_seconds: int = 900
) -> Dict[str, np.ndarray]:
    if not os.path.exists(ldc_csv):
        raise FileNotFoundError(f"LDC CSV not found: {ldc_csv}")

    grouped: Dict[str, List[Tuple[int, float, float]]] = {
        k: [] for k in TRACE_KIND_ORDER
    }
    with open(ldc_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("LDC CSV missing header.")
        required_cols = {
            "trace_kind",
            "resolution_s",
            "rank",
            "fraction_exceeded",
            "power_kw",
        }
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"LDC CSV missing required columns: {missing}")

        for row in reader:
            try:
                resolution_s = float(row["resolution_s"])
            except Exception:
                continue
            if not math.isclose(
                resolution_s,
                float(power_resolution_seconds),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                continue
            trace_kind = str(row["trace_kind"]).strip()
            if trace_kind not in grouped:
                continue
            rank = int(float(row["rank"]))
            fraction_exceeded = _safe_float(
                row["fraction_exceeded"], "fraction_exceeded"
            )
            power_kw = _safe_float(row["power_kw"], "power_kw")
            grouped[trace_kind].append((rank, fraction_exceeded, power_kw))

    out: Dict[str, np.ndarray] = {}
    for trace_kind in TRACE_KIND_ORDER:
        rows = grouped[trace_kind]
        if len(rows) == 0:
            raise ValueError(
                f"Missing LDC rows for trace_kind='{trace_kind}' at resolution={power_resolution_seconds}"
            )
        rows_sorted = sorted(rows, key=lambda x: int(x[0]))
        fractions = np.asarray([x[1] for x in rows_sorted], dtype=np.float64)
        power_mw = np.asarray([x[2] / 1000.0 for x in rows_sorted], dtype=np.float64)
        out[f"{trace_kind}_fraction"] = fractions
        out[f"{trace_kind}_power_mw"] = power_mw
    return out


def _load_rack_matrix_15min_kw(
    aggregated_dir: str,
    rows: int = 10,
    racks_per_row: int = 6,
    heatmap_downsample_seconds: int = 900,
) -> Dict[str, np.ndarray]:
    if int(rows) <= 0 or int(racks_per_row) <= 0:
        raise ValueError("rows and racks_per_row must be > 0")
    if int(heatmap_downsample_seconds) <= 0:
        raise ValueError("heatmap_downsample_seconds must be > 0")
    if int(heatmap_downsample_seconds) % 1 != 0:
        raise ValueError("heatmap_downsample_seconds must be integer seconds")
    factor = int(heatmap_downsample_seconds)

    rack_series: List[np.ndarray] = []
    paths: List[str] = []
    for row_i in range(int(rows)):
        for rack_j in range(int(racks_per_row)):
            path = os.path.join(aggregated_dir, f"rack_{row_i}_{rack_j}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing rack file: {path}")
            arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
            if arr.size <= 0:
                raise ValueError(f"Empty rack trace: {path}")
            rack_series.append(arr)
            paths.append(path)

    lengths = {int(x.size) for x in rack_series}
    if len(lengths) != 1:
        raise ValueError(f"Rack traces have mismatched lengths: {sorted(lengths)}")
    n_1s = int(rack_series[0].size)
    if n_1s % factor != 0:
        raise ValueError(
            f"Rack 1s length {n_1s} is not divisible by heatmap_downsample_seconds={factor}"
        )
    n_bins = int(n_1s // factor)
    matrix = np.zeros((len(rack_series), n_bins), dtype=np.float64)
    for idx, arr in enumerate(rack_series):
        matrix[idx] = _downsample_mean(arr, factor=factor) / 1000.0  # kW

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
        raise ValueError("Ours peak/avg must be > 0 for overestimation stats.")
    over_peak_pct = (tdp_peak - ours_peak) / ours_peak * 100.0
    over_avg_pct = (tdp_avg - ours_avg) / ours_avg * 100.0
    return {
        "tdp_over_peak_pct": float(over_peak_pct),
        "tdp_over_avg_pct": float(over_avg_pct),
    }


def _compute_sizing_metrics_from_rows(
    metrics_rows_15min: Mapping[str, Mapping[str, float]],
    power_resolution_seconds: int = 900,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    scale = 3600.0 / float(power_resolution_seconds)
    for trace_kind in TRACE_KIND_ORDER:
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
    l1 = ax1.plot(
        power_hours,
        site_mw,
        color=COLOR_DARK,
        linewidth=1.8,
        label="Site Power (15-min)",
    )[0]
    ax1.set_xlim(0.0, 24.0)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Site Power (MW)", color=COLOR_DARK)
    ax1.tick_params(axis="y", colors=COLOR_DARK)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    l2 = ax2.plot(
        arrival_hours,
        arrival_rate_req_per_s,
        color=COLOR_LIGHT_GRAY,
        linewidth=1.6,
        alpha=0.95,
        label="Arrival Rate (5-min)",
    )[0]
    ax2.set_ylabel("Arrival Rate (req/s)", color=COLOR_DARK)
    ax2.tick_params(axis="y", colors=COLOR_DARK)
    ax2.grid(False)
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left")
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
        "notes": "Single panel with twin y-axis",
    }


def _plot_figure_2_baseline_comparison(
    *,
    out_path: str,
    hours: np.ndarray,
    ours_mw: np.ndarray,
    tdp_mw: float,
    mean_mw: float,
    over_peak_pct: float,
    over_avg_pct: float,
) -> Dict[str, object]:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(hours, ours_mw, color=COLOR_DARK, linewidth=1.8, label="Ours")
    ax.axhline(tdp_mw, color=COLOR_RED, linestyle="--", linewidth=1.6, label="TDP")
    ax.axhline(mean_mw, color=COLOR_ORANGE, linestyle="--", linewidth=1.6, label="Mean")

    tdp_arr = np.full_like(ours_mw, fill_value=float(tdp_mw), dtype=np.float64)
    shade_mask = tdp_arr > ours_mw
    ax.fill_between(
        hours,
        ours_mw,
        tdp_arr,
        where=shade_mask,
        color=COLOR_RED,
        alpha=0.12,
        interpolate=True,
        label="Overprovisioned Capacity",
    )
    ax.set_xlim(0.0, 24.0)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Site Power (MW)")
    ax.grid(True, alpha=0.25)

    ann = (
        f"TDP overestimates peak by {over_peak_pct:.1f}%\n"
        f"TDP overestimates average by {over_avg_pct:.1f}%"
    )
    ax.text(
        0.015,
        0.985,
        ann,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "15-minute Baseline Comparison",
        "n_points": int(ours_mw.size),
        "stats": {
            "ours_peak_mw": float(np.max(ours_mw)),
            "ours_avg_mw": float(np.mean(ours_mw)),
            "tdp_mw": float(tdp_mw),
            "mean_mw": float(mean_mw),
            "tdp_over_peak_pct": float(over_peak_pct),
            "tdp_over_avg_pct": float(over_avg_pct),
        },
        "notes": "Shaded region denotes overprovisioned capacity where TDP exceeds Ours.",
    }


def _plot_figure_3_ldc(
    *,
    out_path: str,
    ldc_data: Mapping[str, np.ndarray],
    p99_mw_by_method: Mapping[str, float],
) -> Dict[str, object]:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for trace_kind in TRACE_KIND_ORDER:
        frac = np.asarray(ldc_data[f"{trace_kind}_fraction"], dtype=np.float64)
        power_mw = np.asarray(ldc_data[f"{trace_kind}_power_mw"], dtype=np.float64)
        hours_exceeded = frac * 24.0
        ax.plot(
            hours_exceeded,
            power_mw,
            color=TRACE_KIND_COLOR[trace_kind],
            linestyle="-" if trace_kind == "ours" else "--",
            linewidth=1.7,
            label=TRACE_KIND_LABEL[trace_kind],
        )
    ax.set_xlim(0.0, 24.0)
    ax.set_xlabel("Hours Exceeded per Day")
    ax.set_ylabel("Site Power (MW)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    p99_text = "\n".join(
        [
            f"{TRACE_KIND_LABEL[k]} P99: {float(p99_mw_by_method[k]):.3f} MW"
            for k in TRACE_KIND_ORDER
        ]
    )
    ax.text(
        0.015,
        0.985,
        p99_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "Load Duration Curve (15-min)",
        "n_points_per_method": int(np.asarray(ldc_data["ours_power_mw"]).size),
        "stats": {
            "p99_mw": {k: float(v) for k, v in p99_mw_by_method.items()},
        },
        "notes": "X-axis expressed as hours exceeded per day.",
    }


def _plot_figure_b2_ldc_confidence(
    *,
    out_path: str,
    fraction_exceeded: np.ndarray,
    ldc_median_mw: np.ndarray,
    ldc_min_mw: np.ndarray,
    ldc_max_mw: np.ndarray,
    tdp_mw: float,
    mean_mw: float,
    load_factor: float,
    n_seeds: int,
) -> Dict[str, object]:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    frac = np.asarray(fraction_exceeded, dtype=np.float64).reshape(-1)
    median = np.asarray(ldc_median_mw, dtype=np.float64).reshape(-1)
    low = np.asarray(ldc_min_mw, dtype=np.float64).reshape(-1)
    high = np.asarray(ldc_max_mw, dtype=np.float64).reshape(-1)
    if not (frac.size == median.size == low.size == high.size):
        raise ValueError("B2 LDC arrays must have identical lengths.")
    if frac.size <= 0:
        raise ValueError("B2 LDC arrays are empty.")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.fill_between(
        frac,
        low,
        high,
        color=COLOR_BLUE,
        alpha=0.2,
        label=f"Min-Max Envelope ({int(n_seeds)} seeds)",
    )
    ax.plot(
        frac,
        median,
        color=COLOR_BLUE,
        linewidth=1.9,
        label="Median LDC",
    )
    ax.axhline(
        float(tdp_mw),
        color=COLOR_RED,
        linestyle="--",
        linewidth=1.6,
        label=f"TDP ({float(tdp_mw):.2f} MW)",
    )
    ax.axhline(
        float(mean_mw),
        color=COLOR_MEDIUM_GRAY,
        linestyle="--",
        linewidth=1.4,
        label=f"Mean ({float(mean_mw):.2f} MW)",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Fraction of Time Exceeded")
    ax.set_ylabel("Facility Power (MW)")
    ax.grid(True, alpha=0.25)

    ax.text(
        0.015,
        0.985,
        f"Load factor = {float(load_factor):.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "Load-Duration Curve with Confidence Band (24h, 15-min)",
        "n_points": int(frac.size),
        "stats": {
            "tdp_mw": float(tdp_mw),
            "mean_mw": float(mean_mw),
            "load_factor": float(load_factor),
            "n_seeds": int(n_seeds),
        },
        "notes": "Median line with min-max seed envelope; capacity-credit annotation omitted.",
    }


def _plot_figure_4_heatmap(
    *,
    out_path: str,
    rack_matrix_kw: np.ndarray,
    hours: np.ndarray,
    peak_window_hours: float = 3.0,
    row_limit_kw: float = 600.0,
    rack_tdp_kw: float = 26.0,
) -> Dict[str, object]:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.2)
    matrix = np.asarray(rack_matrix_kw, dtype=np.float64)
    hours_arr = np.asarray(hours, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"rack_matrix_kw must be 2D, got shape {matrix.shape}")
    if hours_arr.size != int(matrix.shape[1]):
        raise ValueError(
            f"hours length ({hours_arr.size}) must equal rack_matrix_kw timesteps ({matrix.shape[1]})"
        )
    if int(matrix.shape[1]) <= 0:
        raise ValueError("rack_matrix_kw must have at least one timestep")

    site_kw = np.sum(matrix, axis=0, dtype=np.float64)
    peak_idx = int(np.argmax(site_kw))
    if hours_arr.size > 1:
        dt_hours = float(np.median(np.diff(hours_arr)))
    else:
        dt_hours = 24.0 / float(matrix.shape[1])
    if dt_hours <= 0.0:
        raise ValueError(f"Invalid non-positive dt inferred from hours: {dt_hours}")

    half_window_bins = max(1, int(np.ceil((float(peak_window_hours) * 0.5) / dt_hours)))
    idx_lo = max(0, peak_idx - half_window_bins)
    idx_hi = min(int(matrix.shape[1]), peak_idx + half_window_bins + 1)
    peak_window_kw = np.asarray(matrix[:, idx_lo:idx_hi], dtype=np.float64)
    peak_window_hours_abs = np.asarray(hours_arr[idx_lo:idx_hi], dtype=np.float64)
    peak_window_hours_rel = peak_window_hours_abs - float(peak_window_hours_abs[0])

    def _fmt_hour(h: float) -> str:
        h_mod = float(h) % 24.0
        hh = int(np.floor(h_mod))
        mm = int(np.round((h_mod - float(hh)) * 60.0))
        if mm == 60:
            hh = (hh + 1) % 24
            mm = 0
        return f"{hh:02d}:{mm:02d}"

    vmax = float(np.percentile(peak_window_kw, 95))
    vmin = float(np.percentile(peak_window_kw, 5))
    if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
        vmin = float(np.min(peak_window_kw))
        vmax = float(np.max(peak_window_kw))
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
    ax.set_xlim(0.0, float(peak_window_hours_rel[-1] + dt_hours))
    ax.set_ylim(0.0, float(peak_window_kw.shape[0]))
    tdp_safe_racks = int(max(1, np.floor(float(row_limit_kw) / float(rack_tdp_kw))))
    peak_hour = float(hours_arr[peak_idx])
    peak_start = float(peak_window_hours_abs[0] - (dt_hours * 0.5))
    peak_end = float(peak_window_hours_abs[-1] + (dt_hours * 0.5))
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "Rack Power Heatmap (Peak-window decorrelation)",
        "n_points": {
            "racks": int(matrix.shape[0]),
            "time_bins": int(matrix.shape[1]),
            "peak_window_bins": int(peak_window_kw.shape[1]),
        },
        "stats": {
            "min_kw": float(np.min(matrix)),
            "max_kw": float(np.max(matrix)),
            "mean_kw": float(np.mean(matrix)),
            "peak_hour_of_day": float(peak_hour),
            "peak_window_start_hour": float(peak_start),
            "peak_window_end_hour": float(peak_end),
            "row_limit_kw": float(row_limit_kw),
            "rack_tdp_kw": float(rack_tdp_kw),
            "tdp_safe_racks": int(tdp_safe_racks),
        },
        "notes": "Peak-centered window to emphasize temporal decorrelation across racks.",
    }


def _plot_figure_5_sizing_bars(
    *,
    out_path: str,
    sizing_metrics: Mapping[str, Mapping[str, float]],
) -> Dict[str, object]:
    categories = ["Peak (MW)", "PAR", "Max Ramp (MW/hr)"]
    x = np.arange(len(categories), dtype=np.float64)
    width = 0.24
    offsets = {
        "tdp_baseline": -width,
        "mean_baseline": 0.0,
        "ours": width,
    }
    method_order = ["tdp_baseline", "mean_baseline", "ours"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for method in method_order:
        values = np.asarray(
            [
                float(sizing_metrics[method]["peak_mw"]),
                float(sizing_metrics[method]["par"]),
                float(sizing_metrics[method]["max_ramp_mw_per_hr"]),
            ],
            dtype=np.float64,
        )
        bars = ax.bar(
            x + offsets[method],
            values,
            width=width,
            color=TRACE_KIND_COLOR[method],
            alpha=0.9,
            label=TRACE_KIND_LABEL[method],
        )
        for b in bars:
            h = float(b.get_height())
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h + (0.02 * max(1.0, float(np.max(values)))),
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Metric Value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_pdf(fig, out_path)
    return {
        "file": str(out_path),
        "title": "Infrastructure Sizing Metrics",
        "n_points": int(len(categories) * len(method_order)),
        "stats": {
            "methods": {
                m: {
                    "peak_mw": float(sizing_metrics[m]["peak_mw"]),
                    "par": float(sizing_metrics[m]["par"]),
                    "max_ramp_mw_per_hr": float(
                        sizing_metrics[m]["max_ramp_mw_per_hr"]
                    ),
                }
                for m in method_order
            }
        },
        "notes": "Max ramp derived from 15-min ramp basis.",
    }


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "figures"
    return {
        "parsed_requests_csv": str(
            repo_root
            / "data"
            / "azure_trace"
            / "parsed"
            / "day_2024-05-16_requests.csv"
        ),
        "aggregated_dir": str(repo_root / "results" / "azure_facility" / "aggregated"),
        "metrics_csv": str(
            repo_root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        ),
        "ldc_csv": str(
            repo_root / "results" / "eval_paper" / "azure_facility_ldc_15min.csv"
        ),
        "out_dir": str(out_dir),
        "manifest_path": str(out_dir / "azure_figure_manifest.json"),
    }


def generate_azure_figures(
    *,
    parsed_requests_csv: str,
    aggregated_dir: str,
    metrics_csv: str,
    ldc_csv: str,
    out_dir: str,
    arrival_bin_seconds: int = 300,
    power_resolution_seconds: int = 900,
    heatmap_downsample_seconds: int = 300,
    peak_window_hours: float = 3.0,
    rows: int = 10,
    racks_per_row: int = 6,
    dry_run: bool = False,
    include_figure_b2: bool = False,
    b2_seeds: Sequence[int] = B2_DEFAULT_SEEDS,
    b2_seed_cache_dir: str = "results/azure_facility/seed_runs",
) -> Dict[str, object]:
    if int(power_resolution_seconds) != 900:
        raise ValueError(
            "power_resolution_seconds is fixed to 900 for Experiment 2e plots."
        )
    seeds = tuple(int(x) for x in b2_seeds)
    if bool(include_figure_b2) and len(seeds) == 0:
        raise ValueError("include_figure_b2=True requires at least one seed.")

    apply_publication_style()
    os.makedirs(out_dir, exist_ok=True)

    arrivals = _load_arrival_rate_binned(
        parsed_requests_csv=parsed_requests_csv,
        bin_seconds=int(arrival_bin_seconds),
        day_seconds=86400,
    )
    site = _load_site_15min_mw(
        aggregated_dir=aggregated_dir,
        power_resolution_seconds=int(power_resolution_seconds),
        day_seconds=86400,
    )
    metrics_rows = _load_metrics_15min_rows(
        metrics_csv=metrics_csv,
        power_resolution_seconds=int(power_resolution_seconds),
    )
    ldc_data = _load_ldc_15min(
        ldc_csv=ldc_csv,
        power_resolution_seconds=int(power_resolution_seconds),
    )
    racks = _load_rack_matrix_15min_kw(
        aggregated_dir=aggregated_dir,
        rows=int(rows),
        racks_per_row=int(racks_per_row),
        heatmap_downsample_seconds=int(heatmap_downsample_seconds),
    )

    # Additional hard validation: ensure LDC and site point counts are aligned to a day.
    n_power = int(site["site_mw"].size)
    n_ldc = int(np.asarray(ldc_data["ours_power_mw"]).size)
    if n_power != n_ldc:
        raise ValueError(
            f"LDC rows ({n_ldc}) do not match site_15min points ({n_power}) for Ours."
        )

    baseline_stats = _compute_baseline_comparison_stats(metrics_rows)
    sizing_metrics = _compute_sizing_metrics_from_rows(
        metrics_rows,
        power_resolution_seconds=int(power_resolution_seconds),
    )
    p99_mw_by_method = {
        k: float(metrics_rows[k]["ldc_p99_kw"]) / 1000.0 for k in TRACE_KIND_ORDER
    }
    b2_seed_payload: Optional[Dict[str, object]] = None
    b2_ldc_payload: Optional[Dict[str, np.ndarray | float | int]] = None
    if bool(include_figure_b2):
        b2_seed_payload = _load_or_generate_b2_seed_site_mw(
            seeds=seeds,
            seed_cache_dir=str(b2_seed_cache_dir),
            rows=int(rows),
            racks_per_row=int(racks_per_row),
        )
        b2_ldc_payload = _compute_ldc_band_from_seed_site_mw(
            np.asarray(b2_seed_payload["site_mw_by_seed"], dtype=np.float64)
        )

    paths = {
        "figure_1": str(Path(out_dir) / "azure_figure_1_diurnal_profile.pdf"),
        "figure_2": str(Path(out_dir) / "azure_figure_2_baseline_comparison_15min.pdf"),
        "figure_3": str(Path(out_dir) / "azure_figure_3_load_duration_curve.pdf"),
        "figure_4": str(Path(out_dir) / "azure_figure_4_rack_heatmap.pdf"),
        "figure_5": str(Path(out_dir) / "azure_figure_5_sizing_metrics.pdf"),
        "manifest": str(Path(out_dir) / "azure_figure_manifest.json"),
    }
    if bool(include_figure_b2):
        paths["figure_b2"] = str(
            Path(out_dir) / "azure_figure_b2_load_duration_confidence_band.pdf"
        )

    figb2_meta: Optional[Dict[str, object]] = None

    if dry_run:
        fig1_meta = {
            "file": paths["figure_1"],
            "title": "24-hour Site Power with Arrival Overlay",
            "n_points": {
                "site_power": int(site["site_mw"].size),
                "arrival_rate": int(arrivals["rate_req_per_s"].size),
            },
            "stats": {
                "site_peak_mw": float(np.max(site["site_mw"])),
                "site_avg_mw": float(np.mean(site["site_mw"])),
                "arrival_peak_req_per_s": float(np.max(arrivals["rate_req_per_s"])),
                "arrival_avg_req_per_s": float(np.mean(arrivals["rate_req_per_s"])),
            },
            "notes": "Dry-run metadata only.",
        }
        fig2_meta = {
            "file": paths["figure_2"],
            "title": "15-minute Baseline Comparison",
            "n_points": int(site["site_mw"].size),
            "stats": {
                "tdp_over_peak_pct": float(baseline_stats["tdp_over_peak_pct"]),
                "tdp_over_avg_pct": float(baseline_stats["tdp_over_avg_pct"]),
            },
            "notes": "Dry-run metadata only.",
        }
        fig3_meta = {
            "file": paths["figure_3"],
            "title": "Load Duration Curve (15-min)",
            "n_points_per_method": int(np.asarray(ldc_data["ours_power_mw"]).size),
            "stats": {"p99_mw": {k: float(v) for k, v in p99_mw_by_method.items()}},
            "notes": "Dry-run metadata only.",
        }
        fig4_meta = {
            "file": paths["figure_4"],
            "title": "Rack Power Heatmap (Azure-driven)",
            "n_points": {
                "racks": int(racks["matrix_kw"].shape[0]),
                "time_bins": int(racks["matrix_kw"].shape[1]),
            },
            "stats": {
                "min_kw": float(np.min(racks["matrix_kw"])),
                "max_kw": float(np.max(racks["matrix_kw"])),
                "mean_kw": float(np.mean(racks["matrix_kw"])),
            },
            "notes": "Dry-run metadata only.",
        }
        fig5_meta = {
            "file": paths["figure_5"],
            "title": "Infrastructure Sizing Metrics",
            "n_points": 9,
            "stats": {
                "methods": {
                    k: {kk: float(vv) for kk, vv in sizing_metrics[k].items()}
                    for k in TRACE_KIND_ORDER
                }
            },
            "notes": "Dry-run metadata only.",
        }
        if bool(include_figure_b2):
            assert b2_ldc_payload is not None
            assert b2_seed_payload is not None
            figb2_meta = {
                "file": paths["figure_b2"],
                "title": "Load-Duration Curve with Confidence Band (24h, 15-min)",
                "n_points": int(b2_ldc_payload["n_points"]),
                "stats": {
                    "tdp_mw": float(metrics_rows["tdp_baseline"]["peak_kw"]) / 1000.0,
                    "mean_mw": float(b2_ldc_payload["mean_mw"]),
                    "load_factor": float(b2_ldc_payload["load_factor"]),
                    "n_seeds": int(b2_ldc_payload["n_seeds"]),
                },
                "band_type": "min_max",
                "seeds": [int(x) for x in seeds],
                "seed_sources": b2_seed_payload["seed_sources"],
                "notes": "Dry-run metadata only.",
            }
    else:
        fig1_meta = _plot_figure_1_diurnal_overlay(
            out_path=paths["figure_1"],
            power_hours=np.asarray(site["hours"], dtype=np.float64),
            site_mw=np.asarray(site["site_mw"], dtype=np.float64),
            arrival_hours=np.asarray(arrivals["hours"], dtype=np.float64),
            arrival_rate_req_per_s=np.asarray(
                arrivals["rate_req_per_s"], dtype=np.float64
            ),
        )
        fig2_meta = _plot_figure_2_baseline_comparison(
            out_path=paths["figure_2"],
            hours=np.asarray(site["hours"], dtype=np.float64),
            ours_mw=np.asarray(site["site_mw"], dtype=np.float64),
            tdp_mw=float(metrics_rows["tdp_baseline"]["peak_kw"]) / 1000.0,
            mean_mw=float(metrics_rows["mean_baseline"]["peak_kw"]) / 1000.0,
            over_peak_pct=float(baseline_stats["tdp_over_peak_pct"]),
            over_avg_pct=float(baseline_stats["tdp_over_avg_pct"]),
        )
        fig3_meta = _plot_figure_3_ldc(
            out_path=paths["figure_3"],
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
            sizing_metrics=sizing_metrics,
        )
        if bool(include_figure_b2):
            assert b2_ldc_payload is not None
            assert b2_seed_payload is not None
            figb2_meta = _plot_figure_b2_ldc_confidence(
                out_path=paths["figure_b2"],
                fraction_exceeded=np.asarray(
                    b2_ldc_payload["fraction_exceeded"], dtype=np.float64
                ),
                ldc_median_mw=np.asarray(b2_ldc_payload["ldc_median_mw"], dtype=np.float64),
                ldc_min_mw=np.asarray(b2_ldc_payload["ldc_min_mw"], dtype=np.float64),
                ldc_max_mw=np.asarray(b2_ldc_payload["ldc_max_mw"], dtype=np.float64),
                tdp_mw=float(metrics_rows["tdp_baseline"]["peak_kw"]) / 1000.0,
                mean_mw=float(b2_ldc_payload["mean_mw"]),
                load_factor=float(b2_ldc_payload["load_factor"]),
                n_seeds=int(b2_ldc_payload["n_seeds"]),
            )
            figb2_meta["band_type"] = "min_max"
            figb2_meta["seeds"] = [int(x) for x in seeds]
            figb2_meta["seed_sources"] = b2_seed_payload["seed_sources"]

    manifest: Dict[str, object] = {
        "schema_version": "azure-figures-v1",
        "inputs": {
            "parsed_requests_csv": str(Path(parsed_requests_csv).resolve()),
            "aggregated_dir": str(Path(aggregated_dir).resolve()),
            "metrics_csv": str(Path(metrics_csv).resolve()),
            "ldc_csv": str(Path(ldc_csv).resolve()),
        },
        "config": {
            "arrival_bin_seconds": int(arrival_bin_seconds),
            "power_resolution_seconds": int(power_resolution_seconds),
            "heatmap_downsample_seconds": int(heatmap_downsample_seconds),
            "peak_window_hours": float(peak_window_hours),
            "rows": int(rows),
            "racks_per_row": int(racks_per_row),
            "include_figure_b2": bool(include_figure_b2),
            "b2_seeds": [int(x) for x in seeds],
            "b2_seed_cache_dir": str(b2_seed_cache_dir),
            "style": {
                "color_dark": COLOR_DARK,
                "color_red": COLOR_RED,
                "color_orange": COLOR_ORANGE,
                "color_light_gray": COLOR_LIGHT_GRAY,
                "color_blue": COLOR_BLUE,
                "color_medium_gray": COLOR_MEDIUM_GRAY,
            },
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
            "p99_mw_by_method": {k: float(v) for k, v in p99_mw_by_method.items()},
            "sizing_metrics": {
                k: {kk: float(vv) for kk, vv in sizing_metrics[k].items()}
                for k in TRACE_KIND_ORDER
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
    if bool(include_figure_b2):
        if figb2_meta is None:
            raise RuntimeError("Figure B2 was requested but metadata was not generated.")
        manifest["figures"]["figure_b2_load_duration_confidence"] = figb2_meta
        if b2_ldc_payload is not None:
            manifest["derived_metrics"]["figure_b2"] = {
                "n_seeds": int(b2_ldc_payload["n_seeds"]),
                "n_points": int(b2_ldc_payload["n_points"]),
                "mean_mw": float(b2_ldc_payload["mean_mw"]),
                "peak_mw": float(b2_ldc_payload["peak_mw"]),
                "load_factor": float(b2_ldc_payload["load_factor"]),
                "tdp_mw": float(metrics_rows["tdp_baseline"]["peak_kw"]) / 1000.0,
                "band_type": "min_max",
            }
        manifest["output_paths"]["figure_b2_load_duration_confidence"] = paths[
            "figure_b2"
        ]
    _write_json(paths["manifest"], manifest)
    return manifest


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Generate Azure facility figures from Experiment 2 outputs."
    )
    parser.add_argument(
        "--parsed-requests-csv", default=defaults["parsed_requests_csv"]
    )
    parser.add_argument("--aggregated-dir", default=defaults["aggregated_dir"])
    parser.add_argument("--metrics-csv", default=defaults["metrics_csv"])
    parser.add_argument("--ldc-csv", default=defaults["ldc_csv"])
    parser.add_argument("--out-dir", default=defaults["out_dir"])
    parser.add_argument("--arrival-bin-seconds", type=int, default=300)
    parser.add_argument("--power-resolution-seconds", type=int, default=900)
    parser.add_argument("--heatmap-downsample-seconds", type=int, default=300)
    parser.add_argument("--peak-window-hours", type=float, default=3.0)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--include-figure-b2", action="store_true")
    parser.add_argument(
        "--b2-seeds",
        default=",".join([str(x) for x in B2_DEFAULT_SEEDS]),
        help="Comma-separated seed list for Figure B2 (e.g., 42,43,44,45,46).",
    )
    parser.add_argument(
        "--b2-seed-cache-dir",
        default="results/azure_facility/seed_runs",
        help="Cache directory for per-seed node/aggregate outputs used by Figure B2.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    b2_seeds = _parse_seed_csv(str(args.b2_seeds))

    run = generate_azure_figures(
        parsed_requests_csv=str(args.parsed_requests_csv),
        aggregated_dir=str(args.aggregated_dir),
        metrics_csv=str(args.metrics_csv),
        ldc_csv=str(args.ldc_csv),
        out_dir=str(args.out_dir),
        arrival_bin_seconds=int(args.arrival_bin_seconds),
        power_resolution_seconds=int(args.power_resolution_seconds),
        heatmap_downsample_seconds=int(args.heatmap_downsample_seconds),
        peak_window_hours=float(args.peak_window_hours),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        dry_run=bool(args.dry_run),
        include_figure_b2=bool(args.include_figure_b2),
        b2_seeds=b2_seeds,
        b2_seed_cache_dir=str(args.b2_seed_cache_dir),
    )
    if bool(args.dry_run):
        print("[azure_figures] Dry run complete")
    else:
        print("[azure_figures] Done")
    print(f"  manifest: {run['output_paths']['manifest']}")
    for key in [
        "figure_1_diurnal_profile",
        "figure_2_baseline_comparison_15min",
        "figure_3_load_duration_curve",
        "figure_4_rack_heatmap",
        "figure_5_sizing_metrics",
    ]:
        print(f"  {key}: {run['output_paths'][key]}")
    if "figure_b2_load_duration_confidence" in run["output_paths"]:
        print(
            "  figure_b2_load_duration_confidence: "
            f"{run['output_paths']['figure_b2_load_duration_confidence']}"
        )


if __name__ == "__main__":
    main()
