#!/usr/bin/env python3
"""
Azure oversubscription capacity analysis and figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    DEFAULT_OVERSUB_METHODS,
    build_default_paths,
    ensure_dir_for_file,
    parse_csv_list,
    safe_float,
    write_json,
)

METHOD_LABEL = {
    "mean_baseline": "Mean",
    "splitwise_strict": "Splitwise",
    "ours": "Ours",
}
METHOD_COLOR = {
    "mean_baseline": "#e67e22",
    "splitwise_strict": "#17becf",
    "ours": "#2c3e50",
}
METHOD_LINESTYLE = {
    "mean_baseline": "--",
    "splitwise_strict": ":",
    "ours": "-",
}
TRACE_BACKED_METHODS = {"splitwise_strict", "ours"}
SYNTHETIC_METHODS = {"mean_baseline"}
ALLOWED_METHODS = TRACE_BACKED_METHODS | SYNTHETIC_METHODS


def _validate_positive(name: str, value: float) -> float:
    v = float(value)
    if (not np.isfinite(v)) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return float(v)


def _normalize_methods(methods: Sequence[str] | str) -> List[str]:
    values = (
        parse_csv_list(methods)
        if isinstance(methods, str)
        else [str(x).strip() for x in methods]
    )
    out: List[str] = []
    for value in values:
        if not value:
            continue
        if value not in ALLOWED_METHODS:
            raise ValueError(
                f"Unsupported method '{value}'. Allowed: {sorted(ALLOWED_METHODS)}"
            )
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("No methods selected")
    return out


def _load_method_rack_matrix_kw(aggregated_root: str, method: str) -> np.ndarray:
    method_dir = Path(aggregated_root) / str(method)
    paths = sorted(method_dir.glob("rack_*_*.npy"))
    if len(paths) <= 0:
        raise FileNotFoundError(
            f"No rack traces found for method '{method}' in {method_dir}"
        )

    traces: List[np.ndarray] = []
    for path in paths:
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if arr.size <= 0:
            raise ValueError(f"Empty rack trace: {path}")
        traces.append(arr / 1000.0)

    lengths = sorted({int(arr.size) for arr in traces})
    if len(lengths) != 1:
        raise ValueError(f"Rack trace length mismatch for method '{method}': {lengths}")
    return np.stack(traces, axis=0).astype(np.float64)


def _pick_reference_trace_method(
    aggregated_root: str,
    methods: Sequence[str],
) -> Tuple[str, np.ndarray]:
    candidates = [method for method in methods if method in TRACE_BACKED_METHODS]
    if "ours" in candidates:
        candidates = ["ours"] + [method for method in candidates if method != "ours"]
    for method in candidates:
        try:
            return method, _load_method_rack_matrix_kw(aggregated_root, method)
        except Exception:
            continue
    raise ValueError("At least one trace-backed method with rack traces is required")


def _load_constant_site_kw(metrics_csv: str, trace_kind: str) -> float:
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    preferred: Optional[float] = None
    fallback: Optional[float] = None
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Metrics CSV missing header")
        required = {"trace_kind", "resolution_s", "avg_kw"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Metrics CSV missing required columns: {sorted(missing)}")

        for row in reader:
            kind = str(row.get("trace_kind", "")).strip()
            if kind != str(trace_kind):
                continue
            avg_kw = safe_float(row.get("avg_kw", ""), "avg_kw")
            fallback = avg_kw
            try:
                resolution_s = float(row.get("resolution_s", "nan"))
            except Exception:
                resolution_s = float("nan")
            if math.isclose(resolution_s, 900.0, rel_tol=0.0, abs_tol=1e-9):
                preferred = avg_kw
                break

    if preferred is not None:
        return float(preferred)
    if fallback is not None:
        return float(fallback)
    raise ValueError(
        f"No metrics row found for trace_kind='{trace_kind}' in {metrics_csv}"
    )


def _build_mean_rack_matrix_kw(
    *,
    metrics_csv: str,
    n_racks: int,
    timesteps: int,
) -> np.ndarray:
    if int(n_racks) <= 0:
        raise ValueError("n_racks must be >= 1")
    if int(timesteps) <= 0:
        raise ValueError("timesteps must be >= 1")
    site_mean_kw = _load_constant_site_kw(metrics_csv, trace_kind="mean_baseline")
    rack_mean_kw = float(site_mean_kw) / float(n_racks)
    return np.full(
        (int(n_racks), int(timesteps)), fill_value=float(rack_mean_kw), dtype=np.float64
    )


def _load_method_matrices_kw(
    *,
    aggregated_root: str,
    metrics_csv: str,
    methods: Sequence[str],
) -> Tuple[str, Dict[str, np.ndarray]]:
    reference_method, reference_matrix = _pick_reference_trace_method(
        aggregated_root, methods
    )
    n_racks = int(reference_matrix.shape[0])
    timesteps = int(reference_matrix.shape[1])
    out: Dict[str, np.ndarray] = {}
    for method in methods:
        if method in TRACE_BACKED_METHODS:
            matrix = _load_method_rack_matrix_kw(aggregated_root, method)
            if matrix.shape != reference_matrix.shape:
                raise ValueError(
                    f"Rack matrix shape mismatch for method '{method}': {matrix.shape} != {reference_matrix.shape}"
                )
            out[method] = matrix
        elif method == "mean_baseline":
            out[method] = _build_mean_rack_matrix_kw(
                metrics_csv=metrics_csv,
                n_racks=n_racks,
                timesteps=timesteps,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    return reference_method, out


def _sample_subset_indices(
    *,
    n_total: int,
    n_select: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    n_total_i = int(n_total)
    n_select_i = int(n_select)
    n_samples_i = int(n_samples)
    if n_total_i <= 0:
        raise ValueError("n_total must be >= 1")
    if n_select_i <= 0:
        raise ValueError("n_select must be >= 1")
    if n_samples_i <= 0:
        raise ValueError("n_samples must be >= 1")
    use_replacement = bool(n_select_i > n_total_i)
    sample_mode = (
        "with_replacement_extrapolated"
        if use_replacement
        else "without_replacement_observed"
    )
    if (not use_replacement) and n_select_i == n_total_i:
        return np.tile(np.arange(n_total_i, dtype=np.int64), (1, 1)), sample_mode

    out = np.zeros((n_samples_i, n_select_i), dtype=np.int64)
    for idx in range(n_samples_i):
        out[idx, :] = np.asarray(
            rng.choice(n_total_i, size=n_select_i, replace=use_replacement),
            dtype=np.int64,
        )
    return out, sample_mode


def _aggregate_peaks_for_count(
    *,
    rack_matrix_kw: np.ndarray,
    n_select: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    idx, sample_mode = _sample_subset_indices(
        n_total=int(rack_matrix_kw.shape[0]),
        n_select=int(n_select),
        n_samples=int(n_samples),
        rng=rng,
    )
    peaks = np.zeros((int(idx.shape[0]),), dtype=np.float64)
    for row_idx in range(int(idx.shape[0])):
        trace = np.sum(rack_matrix_kw[idx[row_idx, :], :], axis=0, dtype=np.float64)
        peaks[row_idx] = float(np.max(trace))
    return {
        "peaks_kw": peaks,
        "samples": int(idx.shape[0]),
        "sample_mode": str(sample_mode),
        "is_extrapolated": bool(int(n_select) > int(rack_matrix_kw.shape[0])),
    }


def _sample_traces_for_count(
    *,
    rack_matrix_kw: np.ndarray,
    n_select: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx, _ = _sample_subset_indices(
        n_total=int(rack_matrix_kw.shape[0]),
        n_select=int(n_select),
        n_samples=int(n_samples),
        rng=rng,
    )
    traces = np.zeros(
        (int(idx.shape[0]), int(rack_matrix_kw.shape[1])), dtype=np.float64
    )
    for row_idx in range(int(idx.shape[0])):
        traces[row_idx, :] = np.sum(
            rack_matrix_kw[idx[row_idx, :], :], axis=0, dtype=np.float64
        )
    return traces


def compute_oversubscription_capacity(
    *,
    aggregated_root: str,
    metrics_csv: str,
    out_capacity_plot: str,
    out_lines_plot: str,
    out_csv: str,
    out_json: str,
    methods: Sequence[str] | str = ",".join(DEFAULT_OVERSUB_METHODS),
    row_limit_kw: float = 600.0,
    rack_tdp_kw: float = 26.0,
    risk_percentile: float = 95.0,
    seed: int = 42,
    samples_per_count: int = 200,
    trace_samples: int = 80,
    tdp_racks: Optional[int] = None,
    max_racks_to_evaluate: Optional[int] = None,
) -> Dict[str, object]:
    selected_methods = _normalize_methods(methods)
    row_limit = _validate_positive("row_limit_kw", float(row_limit_kw))
    rack_tdp = _validate_positive("rack_tdp_kw", float(rack_tdp_kw))
    risk_p = float(risk_percentile)
    if (not np.isfinite(risk_p)) or risk_p <= 0.0 or risk_p > 100.0:
        raise ValueError(f"risk_percentile must be in (0,100], got {risk_percentile}")
    if int(samples_per_count) <= 0:
        raise ValueError("samples_per_count must be >= 1")
    if int(trace_samples) <= 0:
        raise ValueError("trace_samples must be >= 1")

    reference_method, method_matrices_kw = _load_method_matrices_kw(
        aggregated_root=aggregated_root,
        metrics_csv=metrics_csv,
        methods=selected_methods,
    )
    reference_matrix = method_matrices_kw[reference_method]
    n_racks_available = int(reference_matrix.shape[0])
    timesteps = int(reference_matrix.shape[1])
    if max_racks_to_evaluate is None:
        max_eval_racks = int(max(2 * n_racks_available, n_racks_available))
    else:
        max_eval_racks = int(max_racks_to_evaluate)
    if max_eval_racks < n_racks_available:
        raise ValueError(
            "max_racks_to_evaluate must be >= available racks "
            f"({n_racks_available}), got {max_eval_racks}"
        )

    if tdp_racks is None:
        tdp_count = int(max(1, min(max_eval_racks, np.floor(row_limit / rack_tdp))))
    else:
        tdp_count = int(tdp_racks)
        if tdp_count <= 0 or tdp_count > max_eval_racks:
            raise ValueError(
                f"tdp_racks must be in [1,{max_eval_racks}], got {tdp_racks}"
            )
    tdp_row_kw = float(tdp_count) * float(rack_tdp)

    rows: List[Dict[str, object]] = []
    peak_samples_by_method_count: Dict[str, Dict[int, np.ndarray]] = {}
    selection_by_method: Dict[str, Dict[str, object]] = {}

    for method_idx, method in enumerate(selected_methods):
        rng_method = np.random.default_rng(int(seed) + (method_idx * 10007))
        rack_matrix_kw = method_matrices_kw[method]
        by_count: Dict[int, np.ndarray] = {}
        method_rows: List[Dict[str, object]] = []

        def _append_count_summary(n: int) -> Dict[str, object]:
            sample = _aggregate_peaks_for_count(
                rack_matrix_kw=rack_matrix_kw,
                n_select=int(n),
                n_samples=int(samples_per_count),
                rng=rng_method,
            )
            peaks_kw = np.asarray(sample["peaks_kw"], dtype=np.float64)
            by_count[int(n)] = peaks_kw
            row = {
                "method": str(method),
                "n_racks": int(n),
                "n_samples": int(peaks_kw.size),
                "peak_mean_kw": float(np.mean(peaks_kw)),
                "peak_p05_kw": float(np.percentile(peaks_kw, 5)),
                "peak_p50_kw": float(np.percentile(peaks_kw, 50)),
                "peak_p95_kw": float(np.percentile(peaks_kw, 95)),
                "peak_prisk_kw": float(np.percentile(peaks_kw, risk_p)),
                "peak_max_kw": float(np.max(peaks_kw)),
                "exceed_prob": float(np.mean(peaks_kw > row_limit)),
                "sample_mode": str(sample["sample_mode"]),
                "is_extrapolated": bool(sample["is_extrapolated"]),
            }
            method_rows.append(row)
            return row

        for count in range(1, n_racks_available + 1):
            _append_count_summary(count)

        if max_eval_racks > n_racks_available and float(
            method_rows[-1]["peak_prisk_kw"]
        ) <= float(row_limit):
            for count in range(n_racks_available + 1, max_eval_racks + 1):
                row = _append_count_summary(count)
                if float(row["peak_prisk_kw"]) > float(row_limit):
                    break

        valid_counts = [
            int(row["n_racks"])
            for row in method_rows
            if float(row["peak_prisk_kw"]) <= float(row_limit)
        ]
        if len(valid_counts) > 0:
            oversub_count = int(max(valid_counts))
            if int(method_rows[-1]["n_racks"]) == int(max_eval_racks) and float(
                method_rows[-1]["peak_prisk_kw"]
            ) <= float(row_limit):
                note = "hit_evaluation_cap"
            elif int(oversub_count) > int(n_racks_available):
                note = "auto_from_risk_rule_extrapolated"
            else:
                note = "auto_from_risk_rule"
        else:
            oversub_count = 1
            note = "no_count_met_risk_rule_clamped_to_1"

        rows.extend(method_rows)
        peak_samples_by_method_count[method] = by_count
        selection_row = next(
            row for row in method_rows if int(row["n_racks"]) == int(oversub_count)
        )
        selection_by_method[method] = {
            "oversub_racks": int(oversub_count),
            "oversub_note": str(note),
            "valid_counts_under_risk_rule": [int(value) for value in valid_counts],
            "peak_prisk_kw": float(selection_row["peak_prisk_kw"]),
            "peak_p50_kw": float(selection_row["peak_p50_kw"]),
            "used_extrapolation": bool(int(oversub_count) > int(n_racks_available)),
            "evaluated_through_n_racks": int(method_rows[-1]["n_racks"]),
        }

    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1.15)

    fig_a, ax_a = plt.subplots(figsize=(10.0, 5.0))
    max_evaluated_racks = max(
        int(selection_by_method[m]["evaluated_through_n_racks"])
        for m in selected_methods
    )
    for method in selected_methods:
        method_rows = [row for row in rows if str(row["method"]) == str(method)]
        x = np.asarray([float(row["n_racks"]) for row in method_rows], dtype=np.float64)
        y = np.asarray(
            [float(row["peak_prisk_kw"]) for row in method_rows], dtype=np.float64
        )
        ax_a.plot(
            x,
            y,
            color=METHOD_COLOR[method],
            linestyle=METHOD_LINESTYLE[method],
            linewidth=2.4,
            label=METHOD_LABEL[method],
        )
        selected_count = int(selection_by_method[method]["oversub_racks"])
        selected_y = float(selection_by_method[method]["peak_prisk_kw"])
        ax_a.scatter(
            np.asarray([selected_count], dtype=np.float64),
            np.asarray([selected_y], dtype=np.float64),
            color=METHOD_COLOR[method],
            s=48,
            zorder=4,
            alpha=0.7,
        )
        ax_a.annotate(
            f"N={selected_count}",
            (float(selected_count), float(selected_y)),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=10,
            color=METHOD_COLOR[method],
        )

    ax_a.axhline(
        row_limit,
        color="#000000",
        linewidth=1.2,
        linestyle="--",
        alpha=0.75,
        label=f"Row limit ({row_limit:.0f} kW)",
    )
    ax_a.axvline(
        tdp_count,
        color="#c0392b",
        linewidth=1.4,
        linestyle=":",
        alpha=0.85,
        label=f"TDP-safe N={tdp_count}",
    )
    ax_a.set_xlabel("Rack count (N)")
    ax_a.set_ylabel(f"P{int(risk_p)} Peak Row Power (kW)")
    ax_a.grid(True, alpha=0.25, linestyle=":")
    ax_a.set_xlim(1.0, float(max_evaluated_racks))
    if max_evaluated_racks > n_racks_available:
        ax_a.axvline(
            n_racks_available,
            color="#7f8c8d",
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
            label=f"Observed racks={n_racks_available}",
        )
    ax_a.legend(loc="best")
    fig_a.tight_layout()
    ensure_dir_for_file(out_capacity_plot)
    fig_a.savefig(out_capacity_plot, bbox_inches="tight")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(8.8, 5.1))
    hours = np.arange(timesteps, dtype=np.float64) / 3600.0
    for method_idx, method in enumerate(selected_methods):
        rng_method = np.random.default_rng(int(seed) + 500000 + (method_idx * 10007))
        selected_count = int(selection_by_method[method]["oversub_racks"])
        traces = _sample_traces_for_count(
            rack_matrix_kw=method_matrices_kw[method],
            n_select=int(selected_count),
            n_samples=int(trace_samples),
            rng=rng_method,
        )
        p05 = np.percentile(traces, 5, axis=0)
        p50 = np.percentile(traces, 50, axis=0)
        p95 = np.percentile(traces, 95, axis=0)
        ax_b.fill_between(
            hours,
            p05,
            p95,
            color=METHOD_COLOR[method],
            alpha=0.10 if method != "mean_baseline" else 0.06,
            linewidth=0.0,
        )
        ax_b.plot(
            hours,
            p50,
            color=METHOD_COLOR[method],
            linestyle=METHOD_LINESTYLE[method],
            linewidth=2.3,
            label=f"{METHOD_LABEL[method]} (N={selected_count})",
            alpha=0.7,
        )

    ax_b.axhline(
        tdp_row_kw,
        color="#c0392b",
        linewidth=1.5,
        linestyle="-.",
        alpha=0.9,
        label=f"TDP (N={tdp_count})",
        zorder=3,
    )
    ax_b.axhline(
        row_limit,
        color="#000000",
        linewidth=1.2,
        linestyle="--",
        alpha=0.75,
        label="Row limit",
    )
    ax_b.set_xlabel("Time (hours)")
    ax_b.set_ylabel("Row Power (kW)")
    ax_b.grid(True, alpha=0.25, linestyle=":")
    ax_b.set_xlim(0.0, float(hours[-1]) if hours.size > 1 else 1.0)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=2)
    fig_b.tight_layout()
    ensure_dir_for_file(out_lines_plot)
    fig_b.savefig(out_lines_plot, bbox_inches="tight")
    plt.close(fig_b)

    ensure_dir_for_file(out_csv)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n_racks",
                "n_samples",
                "peak_mean_kw",
                "peak_p05_kw",
                "peak_p50_kw",
                "peak_p95_kw",
                "peak_prisk_kw",
                "peak_max_kw",
                "exceed_prob",
                "sample_mode",
                "is_extrapolated",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "ok",
        "aggregated_root": str(aggregated_root),
        "metrics_csv": str(metrics_csv),
        "inputs": {
            "methods": list(selected_methods),
            "row_limit_kw": float(row_limit),
            "rack_tdp_kw": float(rack_tdp),
            "risk_percentile": float(risk_p),
            "seed": int(seed),
            "samples_per_count": int(samples_per_count),
            "trace_samples": int(trace_samples),
            "max_racks_to_evaluate": int(max_eval_racks),
        },
        "dataset": {
            "n_racks_available": int(n_racks_available),
            "max_racks_evaluated": int(max_evaluated_racks),
            "timesteps_1s": int(timesteps),
            "reference_method": str(reference_method),
        },
        "selection": {
            "tdp_racks": int(tdp_count),
            "tdp_row_kw": float(tdp_row_kw),
            "selection_by_method": selection_by_method,
        },
        "summary_rows": rows,
        "outputs": {
            "capacity_plot_pdf": str(out_capacity_plot),
            "lines_plot_pdf": str(out_lines_plot),
            "csv": str(out_csv),
            "json": str(out_json),
        },
    }
    write_json(out_json, payload)
    return payload


def main() -> None:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(
        description="Azure oversubscription capacity analysis from aggregated rack traces."
    )
    parser.add_argument("--aggregated-root", default=defaults["aggregated_root"])
    parser.add_argument("--metrics-csv", default=defaults["metrics_csv"])
    parser.add_argument("--methods", default=",".join(DEFAULT_OVERSUB_METHODS))
    parser.add_argument("--row-limit-kw", type=float, default=600.0)
    parser.add_argument("--rack-tdp-kw", type=float, default=26.0)
    parser.add_argument("--risk-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-count", type=int, default=200)
    parser.add_argument("--trace-samples", type=int, default=80)
    parser.add_argument("--tdp-racks", type=int, default=None)
    parser.add_argument("--max-racks-to-evaluate", type=int, default=None)
    parser.add_argument(
        "--out-capacity-plot", default=defaults["oversub_capacity_plot"]
    )
    parser.add_argument("--out-lines-plot", default=defaults["oversub_lines_plot"])
    parser.add_argument("--out-csv", default=defaults["oversub_csv"])
    parser.add_argument("--out-json", default=defaults["oversub_json"])
    args = parser.parse_args()

    result = compute_oversubscription_capacity(
        aggregated_root=str(args.aggregated_root),
        metrics_csv=str(args.metrics_csv),
        out_capacity_plot=str(args.out_capacity_plot),
        out_lines_plot=str(args.out_lines_plot),
        out_csv=str(args.out_csv),
        out_json=str(args.out_json),
        methods=str(args.methods),
        row_limit_kw=float(args.row_limit_kw),
        rack_tdp_kw=float(args.rack_tdp_kw),
        risk_percentile=float(args.risk_percentile),
        seed=int(args.seed),
        samples_per_count=int(args.samples_per_count),
        trace_samples=int(args.trace_samples),
        tdp_racks=args.tdp_racks,
        max_racks_to_evaluate=args.max_racks_to_evaluate,
    )

    print("=" * 72)
    print("Azure Oversubscription Capacity")
    print("=" * 72)
    print(f"Aggregated root: {result['aggregated_root']}")
    print(f"Methods       : {', '.join(result['inputs']['methods'])}")
    print(f"Available racks: {result['dataset']['n_racks_available']}")
    print(f"Evaluated to N : {result['dataset']['max_racks_evaluated']}")
    print(f"TDP-safe N     : {result['selection']['tdp_racks']}")
    for method in result["inputs"]["methods"]:
        selection = result["selection"]["selection_by_method"][method]
        print(f"  {METHOD_LABEL[method]:18s} N={selection['oversub_racks']}")
    print(f"Capacity plot  : {result['outputs']['capacity_plot_pdf']}")
    print(f"Lines plot     : {result['outputs']['lines_plot_pdf']}")
    print(f"CSV            : {result['outputs']['csv']}")
    print(f"JSON           : {result['outputs']['json']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
