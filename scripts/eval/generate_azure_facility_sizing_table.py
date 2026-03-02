#!/usr/bin/env python3
"""
Generate paper-ready infrastructure sizing table values from Azure facility metrics.

This script reads 15-minute facility metrics (Ours / TDP / Mean), computes
table values, and writes:
  - CSV summary
  - JSON summary (with raw numeric metrics)
  - LaTeX table snippet

Optional: recompute metrics directly from Azure aggregated traces before table
generation by calling scripts/eval/azure_metrics.py logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.azure_metrics import compute_azure_facility_metrics

TRACE_KIND_ORDER = ("tdp_baseline", "mean_baseline", "ours")
TRACE_KIND_HEADER = {
    "tdp_baseline": "TDP",
    "mean_baseline": "Mean",
    "ours": "Ours",
}


def _ensure_dir_for_file(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _safe_float(value: object, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"Unable to parse float for {field_name}: {value}") from exc
    if not np.isfinite(out):
        raise ValueError(f"Non-finite float for {field_name}: {value}")
    return out


def _load_15min_metrics_rows(metrics_csv: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

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
            "load_factor",
            "ramp_max_up_kw_per_step",
            "ramp_max_down_kw_per_step",
        }
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Metrics CSV missing required columns: {sorted(missing)}")

        for row in reader:
            kind = str(row.get("trace_kind", "")).strip()
            if kind not in TRACE_KIND_ORDER:
                continue
            resolution_s = _safe_float(row.get("resolution_s", ""), "resolution_s")
            if not math.isclose(resolution_s, 900.0, rel_tol=0.0, abs_tol=1e-9):
                continue
            out[kind] = {
                "peak_kw": _safe_float(row.get("peak_kw", ""), "peak_kw"),
                "avg_kw": _safe_float(row.get("avg_kw", ""), "avg_kw"),
                "par": _safe_float(row.get("par", ""), "par"),
                "load_factor": _safe_float(row.get("load_factor", ""), "load_factor"),
                "ramp_max_up_kw_per_step": _safe_float(
                    row.get("ramp_max_up_kw_per_step", ""),
                    "ramp_max_up_kw_per_step",
                ),
                "ramp_max_down_kw_per_step": _safe_float(
                    row.get("ramp_max_down_kw_per_step", ""),
                    "ramp_max_down_kw_per_step",
                ),
            }

    missing_kinds = [k for k in TRACE_KIND_ORDER if k not in out]
    if missing_kinds:
        raise ValueError(
            f"Missing required 15-minute rows in {metrics_csv}: {missing_kinds}"
        )
    return out


def _derive_method_metrics(
    *,
    rows_15min: Mapping[str, Mapping[str, float]],
    power_factor: float,
) -> Dict[str, Dict[str, float]]:
    if (not np.isfinite(float(power_factor))) or float(power_factor) <= 0.0:
        raise ValueError("power_factor must be > 0")

    out: Dict[str, Dict[str, float]] = {}
    for kind in TRACE_KIND_ORDER:
        row = rows_15min[kind]
        peak_mw = float(row["peak_kw"]) / 1000.0
        avg_mw = float(row["avg_kw"]) / 1000.0
        par = float(row["par"])
        load_factor = float(row["load_factor"])
        max_abs_kw_per_15min = max(
            abs(float(row["ramp_max_up_kw_per_step"])),
            abs(float(row["ramp_max_down_kw_per_step"])),
        )
        max_ramp_mw_per_hr = (max_abs_kw_per_15min / 1000.0) * (3600.0 / 900.0)
        transformer_mva = peak_mw / float(power_factor)
        out[kind] = {
            "peak_mw": float(peak_mw),
            "avg_mw": float(avg_mw),
            "par": float(par),
            "max_ramp_mw_per_hr": float(max_ramp_mw_per_hr),
            "load_factor": float(load_factor),
            "transformer_mva": float(transformer_mva),
        }
    return out


def _fmt(value: Optional[float], decimals: int, dash: str = "---") -> str:
    if value is None:
        return dash
    if not np.isfinite(float(value)):
        return dash
    return f"{float(value):.{int(decimals)}f}"


def _build_table_rows(
    *,
    method_metrics: Mapping[str, Mapping[str, float]],
    decimals_power: int,
    decimals_ratio: int,
    decimals_ramp: int,
    hide_redundant_cells: bool,
) -> List[Dict[str, str]]:
    tdp = method_metrics["tdp_baseline"]
    mean = method_metrics["mean_baseline"]
    ours = method_metrics["ours"]

    peak_mean_cell = None if hide_redundant_cells else float(mean["peak_mw"])
    avg_tdp_cell = None if hide_redundant_cells else float(tdp["avg_mw"])

    return [
        {
            "metric": "Peak facility power (MW)",
            "tdp": _fmt(float(tdp["peak_mw"]), decimals_power),
            "mean": _fmt(peak_mean_cell, decimals_power),
            "ours": _fmt(float(ours["peak_mw"]), decimals_power),
        },
        {
            "metric": "Average facility power (MW)",
            "tdp": _fmt(avg_tdp_cell, decimals_power),
            "mean": _fmt(float(mean["avg_mw"]), decimals_power),
            "ours": _fmt(float(ours["avg_mw"]), decimals_power),
        },
        {
            "metric": "Peak-to-average ratio",
            "tdp": _fmt(float(tdp["par"]), decimals_ratio),
            "mean": _fmt(float(mean["par"]), decimals_ratio),
            "ours": _fmt(float(ours["par"]), decimals_ratio),
        },
        {
            "metric": "Max 15-min ramp rate (MW/hr)",
            "tdp": _fmt(float(tdp["max_ramp_mw_per_hr"]), decimals_ramp),
            "mean": _fmt(float(mean["max_ramp_mw_per_hr"]), decimals_ramp),
            "ours": _fmt(float(ours["max_ramp_mw_per_hr"]), decimals_ramp),
        },
        {
            "metric": "Load factor",
            "tdp": _fmt(float(tdp["load_factor"]), decimals_ratio),
            "mean": _fmt(float(mean["load_factor"]), decimals_ratio),
            "ours": _fmt(float(ours["load_factor"]), decimals_ratio),
        },
    ]


def _build_latex_table(
    *,
    rows: List[Mapping[str, str]],
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\vspace{0.4em}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{TDP} & \textbf{Mean} & \textbf{Ours} \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            f"{row['metric']} & {row['tdp']} & {row['mean']} & {row['ours']} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: str, rows: List[Mapping[str, str]]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "tdp", "mean", "ours"])
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "aggregated_dir": str(repo_root / "results" / "azure_facility" / "aggregated"),
        "node_trace_dir": str(repo_root / "results" / "azure_facility" / "node_traces"),
        "experimental_manifest": str(
            repo_root / "results" / "experimental_continuous_v1" / "manifest.json"
        ),
        "metrics_csv": str(repo_root / "results" / "eval_paper" / "azure_facility_metrics.csv"),
        "ldc_csv": str(repo_root / "results" / "eval_paper" / "azure_facility_ldc_15min.csv"),
        "out_csv": str(repo_root / "results" / "eval_paper" / "azure_facility_sizing_table.csv"),
        "out_json": str(repo_root / "results" / "eval_paper" / "azure_facility_sizing_table.json"),
        "out_tex": str(repo_root / "results" / "eval_paper" / "azure_facility_sizing_table.tex"),
    }


def generate_azure_facility_sizing_table(
    *,
    metrics_csv: str,
    out_csv: str,
    out_json: str,
    out_tex: str,
    power_factor: float = 0.9,
    decimals_power: int = 2,
    decimals_ratio: int = 2,
    decimals_ramp: int = 2,
    hide_redundant_cells: bool = True,
    caption: str = (
        "Infrastructure sizing from 24-hour facility simulation "
        "(240 nodes, PUE\\,$=$\\,1.3). "
        "Implied transformer sizing assumes 0.9 power factor."
    ),
    label: str = "tab:facility-sizing",
    recompute_metrics: bool = False,
    aggregated_dir: str = "",
    node_trace_dir: str = "",
    experimental_manifest: str = "",
    ldc_csv: str = "",
    config_id: str = "deepseek-r1-distill-70b_H100_tp4",
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
    tp_gpus: int = 4,
    gpu_tdp_w: float = 700.0,
    non_gpu_overhead_w: float = 1000.0,
    pue: float = 1.3,
) -> Dict[str, object]:
    if recompute_metrics:
        compute_azure_facility_metrics(
            aggregated_dir=aggregated_dir,
            node_trace_dir=node_trace_dir,
            experimental_manifest=experimental_manifest,
            metrics_csv=metrics_csv,
            ldc_csv=ldc_csv,
            config_id=config_id,
            rows=int(rows),
            racks_per_row=int(racks_per_row),
            nodes_per_rack=int(nodes_per_rack),
            tp_gpus=int(tp_gpus),
            gpu_tdp_w=float(gpu_tdp_w),
            non_gpu_overhead_w=float(non_gpu_overhead_w),
            pue=float(pue),
        )

    rows_15min = _load_15min_metrics_rows(metrics_csv)
    method_metrics = _derive_method_metrics(
        rows_15min=rows_15min,
        power_factor=float(power_factor),
    )
    table_rows = _build_table_rows(
        method_metrics=method_metrics,
        decimals_power=int(decimals_power),
        decimals_ratio=int(decimals_ratio),
        decimals_ramp=int(decimals_ramp),
        hide_redundant_cells=bool(hide_redundant_cells),
    )
    latex = _build_latex_table(rows=table_rows, caption=caption, label=label)

    _write_csv(out_csv, table_rows)
    _write_json(
        out_json,
        {
            "inputs": {
                "metrics_csv": str(Path(metrics_csv).resolve()),
                "recompute_metrics": bool(recompute_metrics),
                "power_factor": float(power_factor),
            },
            "formatting": {
                "decimals_power": int(decimals_power),
                "decimals_ratio": int(decimals_ratio),
                "decimals_ramp": int(decimals_ramp),
                "hide_redundant_cells": bool(hide_redundant_cells),
                "caption": str(caption),
                "label": str(label),
            },
            "method_metrics_15min": {
                TRACE_KIND_HEADER[k]: {kk: float(vv) for kk, vv in v.items()}
                for k, v in method_metrics.items()
            },
            "table_rows": [dict(r) for r in table_rows],
        },
    )
    _ensure_dir_for_file(out_tex)
    with open(out_tex, "w") as f:
        f.write(latex)

    return {
        "metrics_csv": metrics_csv,
        "out_csv": out_csv,
        "out_json": out_json,
        "out_tex": out_tex,
        "method_metrics_15min": method_metrics,
        "table_rows": table_rows,
        "latex": latex,
    }


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Generate the infrastructure sizing table (TDP/Mean/Ours) from Azure "
            "facility metrics at 15-minute resolution."
        )
    )
    parser.add_argument("--metrics-csv", default=defaults["metrics_csv"])
    parser.add_argument("--out-csv", default=defaults["out_csv"])
    parser.add_argument("--out-json", default=defaults["out_json"])
    parser.add_argument("--out-tex", default=defaults["out_tex"])
    parser.add_argument("--power-factor", type=float, default=0.9)
    parser.add_argument("--decimals-power", type=int, default=2)
    parser.add_argument("--decimals-ratio", type=int, default=2)
    parser.add_argument("--decimals-ramp", type=int, default=2)
    parser.add_argument(
        "--show-redundant-baseline-cells",
        action="store_true",
        help="Show Mean-peak and TDP-average cells instead of '---'.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Infrastructure sizing from 24-hour facility simulation "
            "(240 nodes, PUE\\,$=$\\,1.3). "
            "Implied transformer sizing assumes 0.9 power factor."
        ),
    )
    parser.add_argument("--label", default="tab:facility-sizing")

    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Recompute azure_facility_metrics.csv before generating the table.",
    )
    parser.add_argument("--aggregated-dir", default=defaults["aggregated_dir"])
    parser.add_argument("--node-trace-dir", default=defaults["node_trace_dir"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--ldc-csv", default=defaults["ldc_csv"])
    parser.add_argument("--config-id", default="deepseek-r1-distill-70b_H100_tp4")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--nodes-per-rack", type=int, default=4)
    parser.add_argument("--tp-gpus", type=int, default=4)
    parser.add_argument("--gpu-tdp-w", type=float, default=700.0)
    parser.add_argument("--non-gpu-overhead-w", type=float, default=1000.0)
    parser.add_argument("--pue", type=float, default=1.3)

    args = parser.parse_args()

    result = generate_azure_facility_sizing_table(
        metrics_csv=str(args.metrics_csv),
        out_csv=str(args.out_csv),
        out_json=str(args.out_json),
        out_tex=str(args.out_tex),
        power_factor=float(args.power_factor),
        decimals_power=int(args.decimals_power),
        decimals_ratio=int(args.decimals_ratio),
        decimals_ramp=int(args.decimals_ramp),
        hide_redundant_cells=not bool(args.show_redundant_baseline_cells),
        caption=str(args.caption),
        label=str(args.label),
        recompute_metrics=bool(args.recompute_metrics),
        aggregated_dir=str(args.aggregated_dir),
        node_trace_dir=str(args.node_trace_dir),
        experimental_manifest=str(args.experimental_manifest),
        ldc_csv=str(args.ldc_csv),
        config_id=str(args.config_id),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        nodes_per_rack=int(args.nodes_per_rack),
        tp_gpus=int(args.tp_gpus),
        gpu_tdp_w=float(args.gpu_tdp_w),
        non_gpu_overhead_w=float(args.non_gpu_overhead_w),
        pue=float(args.pue),
    )

    print("[generate_azure_facility_sizing_table] Done")
    print(f"  metrics_csv : {result['metrics_csv']}")
    print(f"  out_csv     : {result['out_csv']}")
    print(f"  out_json    : {result['out_json']}")
    print(f"  out_tex     : {result['out_tex']}")
    print("  15-min metrics (numeric):")
    for kind in TRACE_KIND_ORDER:
        name = TRACE_KIND_HEADER[kind]
        m = result["method_metrics_15min"][kind]
        print(
            f"    {name:>4s}: peak={m['peak_mw']:.4f} MW, avg={m['avg_mw']:.4f} MW, "
            f"PAR={m['par']:.4f}, ramp={m['max_ramp_mw_per_hr']:.4f} MW/hr, "
            f"LF={m['load_factor']:.4f}, xfmr={m['transformer_mva']:.4f} MVA"
        )


if __name__ == "__main__":
    main()
