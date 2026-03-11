#!/usr/bin/env python3
"""
Generate paper-ready infrastructure sizing table from isolated Azure metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.azure_scripts_baselines_included.azure_metrics import (
    compute_azure_facility_metrics,
)
from scripts.eval.azure_scripts_baselines_included.defaults import (
    DEFAULT_CONFIG_ID,
    DEFAULT_TRACE_KINDS,
    build_default_paths,
)
from scripts.eval.azure_scripts_baselines_included.io_utils import (
    ensure_dir_for_file,
    parse_csv_list,
    safe_float,
    write_json,
)

TRACE_KIND_HEADER = {
    "tdp_baseline": "TDP",
    "mean_baseline": "Mean",
    "splitwise_lut": "Splitwise (Tuned)",
    "splitwise_strict": "Splitwise (Strict)",
    "ours": "Ours",
}


def _normalize_method_order(method_order: Sequence[str] | str) -> List[str]:
    values = parse_csv_list(method_order) if isinstance(method_order, str) else [str(x).strip() for x in method_order]
    allowed = set(DEFAULT_TRACE_KINDS)
    out: List[str] = []
    for v in values:
        if not v:
            continue
        if v not in allowed:
            raise ValueError(f"Unsupported method '{v}'. Allowed: {sorted(allowed)}")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("No methods selected")
    return out


def _load_15min_metrics_rows(metrics_csv: str, method_order: Sequence[str]) -> Dict[str, Dict[str, float]]:
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
            if kind not in method_order:
                continue
            resolution_s = safe_float(row.get("resolution_s", ""), "resolution_s")
            if not math.isclose(resolution_s, 900.0, rel_tol=0.0, abs_tol=1e-9):
                continue
            out[kind] = {
                "peak_kw": safe_float(row.get("peak_kw", ""), "peak_kw"),
                "avg_kw": safe_float(row.get("avg_kw", ""), "avg_kw"),
                "par": safe_float(row.get("par", ""), "par"),
                "load_factor": safe_float(row.get("load_factor", ""), "load_factor"),
                "ramp_max_up_kw_per_step": safe_float(row.get("ramp_max_up_kw_per_step", ""), "ramp_max_up_kw_per_step"),
                "ramp_max_down_kw_per_step": safe_float(
                    row.get("ramp_max_down_kw_per_step", ""),
                    "ramp_max_down_kw_per_step",
                ),
            }

    missing_kinds = [k for k in method_order if k not in out]
    if missing_kinds:
        raise ValueError(f"Missing required 15-minute rows in {metrics_csv}: {missing_kinds}")
    return out


def _derive_method_metrics(
    *,
    rows_15min: Mapping[str, Mapping[str, float]],
    method_order: Sequence[str],
    power_factor: float,
) -> Dict[str, Dict[str, float]]:
    if (not np.isfinite(float(power_factor))) or float(power_factor) <= 0.0:
        raise ValueError("power_factor must be > 0")

    out: Dict[str, Dict[str, float]] = {}
    for kind in method_order:
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
    method_order: Sequence[str],
    decimals_power: int,
    decimals_ratio: int,
    decimals_ramp: int,
    hide_redundant_cells: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    def _row(metric: str, key: str, decimals: int) -> Dict[str, str]:
        r = {"metric": metric}
        for m in method_order:
            r[m] = _fmt(float(method_metrics[m][key]), decimals)
        return r

    rows.append(_row("Peak facility power (MW)", "peak_mw", decimals_power))
    rows.append(_row("Average facility power (MW)", "avg_mw", decimals_power))
    rows.append(_row("Peak-to-average ratio", "par", decimals_ratio))
    rows.append(_row("Max 15-min ramp rate (MW/hr)", "max_ramp_mw_per_hr", decimals_ramp))
    rows.append(_row("Load factor", "load_factor", decimals_ratio))

    if hide_redundant_cells and "tdp_baseline" in method_order and "mean_baseline" in method_order:
        # Preserve previous paper convention for the two deterministic redundant cells.
        rows[0]["mean_baseline"] = "---"
        rows[1]["tdp_baseline"] = "---"

    return rows


def _build_latex_table(
    *,
    rows: List[Mapping[str, str]],
    method_order: Sequence[str],
    caption: str,
    label: str,
) -> str:
    headers = [TRACE_KIND_HEADER.get(m, m) for m in method_order]
    align = "l" + ("c" * len(method_order))
    head_line = " & ".join([r"\textbf{Metric}"] + [rf"\textbf{{{h}}}" for h in headers]) + r" \\" 

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\vspace{0.4em}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        head_line,
        r"\midrule",
    ]

    for row in rows:
        values = [row["metric"]] + [row[m] for m in method_order]
        lines.append(" & ".join(values) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _write_csv(path: str, rows: List[Mapping[str, str]], method_order: Sequence[str]) -> None:
    ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric"] + list(method_order))
        writer.writeheader()
        writer.writerows(rows)


def generate_azure_facility_sizing_table(
    *,
    metrics_csv: str,
    out_csv: str,
    out_json: str,
    out_tex: str,
    method_order: Sequence[str] | str = ",".join(DEFAULT_TRACE_KINDS),
    power_factor: float = 0.9,
    decimals_power: int = 2,
    decimals_ratio: int = 2,
    decimals_ramp: int = 2,
    hide_redundant_cells: bool = False,
    caption: str = (
        "Infrastructure sizing from 24-hour facility simulation "
        "(240 nodes, PUE\\,$=$\\,1.3). "
        "Implied transformer sizing assumes 0.9 power factor."
    ),
    label: str = "tab:facility-sizing-bi",
    recompute_metrics: bool = False,
    aggregated_root: str = "",
    node_traces_root: str = "",
    experimental_manifest: str = "",
    ldc_csv: str = "",
    site_traces_15min_csv: str = "",
    config_id: str = DEFAULT_CONFIG_ID,
    trace_kinds: Sequence[str] | str = ",".join(DEFAULT_TRACE_KINDS),
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
    tp_gpus: int = 4,
    gpu_tdp_w: float = 700.0,
    non_gpu_overhead_w: float = 1000.0,
    pue: float = 1.3,
) -> Dict[str, object]:
    method_keys = _normalize_method_order(method_order)

    if recompute_metrics:
        compute_azure_facility_metrics(
            aggregated_root=aggregated_root,
            node_traces_root=node_traces_root,
            experimental_manifest=experimental_manifest,
            metrics_csv=metrics_csv,
            ldc_csv=ldc_csv,
            site_traces_15min_csv=site_traces_15min_csv,
            config_id=config_id,
            trace_kinds=trace_kinds,
            rows=int(rows),
            racks_per_row=int(racks_per_row),
            nodes_per_rack=int(nodes_per_rack),
            tp_gpus=int(tp_gpus),
            gpu_tdp_w=float(gpu_tdp_w),
            non_gpu_overhead_w=float(non_gpu_overhead_w),
            pue=float(pue),
        )

    rows_15min = _load_15min_metrics_rows(metrics_csv, method_keys)
    method_metrics = _derive_method_metrics(
        rows_15min=rows_15min,
        method_order=method_keys,
        power_factor=float(power_factor),
    )
    table_rows = _build_table_rows(
        method_metrics=method_metrics,
        method_order=method_keys,
        decimals_power=int(decimals_power),
        decimals_ratio=int(decimals_ratio),
        decimals_ramp=int(decimals_ramp),
        hide_redundant_cells=bool(hide_redundant_cells),
    )
    latex = _build_latex_table(rows=table_rows, method_order=method_keys, caption=caption, label=label)

    _write_csv(out_csv, table_rows, method_keys)
    write_json(
        out_json,
        {
            "inputs": {
                "metrics_csv": str(Path(metrics_csv).resolve()),
                "recompute_metrics": bool(recompute_metrics),
                "power_factor": float(power_factor),
                "method_order": list(method_keys),
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
                TRACE_KIND_HEADER.get(k, k): {kk: float(vv) for kk, vv in v.items()} for k, v in method_metrics.items()
            },
            "table_rows": [dict(r) for r in table_rows],
        },
    )
    ensure_dir_for_file(out_tex)
    with open(out_tex, "w") as f:
        f.write(latex)

    return {
        "metrics_csv": metrics_csv,
        "out_csv": out_csv,
        "out_json": out_json,
        "out_tex": out_tex,
        "method_order": method_keys,
        "method_metrics_15min": method_metrics,
        "table_rows": table_rows,
        "latex": latex,
    }


def main() -> None:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Generate isolated Azure facility sizing table.")
    parser.add_argument("--metrics-csv", default=defaults["metrics_csv"])
    parser.add_argument("--out-csv", default=defaults["sizing_out_csv"])
    parser.add_argument("--out-json", default=defaults["sizing_out_json"])
    parser.add_argument("--out-tex", default=defaults["sizing_out_tex"])
    parser.add_argument("--method-order", default=",".join(DEFAULT_TRACE_KINDS))
    parser.add_argument("--power-factor", type=float, default=0.9)
    parser.add_argument("--decimals-power", type=int, default=2)
    parser.add_argument("--decimals-ratio", type=int, default=2)
    parser.add_argument("--decimals-ramp", type=int, default=2)
    parser.add_argument("--hide-redundant-cells", action="store_true")
    parser.add_argument(
        "--caption",
        default=(
            "Infrastructure sizing from 24-hour facility simulation "
            "(240 nodes, PUE\\,$=$\\,1.3). "
            "Implied transformer sizing assumes 0.9 power factor."
        ),
    )
    parser.add_argument("--label", default="tab:facility-sizing-bi")

    parser.add_argument("--recompute-metrics", action="store_true")
    parser.add_argument("--aggregated-root", default=defaults["aggregated_root"])
    parser.add_argument("--node-traces-root", default=defaults["node_traces_root"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--ldc-csv", default=defaults["ldc_csv"])
    parser.add_argument("--site-traces-15min-csv", default=defaults["site_traces_15min_csv"])
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--trace-kinds", default=",".join(DEFAULT_TRACE_KINDS))
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
        method_order=str(args.method_order),
        power_factor=float(args.power_factor),
        decimals_power=int(args.decimals_power),
        decimals_ratio=int(args.decimals_ratio),
        decimals_ramp=int(args.decimals_ramp),
        hide_redundant_cells=bool(args.hide_redundant_cells),
        caption=str(args.caption),
        label=str(args.label),
        recompute_metrics=bool(args.recompute_metrics),
        aggregated_root=str(args.aggregated_root),
        node_traces_root=str(args.node_traces_root),
        experimental_manifest=str(args.experimental_manifest),
        ldc_csv=str(args.ldc_csv),
        site_traces_15min_csv=str(args.site_traces_15min_csv),
        config_id=str(args.config_id),
        trace_kinds=str(args.trace_kinds),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        nodes_per_rack=int(args.nodes_per_rack),
        tp_gpus=int(args.tp_gpus),
        gpu_tdp_w=float(args.gpu_tdp_w),
        non_gpu_overhead_w=float(args.non_gpu_overhead_w),
        pue=float(args.pue),
    )

    print("[generate_azure_facility_sizing_table_bi] Done")
    print(f"  metrics_csv : {result['metrics_csv']}")
    print(f"  out_csv     : {result['out_csv']}")
    print(f"  out_json    : {result['out_json']}")
    print(f"  out_tex     : {result['out_tex']}")
    print(f"  methods     : {', '.join(result['method_order'])}")


if __name__ == "__main__":
    main()
