#!/usr/bin/env python3
"""
Experiment 2d: Compute facility-level metrics and baselines for Azure traces.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.azure_defaults import (
    DEFAULT_CONFIG_ID,
    DEFAULT_NON_GPU_OVERHEAD_W,
    DEFAULT_PUE,
    DEFAULT_TRACE_KINDS,
    build_default_paths,
    ensure_dir_for_file,
    load_json,
    parse_csv_list,
)
from scripts.eval.facility import FacilityLayout
from scripts.eval.pipeline_utils import resolve_experimental_paths

RESOLUTION_FILE_MAP = {
    0.25: "site_250ms.npy",
    1.0: "site_1s.npy",
    60.0: "site_1min.npy",
    900.0: "site_15min.npy",
}
RESOLUTION_FILE_MAP_IT = {
    0.25: "site_it_250ms.npy",
    1.0: "site_it_1s.npy",
    60.0: "site_it_1min.npy",
    900.0: "site_it_15min.npy",
}
NON_CONSTANT_METHODS = {"ours", "splitwise_strict"}


def _load_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required array not found: {path}")
    arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty array: {path}")
    return arr


def _load_train_mean_gpu_power_w(
    *,
    config_id: str,
    experimental_manifest_path: str,
) -> float:
    manifest = load_json(experimental_manifest_path)
    base = str(Path(experimental_manifest_path).resolve().parent)
    dataset_path, split_path = resolve_experimental_paths(
        manifest,
        config_id=config_id,
        experimental_base=base,
    )
    split_payload = load_json(split_path)
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]

    with np.load(dataset_path, allow_pickle=True) as data:
        power_arr = np.asarray(data["power"], dtype=object)
    n_total = int(len(power_arr))

    traces: List[np.ndarray] = []
    for idx in train_indices:
        if idx < 0 or idx >= n_total:
            continue
        power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if power.size > 0:
            traces.append(power.astype(np.float64))
    if len(traces) == 0:
        for idx in range(n_total):
            power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
            if power.size > 0:
                traces.append(power.astype(np.float64))
    if len(traces) == 0:
        raise ValueError(f"No training traces available for {config_id}")

    flat_total = np.concatenate(traces, axis=0).astype(np.float64)
    if flat_total.size == 0:
        raise ValueError("Empty train GPU power pool")
    return float(np.mean(flat_total))


def _compute_metrics(arr_kw: np.ndarray, resolution_s: float) -> Dict[str, float]:
    x = np.asarray(arr_kw, dtype=np.float64).reshape(-1)
    if x.size <= 0:
        raise ValueError("Cannot compute metrics for empty array")

    peak = float(np.max(x))
    avg = float(np.mean(x))
    par = float(peak / avg) if avg > 0 else float("nan")
    load_factor = float(avg / peak) if peak > 0 else float("nan")

    ramps = np.diff(x)
    if ramps.size > 0:
        ramp_p50 = float(np.percentile(ramps, 50))
        ramp_p95_abs = float(np.percentile(np.abs(ramps), 95))
        ramp_p99_abs = float(np.percentile(np.abs(ramps), 99))
        ramp_max_up = float(np.max(ramps))
        ramp_max_down = float(np.min(ramps))
    else:
        ramp_p50 = float("nan")
        ramp_p95_abs = float("nan")
        ramp_p99_abs = float("nan")
        ramp_max_up = float("nan")
        ramp_max_down = float("nan")

    ramp_p95_abs_per_s = (
        float(ramp_p95_abs / float(resolution_s))
        if np.isfinite(ramp_p95_abs) and float(resolution_s) > 0.0
        else float("nan")
    )
    ldc_p95 = float(np.percentile(x, 95))
    ldc_p99 = float(np.percentile(x, 99))
    return {
        "peak_kw": peak,
        "avg_kw": avg,
        "par": par,
        "load_factor": load_factor,
        "ramp_p50_kw_per_step": ramp_p50,
        "ramp_p95_abs_kw_per_step": ramp_p95_abs,
        "ramp_p99_abs_kw_per_step": ramp_p99_abs,
        "ramp_max_up_kw_per_step": ramp_max_up,
        "ramp_max_down_kw_per_step": ramp_max_down,
        "ramp_p95_abs_kw_per_s": ramp_p95_abs_per_s,
        "ldc_p95_kw": ldc_p95,
        "ldc_p99_kw": ldc_p99,
    }


def _compute_diversity_factor_it(
    *,
    node_trace_dir: str,
    site_it_250ms_w: np.ndarray,
    layout: FacilityLayout,
    non_gpu_overhead_w: float,
) -> float:
    site_peak = float(np.max(np.asarray(site_it_250ms_w, dtype=np.float64).reshape(-1)))
    node_peaks: List[float] = []
    for row, rack, node in layout.iter_nodes():
        path = os.path.join(node_trace_dir, f"node_{row}_{rack}_{node}.npy")
        node_gpu = _load_array(path)
        node_it = node_gpu + float(non_gpu_overhead_w)
        node_peaks.append(float(np.max(node_it)))
    single_node_peak = float(np.max(np.asarray(node_peaks, dtype=np.float64)))
    denom = float(layout.n_nodes) * float(single_node_peak)
    if denom <= 0.0 or (not np.isfinite(denom)):
        return float("nan")
    return float(site_peak / denom)


def _load_method_site_arrays(aggregated_root: str, method: str) -> Dict[float, np.ndarray]:
    method_dir = os.path.join(aggregated_root, method)
    return {res_s: _load_array(os.path.join(method_dir, fname)) for res_s, fname in RESOLUTION_FILE_MAP.items()}


def _load_method_site_it_arrays(aggregated_root: str, method: str) -> Dict[float, np.ndarray]:
    method_dir = os.path.join(aggregated_root, method)
    return {
        res_s: _load_array(os.path.join(method_dir, fname))
        for res_s, fname in RESOLUTION_FILE_MAP_IT.items()
    }


def _normalize_trace_kinds(trace_kinds: Sequence[str] | str) -> List[str]:
    kinds = parse_csv_list(trace_kinds) if isinstance(trace_kinds, str) else [str(x).strip() for x in trace_kinds]
    out: List[str] = []
    allowed = set(DEFAULT_TRACE_KINDS)
    for kind in kinds:
        if not kind:
            continue
        if kind not in allowed:
            raise ValueError(f"Unsupported trace_kind '{kind}'. Allowed: {sorted(allowed)}")
        if kind not in out:
            out.append(kind)
    if not out:
        raise ValueError("No trace kinds selected")
    return out


def compute_azure_facility_metrics(
    *,
    aggregated_root: str,
    node_traces_root: str,
    experimental_manifest: str,
    metrics_csv: str,
    ldc_csv: str,
    site_traces_15min_csv: str,
    config_id: str = DEFAULT_CONFIG_ID,
    trace_kinds: Sequence[str] | str = ",".join(DEFAULT_TRACE_KINDS),
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
    tp_gpus: int = 4,
    gpu_tdp_w: float = 700.0,
    non_gpu_overhead_w: float = DEFAULT_NON_GPU_OVERHEAD_W,
    pue: float = DEFAULT_PUE,
) -> Dict[str, object]:
    if int(tp_gpus) <= 0:
        raise ValueError("tp_gpus must be >= 1")
    if float(gpu_tdp_w) <= 0.0:
        raise ValueError("gpu_tdp_w must be > 0")
    if float(non_gpu_overhead_w) < 0.0:
        raise ValueError("non_gpu_overhead_w must be >= 0")
    if float(pue) <= 0.0:
        raise ValueError("pue must be > 0")

    selected_kinds = _normalize_trace_kinds(trace_kinds)
    layout = FacilityLayout(rows=int(rows), racks_per_row=int(racks_per_row), nodes_per_rack=int(nodes_per_rack))
    n_nodes = int(layout.n_nodes)

    method_site_w: Dict[str, Dict[float, np.ndarray]] = {}
    method_site_it_w: Dict[str, Dict[float, np.ndarray]] = {}
    for method in NON_CONSTANT_METHODS:
        try:
            method_site_w[method] = _load_method_site_arrays(aggregated_root, method)
            method_site_it_w[method] = _load_method_site_it_arrays(aggregated_root, method)
        except Exception:
            continue
    if len(method_site_w) == 0:
        raise ValueError("No aggregated non-constant method traces were found")

    reference_method = "ours" if "ours" in method_site_w else sorted(method_site_w.keys())[0]
    reference_by_resolution = method_site_w[reference_method]

    node_tdp_it_w = float(tp_gpus) * float(gpu_tdp_w) + float(non_gpu_overhead_w)
    site_tdp_w = float(n_nodes) * node_tdp_it_w * float(pue)

    train_mean_gpu_w = _load_train_mean_gpu_power_w(
        config_id=config_id,
        experimental_manifest_path=experimental_manifest,
    )
    node_mean_it_w = float(train_mean_gpu_w) + float(non_gpu_overhead_w)
    site_mean_w = float(n_nodes) * node_mean_it_w * float(pue)

    diversity_by_method: Dict[str, float] = {}
    for method, site_it_by_res in method_site_it_w.items():
        diversity_by_method[method] = _compute_diversity_factor_it(
            node_trace_dir=os.path.join(node_traces_root, method),
            site_it_250ms_w=site_it_by_res[0.25],
            layout=layout,
            non_gpu_overhead_w=float(non_gpu_overhead_w),
        )

    rows_out: List[Dict[str, object]] = []
    ldc_rows: List[Dict[str, object]] = []
    site_trace_rows: List[Dict[str, object]] = []

    for resolution_s in (0.25, 1.0, 60.0, 900.0):
        n_samples = int(reference_by_resolution[resolution_s].size)
        values_by_kind: Dict[str, np.ndarray] = {}
        for trace_kind in selected_kinds:
            if trace_kind in NON_CONSTANT_METHODS:
                if trace_kind not in method_site_w:
                    raise ValueError(f"trace_kind '{trace_kind}' requested but method traces are missing")
                values_by_kind[trace_kind] = method_site_w[trace_kind][resolution_s] / 1000.0
            elif trace_kind == "tdp_baseline":
                values_by_kind[trace_kind] = np.full((n_samples,), fill_value=float(site_tdp_w / 1000.0), dtype=np.float64)
            elif trace_kind == "mean_baseline":
                values_by_kind[trace_kind] = np.full((n_samples,), fill_value=float(site_mean_w / 1000.0), dtype=np.float64)
            else:
                raise ValueError(f"Unsupported trace_kind: {trace_kind}")

        for trace_kind in selected_kinds:
            values_kw = np.asarray(values_by_kind[trace_kind], dtype=np.float64).reshape(-1)
            notes = ""
            diversity = 1.0
            if trace_kind in NON_CONSTANT_METHODS:
                diversity = float(diversity_by_method.get(trace_kind, float("nan")))
            else:
                notes = "constant_trace"

            metrics = _compute_metrics(values_kw, resolution_s=resolution_s)
            rows_out.append(
                {
                    "trace_kind": str(trace_kind),
                    "resolution_s": float(resolution_s),
                    "n_samples": int(values_kw.size),
                    "peak_kw": float(metrics["peak_kw"]),
                    "avg_kw": float(metrics["avg_kw"]),
                    "par": float(metrics["par"]),
                    "load_factor": float(metrics["load_factor"]),
                    "ramp_p50_kw_per_step": float(metrics["ramp_p50_kw_per_step"]),
                    "ramp_p95_abs_kw_per_step": float(metrics["ramp_p95_abs_kw_per_step"]),
                    "ramp_p99_abs_kw_per_step": float(metrics["ramp_p99_abs_kw_per_step"]),
                    "ramp_max_up_kw_per_step": float(metrics["ramp_max_up_kw_per_step"]),
                    "ramp_max_down_kw_per_step": float(metrics["ramp_max_down_kw_per_step"]),
                    "ramp_p95_abs_kw_per_s": float(metrics["ramp_p95_abs_kw_per_s"]),
                    "ldc_p95_kw": float(metrics["ldc_p95_kw"]),
                    "ldc_p99_kw": float(metrics["ldc_p99_kw"]),
                    "diversity_factor_it": float(diversity),
                    "status": "evaluated",
                    "notes": str(notes),
                }
            )

            if float(resolution_s) == 900.0:
                sorted_desc = np.sort(values_kw)[::-1]
                n = int(sorted_desc.size)
                for idx, value in enumerate(sorted_desc):
                    ldc_rows.append(
                        {
                            "trace_kind": str(trace_kind),
                            "resolution_s": float(resolution_s),
                            "rank": int(idx),
                            "fraction_exceeded": float(idx / max(1, n)),
                            "power_kw": float(value),
                        }
                    )

                hours = (np.arange(values_kw.size, dtype=np.float64) + 0.5) * (900.0 / 3600.0)
                for idx, value in enumerate(values_kw):
                    site_trace_rows.append(
                        {
                            "trace_kind": str(trace_kind),
                            "bin_idx": int(idx),
                            "hour": float(hours[idx]),
                            "power_kw": float(value),
                            "power_mw": float(value / 1000.0),
                        }
                    )

    ensure_dir_for_file(metrics_csv)
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_kind",
                "resolution_s",
                "n_samples",
                "peak_kw",
                "avg_kw",
                "par",
                "load_factor",
                "ramp_p50_kw_per_step",
                "ramp_p95_abs_kw_per_step",
                "ramp_p99_abs_kw_per_step",
                "ramp_max_up_kw_per_step",
                "ramp_max_down_kw_per_step",
                "ramp_p95_abs_kw_per_s",
                "ldc_p95_kw",
                "ldc_p99_kw",
                "diversity_factor_it",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    ensure_dir_for_file(ldc_csv)
    with open(ldc_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_kind", "resolution_s", "rank", "fraction_exceeded", "power_kw"],
        )
        writer.writeheader()
        writer.writerows(ldc_rows)

    ensure_dir_for_file(site_traces_15min_csv)
    with open(site_traces_15min_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_kind", "bin_idx", "hour", "power_kw", "power_mw"],
        )
        writer.writeheader()
        writer.writerows(site_trace_rows)

    return {
        "status": "ok",
        "config_id": config_id,
        "aggregated_root": aggregated_root,
        "node_traces_root": node_traces_root,
        "trace_kinds": selected_kinds,
        "metrics_csv": metrics_csv,
        "ldc_csv": ldc_csv,
        "site_traces_15min_csv": site_traces_15min_csv,
        "n_rows_metrics": int(len(rows_out)),
        "n_rows_ldc": int(len(ldc_rows)),
        "n_rows_site_traces": int(len(site_trace_rows)),
        "layout": {
            "rows": int(layout.rows),
            "racks_per_row": int(layout.racks_per_row),
            "nodes_per_rack": int(layout.nodes_per_rack),
            "n_nodes": int(layout.n_nodes),
        },
        "baselines": {
            "site_tdp_w": float(site_tdp_w),
            "site_mean_w": float(site_mean_w),
            "train_mean_gpu_w": float(train_mean_gpu_w),
        },
        "diversity_by_method": {key: float(value) for key, value in diversity_by_method.items()},
        "reference_method": str(reference_method),
    }


def main() -> None:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Compute Azure facility metrics with Splitwise baselines.")
    parser.add_argument("--aggregated-root", default=defaults["aggregated_root"])
    parser.add_argument("--node-traces-root", default=defaults["node_traces_root"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--metrics-csv", default=defaults["metrics_csv"])
    parser.add_argument("--ldc-csv", default=defaults["ldc_csv"])
    parser.add_argument("--site-traces-15min-csv", default=defaults["site_traces_15min_csv"])
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--trace-kinds", default=",".join(DEFAULT_TRACE_KINDS))
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--nodes-per-rack", type=int, default=4)
    parser.add_argument("--tp-gpus", type=int, default=4)
    parser.add_argument("--gpu-tdp-w", type=float, default=700.0)
    parser.add_argument("--non-gpu-overhead-w", type=float, default=DEFAULT_NON_GPU_OVERHEAD_W)
    parser.add_argument("--pue", type=float, default=DEFAULT_PUE)
    args = parser.parse_args()

    summary = compute_azure_facility_metrics(
        aggregated_root=str(args.aggregated_root),
        node_traces_root=str(args.node_traces_root),
        experimental_manifest=str(args.experimental_manifest),
        metrics_csv=str(args.metrics_csv),
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

    print("=" * 72)
    print("Azure Facility Metrics")
    print("=" * 72)
    print(f"Config              : {summary['config_id']}")
    print(f"Trace kinds         : {', '.join(summary['trace_kinds'])}")
    print(f"Metrics CSV         : {summary['metrics_csv']}")
    print(f"LDC CSV             : {summary['ldc_csv']}")
    print(f"Site traces 15min   : {summary['site_traces_15min_csv']}")
    print(f"Metric rows         : {summary['n_rows_metrics']}")
    print(f"LDC rows            : {summary['n_rows_ldc']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
