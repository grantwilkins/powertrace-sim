#!/usr/bin/env python3
"""
Experiment 2c: Hierarchical aggregation for Azure facility traces.

Hierarchy per method:
  node (250ms GPU trace) -> +non-GPU overhead -> rack (1s) -> row (1s) -> site.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.azure_defaults import (
    DEFAULT_METHODS_GENERATION,
    DEFAULT_NON_GPU_OVERHEAD_W,
    DEFAULT_PUE,
    build_default_paths,
    ensure_dir,
    parse_csv_list,
    write_json,
)
from scripts.eval.facility import FacilityLayout, downsample_mean


def aggregate_single_method(
    *,
    node_trace_dir: str,
    out_dir: str,
    dt: float = 0.25,
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
    non_gpu_overhead_w: float = DEFAULT_NON_GPU_OVERHEAD_W,
    pue: float = DEFAULT_PUE,
) -> Dict[str, object]:
    if float(dt) <= 0.0:
        raise ValueError("dt must be > 0")
    if float(non_gpu_overhead_w) < 0.0:
        raise ValueError("non_gpu_overhead_w must be >= 0")
    if float(pue) <= 0.0:
        raise ValueError("pue must be > 0")

    layout = FacilityLayout(
        rows=int(rows),
        racks_per_row=int(racks_per_row),
        nodes_per_rack=int(nodes_per_rack),
    )
    if layout.n_nodes <= 0:
        raise ValueError("layout yields zero nodes")

    bins_per_sec = int(round(1.0 / float(dt)))
    if bins_per_sec <= 0:
        raise ValueError("invalid dt: could not derive bins_per_sec")

    expected_paths: List[Tuple[int, int, int, str]] = []
    for row_i, rack_j, node_k in layout.iter_nodes():
        expected_paths.append(
            (row_i, rack_j, node_k, os.path.join(node_trace_dir, f"node_{row_i}_{rack_j}_{node_k}.npy"))
        )
    missing = [path for _, _, _, path in expected_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} node trace files in {node_trace_dir}. First missing: {missing[0]}"
        )

    first_trace = np.asarray(np.load(expected_paths[0][3]), dtype=np.float64).reshape(-1)
    if first_trace.size == 0:
        raise ValueError(f"Empty node trace: {expected_paths[0][3]}")
    t_horizon = int(first_trace.size)
    for _, _, _, path in expected_paths[1:]:
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        if int(arr.size) != t_horizon:
            raise ValueError(f"Trace length mismatch: {path} has {arr.size}, expected {t_horizon}")

    if t_horizon % bins_per_sec != 0:
        raise ValueError(
            f"T={t_horizon} not divisible by bins_per_sec={bins_per_sec} for 1s downsample"
        )
    n_sec = int(t_horizon // bins_per_sec)
    if n_sec % 60 != 0:
        raise ValueError(f"1s length {n_sec} not divisible by 60 for 1min outputs")
    if n_sec % 900 != 0:
        raise ValueError(f"1s length {n_sec} not divisible by 900 for 15min outputs")

    ensure_dir(out_dir)
    site_it_250ms = np.zeros((t_horizon,), dtype=np.float64)
    site_it_1s = np.zeros((n_sec,), dtype=np.float64)

    for row_i in range(int(layout.rows)):
        row_1s = np.zeros((n_sec,), dtype=np.float64)
        for rack_j in range(int(layout.racks_per_row)):
            rack_250ms = np.zeros((t_horizon,), dtype=np.float64)
            for node_k in range(int(layout.nodes_per_rack)):
                node_path = os.path.join(node_trace_dir, f"node_{row_i}_{rack_j}_{node_k}.npy")
                node_gpu = np.asarray(np.load(node_path), dtype=np.float64).reshape(-1)
                rack_250ms += node_gpu + float(non_gpu_overhead_w)
            site_it_250ms += rack_250ms
            rack_1s = downsample_mean(rack_250ms, bins_per_sec)
            row_1s += rack_1s
            np.save(os.path.join(out_dir, f"rack_{row_i}_{rack_j}.npy"), np.asarray(rack_1s, dtype=np.float32))
        site_it_1s += row_1s
        np.save(os.path.join(out_dir, f"row_{row_i}.npy"), np.asarray(row_1s, dtype=np.float32))

    site_it_1s_from_250ms = downsample_mean(site_it_250ms, bins_per_sec)
    if not np.allclose(site_it_1s, site_it_1s_from_250ms, atol=1e-5, rtol=1e-6):
        raise RuntimeError("Internal consistency check failed: site_it_1s mismatch")

    site_it_1min = downsample_mean(site_it_1s, 60)
    site_it_15min = downsample_mean(site_it_1s, 900)

    site_250ms = site_it_250ms * float(pue)
    site_1s = site_it_1s * float(pue)
    site_1min = site_it_1min * float(pue)
    site_15min = site_it_15min * float(pue)

    np.save(os.path.join(out_dir, "site_it_250ms.npy"), np.asarray(site_it_250ms, dtype=np.float32))
    np.save(os.path.join(out_dir, "site_it_1s.npy"), np.asarray(site_it_1s, dtype=np.float32))
    np.save(os.path.join(out_dir, "site_it_1min.npy"), np.asarray(site_it_1min, dtype=np.float32))
    np.save(os.path.join(out_dir, "site_it_15min.npy"), np.asarray(site_it_15min, dtype=np.float32))
    np.save(os.path.join(out_dir, "site_250ms.npy"), np.asarray(site_250ms, dtype=np.float32))
    np.save(os.path.join(out_dir, "site_1s.npy"), np.asarray(site_1s, dtype=np.float32))
    np.save(os.path.join(out_dir, "site_1min.npy"), np.asarray(site_1min, dtype=np.float32))
    np.save(os.path.join(out_dir, "site_15min.npy"), np.asarray(site_15min, dtype=np.float32))

    meta = {
        "status": "ok",
        "node_trace_dir": node_trace_dir,
        "out_dir": out_dir,
        "layout": {
            "rows": int(layout.rows),
            "racks_per_row": int(layout.racks_per_row),
            "nodes_per_rack": int(layout.nodes_per_rack),
            "n_nodes": int(layout.n_nodes),
        },
        "timing": {
            "dt": float(dt),
            "timesteps_250ms": int(t_horizon),
            "timesteps_1s": int(n_sec),
            "timesteps_1min": int(site_1min.size),
            "timesteps_15min": int(site_15min.size),
        },
        "power": {
            "non_gpu_overhead_w": float(non_gpu_overhead_w),
            "pue": float(pue),
            "site_it_peak_w": float(np.max(site_it_250ms)),
            "site_peak_w": float(np.max(site_250ms)),
            "site_avg_w": float(np.mean(site_250ms)),
        },
    }
    write_json(os.path.join(out_dir, "aggregation_metadata.json"), meta)
    return meta


def aggregate_facility_traces(**kwargs) -> Dict[str, object]:
    return aggregate_single_method(**kwargs)


def aggregate_all_methods(
    *,
    node_traces_root: str,
    aggregated_root: str,
    methods: Sequence[str] | str = ",".join(DEFAULT_METHODS_GENERATION),
    dt: float = 0.25,
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
    non_gpu_overhead_w: float = DEFAULT_NON_GPU_OVERHEAD_W,
    pue: float = DEFAULT_PUE,
) -> Dict[str, object]:
    method_list = parse_csv_list(methods) if isinstance(methods, str) else [str(x) for x in methods]
    if len(method_list) == 0:
        raise ValueError("No methods selected for aggregation")

    ensure_dir(aggregated_root)
    by_method: Dict[str, object] = {}
    for method in method_list:
        by_method[method] = aggregate_single_method(
            node_trace_dir=os.path.join(node_traces_root, method),
            out_dir=os.path.join(aggregated_root, method),
            dt=float(dt),
            rows=int(rows),
            racks_per_row=int(racks_per_row),
            nodes_per_rack=int(nodes_per_rack),
            non_gpu_overhead_w=float(non_gpu_overhead_w),
            pue=float(pue),
        )

    summary = {
        "status": "ok",
        "node_traces_root": node_traces_root,
        "aggregated_root": aggregated_root,
        "methods": method_list,
        "by_method": by_method,
    }
    write_json(os.path.join(aggregated_root, "aggregation_summary.json"), summary)
    return summary


def main() -> None:
    defaults = build_default_paths()
    parser = argparse.ArgumentParser(description="Aggregate Azure node traces by method.")
    parser.add_argument("--node-traces-root", default=defaults["node_traces_root"])
    parser.add_argument("--output-root", default=defaults["aggregated_root"])
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS_GENERATION))
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--nodes-per-rack", type=int, default=4)
    parser.add_argument("--non-gpu-overhead-w", type=float, default=DEFAULT_NON_GPU_OVERHEAD_W)
    parser.add_argument("--pue", type=float, default=DEFAULT_PUE)
    args = parser.parse_args()

    meta = aggregate_all_methods(
        node_traces_root=str(args.node_traces_root),
        aggregated_root=str(args.output_root),
        methods=str(args.methods),
        dt=float(args.dt),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        nodes_per_rack=int(args.nodes_per_rack),
        non_gpu_overhead_w=float(args.non_gpu_overhead_w),
        pue=float(args.pue),
    )

    print("=" * 72)
    print("Azure Hierarchical Aggregation")
    print("=" * 72)
    print(f"Methods            : {', '.join(meta['methods'])}")
    print(f"Node traces root   : {meta['node_traces_root']}")
    print(f"Output root        : {meta['aggregated_root']}")
    print(f"Summary            : {os.path.join(meta['aggregated_root'], 'aggregation_summary.json')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
