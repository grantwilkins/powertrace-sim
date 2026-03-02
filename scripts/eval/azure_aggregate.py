#!/usr/bin/env python3
"""
Experiment 2c: Hierarchical aggregation for Azure facility traces.

Hierarchy:
  node (250ms GPU trace) -> +non-GPU overhead -> rack (1s) -> row (1s) -> site.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FacilityLayout:
    rows: int = 10
    racks_per_row: int = 6
    nodes_per_rack: int = 4

    @property
    def n_nodes(self) -> int:
        return int(self.rows) * int(self.racks_per_row) * int(self.nodes_per_rack)

    def iter_nodes(self) -> Sequence[Tuple[int, int, int]]:
        for row in range(int(self.rows)):
            for rack in range(int(self.racks_per_row)):
                for node in range(int(self.nodes_per_rack)):
                    yield (int(row), int(rack), int(node))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _downsample_mean(values: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    f = int(factor)
    if f <= 0:
        raise ValueError("downsample factor must be >= 1")
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if arr.size % f != 0:
        raise ValueError(f"Array length {arr.size} not divisible by factor {f}")
    return np.mean(arr.reshape(-1, f), axis=1).astype(np.float64)


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "node_trace_dir": str(repo_root / "results" / "azure_facility" / "node_traces"),
        "out_dir": str(repo_root / "results" / "azure_facility" / "aggregated"),
    }


def aggregate_facility_traces(
    *,
    node_trace_dir: str,
    out_dir: str,
    dt: float = 0.25,
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
    non_gpu_overhead_w: float = 1000.0,
    pue: float = 1.3,
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
        path = os.path.join(node_trace_dir, f"node_{row_i}_{rack_j}_{node_k}.npy")
        expected_paths.append((row_i, rack_j, node_k, path))
    missing = [p for _, _, _, p in expected_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} node trace files. First missing: {missing[0]}"
        )

    # Infer T and validate all node traces have same length.
    first_path = expected_paths[0][3]
    first_trace = np.asarray(np.load(first_path), dtype=np.float64).reshape(-1)
    if first_trace.size == 0:
        raise ValueError(f"Empty node trace: {first_path}")
    t_horizon = int(first_trace.size)
    for _, _, _, p in expected_paths[1:]:
        arr = np.asarray(np.load(p), dtype=np.float64).reshape(-1)
        if int(arr.size) != t_horizon:
            raise ValueError(f"Trace length mismatch: {p} has {arr.size}, expected {t_horizon}")

    if t_horizon % bins_per_sec != 0:
        raise ValueError(
            f"T={t_horizon} not divisible by bins_per_sec={bins_per_sec} for 1s downsample"
        )
    n_sec = int(t_horizon // bins_per_sec)
    if n_sec % 60 != 0:
        raise ValueError(f"1s length {n_sec} not divisible by 60 for 1min outputs")
    if n_sec % 900 != 0:
        raise ValueError(f"1s length {n_sec} not divisible by 900 for 15min outputs")

    _ensure_dir(out_dir)

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
            rack_1s = _downsample_mean(rack_250ms, bins_per_sec)
            row_1s += rack_1s
            rack_out_path = os.path.join(out_dir, f"rack_{row_i}_{rack_j}.npy")
            np.save(rack_out_path, np.asarray(rack_1s, dtype=np.float32))
        site_it_1s += row_1s
        row_out_path = os.path.join(out_dir, f"row_{row_i}.npy")
        np.save(row_out_path, np.asarray(row_1s, dtype=np.float32))

    site_it_1s_from_250ms = _downsample_mean(site_it_250ms, bins_per_sec)
    if not np.allclose(site_it_1s, site_it_1s_from_250ms, atol=1e-5, rtol=1e-6):
        raise RuntimeError("Internal consistency check failed: site_it_1s mismatch.")

    site_it_1min = _downsample_mean(site_it_1s, 60)
    site_it_15min = _downsample_mean(site_it_1s, 900)

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

    metadata = {
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
    with open(os.path.join(out_dir, "aggregation_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(description="Experiment 2c: aggregate node traces to rack/row/site.")
    parser.add_argument("--node-trace-dir", default=defaults["node_trace_dir"])
    parser.add_argument("--out-dir", default=defaults["out_dir"])
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--nodes-per-rack", type=int, default=4)
    parser.add_argument("--non-gpu-overhead-w", type=float, default=1000.0)
    parser.add_argument("--pue", type=float, default=1.3)
    args = parser.parse_args()

    meta = aggregate_facility_traces(
        node_trace_dir=str(args.node_trace_dir),
        out_dir=str(args.out_dir),
        dt=float(args.dt),
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        nodes_per_rack=int(args.nodes_per_rack),
        non_gpu_overhead_w=float(args.non_gpu_overhead_w),
        pue=float(args.pue),
    )

    print("=" * 72)
    print("Azure Hierarchical Aggregation (Experiment 2c)")
    print("=" * 72)
    print(f"Node traces         : {meta['node_trace_dir']}")
    print(f"Output dir          : {meta['out_dir']}")
    print(f"Nodes               : {meta['layout']['n_nodes']}")
    print(f"Site peak (kW)      : {meta['power']['site_peak_w'] / 1000.0:.3f}")
    print(f"Site avg (kW)       : {meta['power']['site_avg_w'] / 1000.0:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
