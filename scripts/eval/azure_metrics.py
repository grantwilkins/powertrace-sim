#!/usr/bin/env python3
"""
Experiment 2d: Compute facility-level metrics and baselines for Azure traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
                    yield int(row), int(rack), int(node)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_existing_path(path_str: str, base_dir: str) -> Optional[str]:
    raw = Path(path_str)
    if raw.is_absolute():
        return str(raw) if raw.exists() else None
    local = Path(path_str)
    if local.exists():
        return str(local)
    from_base = Path(base_dir) / raw
    if from_base.exists():
        return str(from_base)
    return None


def _resolve_experimental_paths(
    experimental_manifest: Mapping[str, object],
    *,
    config_id: str,
    experimental_base: str,
) -> Tuple[str, str]:
    cfgs = experimental_manifest.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("Invalid experimental manifest format")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"config_id '{config_id}' not found in experimental manifest")
    dataset_path = _resolve_existing_path(str(row.get("dataset_npz", "")), experimental_base)
    split_path = _resolve_existing_path(str(row.get("split_json", "")), experimental_base)
    if dataset_path is None:
        raise ValueError(f"Dataset path not found for '{config_id}'")
    if split_path is None:
        raise ValueError(f"Split path not found for '{config_id}'")
    return dataset_path, split_path


def _load_train_mean_gpu_power_w(
    *,
    config_id: str,
    experimental_manifest_path: str,
    non_gpu_overhead_w: float,
) -> float:
    manifest = _load_json(experimental_manifest_path)
    base = str(Path(experimental_manifest_path).resolve().parent)
    dataset_path, split_path = _resolve_experimental_paths(manifest, config_id=config_id, experimental_base=base)
    split_payload = _load_json(split_path)
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]

    with np.load(dataset_path, allow_pickle=True) as data:
        power_arr = np.asarray(data["power"], dtype=object)
    n_total = int(len(power_arr))

    traces: List[np.ndarray] = []
    for idx in train_indices:
        if idx < 0 or idx >= n_total:
            continue
        p = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
        if p.size > 0:
            traces.append(p.astype(np.float64))
    if len(traces) == 0:
        for i in range(n_total):
            p = np.asarray(power_arr[i], dtype=np.float64).reshape(-1)
            if p.size > 0:
                traces.append(p.astype(np.float64))
    if len(traces) == 0:
        raise ValueError(f"No training traces available for {config_id}")

    flat_total = np.concatenate(traces, axis=0).astype(np.float64)
    flat_gpu = np.clip(flat_total - float(non_gpu_overhead_w), a_min=0.0, a_max=None)
    if flat_gpu.size == 0:
        raise ValueError("Empty train GPU power pool")
    return float(np.mean(flat_gpu))


def _load_array(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required array not found: {path}")
    arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty array: {path}")
    return arr


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


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "aggregated_dir": str(repo_root / "results" / "azure_facility" / "aggregated"),
        "node_trace_dir": str(repo_root / "results" / "azure_facility" / "node_traces"),
        "experimental_manifest": str(repo_root / "results" / "experimental_continuous_v1" / "manifest.json"),
        "metrics_csv": str(repo_root / "results" / "eval_paper" / "azure_facility_metrics.csv"),
        "ldc_csv": str(repo_root / "results" / "eval_paper" / "azure_facility_ldc_15min.csv"),
    }


def compute_azure_facility_metrics(
    *,
    aggregated_dir: str,
    node_trace_dir: str,
    experimental_manifest: str,
    metrics_csv: str,
    ldc_csv: str,
    config_id: str = "deepseek-r1-distill-70b_H100_tp4",
    rows: int = 10,
    racks_per_row: int = 6,
    nodes_per_rack: int = 4,
    tp_gpus: int = 4,
    gpu_tdp_w: float = 700.0,
    non_gpu_overhead_w: float = 1000.0,
    pue: float = 1.3,
) -> Dict[str, object]:
    if int(tp_gpus) <= 0:
        raise ValueError("tp_gpus must be >= 1")
    if float(gpu_tdp_w) <= 0.0:
        raise ValueError("gpu_tdp_w must be > 0")
    if float(non_gpu_overhead_w) < 0.0:
        raise ValueError("non_gpu_overhead_w must be >= 0")
    if float(pue) <= 0.0:
        raise ValueError("pue must be > 0")

    layout = FacilityLayout(rows=int(rows), racks_per_row=int(racks_per_row), nodes_per_rack=int(nodes_per_rack))
    n_nodes = int(layout.n_nodes)

    site_by_resolution_w = {
        0.25: _load_array(os.path.join(aggregated_dir, "site_250ms.npy")),
        1.0: _load_array(os.path.join(aggregated_dir, "site_1s.npy")),
        60.0: _load_array(os.path.join(aggregated_dir, "site_1min.npy")),
        900.0: _load_array(os.path.join(aggregated_dir, "site_15min.npy")),
    }
    site_it_by_resolution_w = {
        0.25: _load_array(os.path.join(aggregated_dir, "site_it_250ms.npy")),
        1.0: _load_array(os.path.join(aggregated_dir, "site_it_1s.npy")),
        60.0: _load_array(os.path.join(aggregated_dir, "site_it_1min.npy")),
        900.0: _load_array(os.path.join(aggregated_dir, "site_it_15min.npy")),
    }

    diversity_ours = _compute_diversity_factor_it(
        node_trace_dir=node_trace_dir,
        site_it_250ms_w=site_it_by_resolution_w[0.25],
        layout=layout,
        non_gpu_overhead_w=float(non_gpu_overhead_w),
    )

    node_tdp_it_w = float(tp_gpus) * float(gpu_tdp_w) + float(non_gpu_overhead_w)
    site_tdp_w = float(n_nodes) * node_tdp_it_w * float(pue)

    train_mean_gpu_w = _load_train_mean_gpu_power_w(
        config_id=config_id,
        experimental_manifest_path=experimental_manifest,
        non_gpu_overhead_w=float(non_gpu_overhead_w),
    )
    node_mean_it_w = float(train_mean_gpu_w) + float(non_gpu_overhead_w)
    site_mean_w = float(n_nodes) * node_mean_it_w * float(pue)

    rows_out: List[Dict[str, object]] = []
    ldc_rows: List[Dict[str, object]] = []
    for resolution_s in (0.25, 1.0, 60.0, 900.0):
        ours_w = site_by_resolution_w[resolution_s]
        ours_kw = ours_w / 1000.0
        tdp_kw = np.full((ours_kw.size,), fill_value=float(site_tdp_w / 1000.0), dtype=np.float64)
        mean_kw = np.full((ours_kw.size,), fill_value=float(site_mean_w / 1000.0), dtype=np.float64)

        bundles = [
            ("ours", ours_kw, diversity_ours, ""),
            ("tdp_baseline", tdp_kw, 1.0, "constant_trace"),
            ("mean_baseline", mean_kw, 1.0, "constant_trace"),
        ]
        for trace_kind, values_kw, diversity, notes in bundles:
            m = _compute_metrics(values_kw, resolution_s=resolution_s)
            rows_out.append(
                {
                    "trace_kind": str(trace_kind),
                    "resolution_s": float(resolution_s),
                    "n_samples": int(values_kw.size),
                    "peak_kw": float(m["peak_kw"]),
                    "avg_kw": float(m["avg_kw"]),
                    "par": float(m["par"]),
                    "load_factor": float(m["load_factor"]),
                    "ramp_p50_kw_per_step": float(m["ramp_p50_kw_per_step"]),
                    "ramp_p95_abs_kw_per_step": float(m["ramp_p95_abs_kw_per_step"]),
                    "ramp_p99_abs_kw_per_step": float(m["ramp_p99_abs_kw_per_step"]),
                    "ramp_max_up_kw_per_step": float(m["ramp_max_up_kw_per_step"]),
                    "ramp_max_down_kw_per_step": float(m["ramp_max_down_kw_per_step"]),
                    "ramp_p95_abs_kw_per_s": float(m["ramp_p95_abs_kw_per_s"]),
                    "ldc_p95_kw": float(m["ldc_p95_kw"]),
                    "ldc_p99_kw": float(m["ldc_p99_kw"]),
                    "diversity_factor_it": float(diversity),
                    "status": "evaluated",
                    "notes": str(notes),
                }
            )

            if float(resolution_s) == 900.0:
                sorted_desc = np.sort(values_kw)[::-1]
                n = int(sorted_desc.size)
                for i, v in enumerate(sorted_desc):
                    ldc_rows.append(
                        {
                            "trace_kind": str(trace_kind),
                            "resolution_s": float(resolution_s),
                            "rank": int(i),
                            "fraction_exceeded": float(i / max(1, n)),
                            "power_kw": float(v),
                        }
                    )

    _ensure_dir(os.path.dirname(metrics_csv) or ".")
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

    with open(ldc_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_kind", "resolution_s", "rank", "fraction_exceeded", "power_kw"],
        )
        writer.writeheader()
        writer.writerows(ldc_rows)

    summary = {
        "status": "ok",
        "config_id": config_id,
        "aggregated_dir": aggregated_dir,
        "node_trace_dir": node_trace_dir,
        "metrics_csv": metrics_csv,
        "ldc_csv": ldc_csv,
        "n_rows_metrics": int(len(rows_out)),
        "n_rows_ldc": int(len(ldc_rows)),
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
        "ours": {"diversity_factor_it": float(diversity_ours)},
    }
    return summary


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(description="Experiment 2d: compute Azure facility metrics and baselines.")
    parser.add_argument("--aggregated-dir", default=defaults["aggregated_dir"])
    parser.add_argument("--node-trace-dir", default=defaults["node_trace_dir"])
    parser.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    parser.add_argument("--metrics-csv", default=defaults["metrics_csv"])
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

    summary = compute_azure_facility_metrics(
        aggregated_dir=str(args.aggregated_dir),
        node_trace_dir=str(args.node_trace_dir),
        experimental_manifest=str(args.experimental_manifest),
        metrics_csv=str(args.metrics_csv),
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

    print("=" * 72)
    print("Azure Facility Metrics (Experiment 2d)")
    print("=" * 72)
    print(f"Config              : {summary['config_id']}")
    print(f"Metrics CSV         : {summary['metrics_csv']}")
    print(f"LDC CSV             : {summary['ldc_csv']}")
    print(f"Metric rows         : {summary['n_rows_metrics']}")
    print(f"LDC rows            : {summary['n_rows_ldc']}")
    print(f"Diversity factor IT : {summary['ours']['diversity_factor_it']:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
