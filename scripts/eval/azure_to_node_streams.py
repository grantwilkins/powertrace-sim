#!/usr/bin/env python3
"""
Experiment 2a: Aggregate Azure day trace -> per-node request streams.

Policy:
  - Aggregate trace input is randomly thinned across facility nodes.
  - Each request is assigned to exactly one node (uniform random over nodes).
  - No time offsets, circular shifts, or request duplication.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

    def node_id_to_coords(self, node_id: int) -> Tuple[int, int, int]:
        npr = int(self.nodes_per_rack)
        rpr = int(self.racks_per_row)
        per_row = rpr * npr
        row = int(node_id) // per_row
        rem = int(node_id) % per_row
        rack = rem // npr
        node = rem % npr
        return int(row), int(rack), int(node)

    def iter_node_ids(self) -> Sequence[int]:
        return range(self.n_nodes)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    in_csv = repo_root / "data" / "azure_trace" / "parsed" / "day_2024-05-16_requests.csv"
    out_dir = repo_root / "data" / "azure_facility" / "node_streams"
    return {
        "input_csv": str(in_csv),
        "output_dir": str(out_dir),
        "manifest_csv": str(out_dir / "stream_manifest.csv"),
        "summary_json": str(out_dir / "stream_summary.json"),
    }


def _validate_required_columns(fieldnames: Sequence[str], required: Sequence[str]) -> None:
    missing = [c for c in required if c not in set(fieldnames)]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}. Found: {list(fieldnames)}")


def split_azure_requests_to_nodes(
    *,
    input_csv: str,
    output_dir: str,
    manifest_csv: str,
    summary_json: str,
    layout: FacilityLayout,
    seed: int = 42,
) -> Dict[str, object]:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if layout.n_nodes <= 0:
        raise ValueError("Facility layout yields zero nodes.")

    _ensure_dir(output_dir)

    rng = np.random.default_rng(int(seed))
    n_nodes = int(layout.n_nodes)

    # Per-node stats (arrays keep this fast for large traces).
    count_by_node = np.zeros((n_nodes,), dtype=np.int64)
    sum_in_by_node = np.zeros((n_nodes,), dtype=np.int64)
    sum_out_by_node = np.zeros((n_nodes,), dtype=np.int64)
    min_arrival_by_node = np.full((n_nodes,), np.inf, dtype=np.float64)
    max_arrival_by_node = np.full((n_nodes,), -np.inf, dtype=np.float64)

    global_count = 0
    global_sum_in = 0
    global_sum_out = 0
    global_min_arrival = float("inf")
    global_max_arrival = float("-inf")

    file_handles = {}
    writers = {}
    node_paths: Dict[int, str] = {}
    try:
        # Open all node output files once and stream-write assignments.
        for node_id in layout.iter_node_ids():
            row, rack, node = layout.node_id_to_coords(int(node_id))
            out_path = os.path.join(output_dir, f"node_{row}_{rack}_{node}.csv")
            f = open(out_path, "w", newline="")
            w = csv.DictWriter(f, fieldnames=["arrival_time", "n_in", "n_out"])
            w.writeheader()
            file_handles[int(node_id)] = f
            writers[int(node_id)] = w
            node_paths[int(node_id)] = out_path

        with open(input_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("Input CSV is empty or missing header.")
            _validate_required_columns(reader.fieldnames, ["arrival_time", "n_in", "n_out"])

            for row_idx, row in enumerate(reader, start=2):
                try:
                    arrival_time = float(row["arrival_time"])
                    n_in = int(float(row["n_in"]))
                    n_out = int(float(row["n_out"]))
                except Exception as exc:
                    raise ValueError(f"Failed parsing input row {row_idx}: {exc}") from exc

                if (not np.isfinite(arrival_time)) or arrival_time < 0.0:
                    raise ValueError(
                        f"Invalid arrival_time at row {row_idx}: {arrival_time} (must be finite and >= 0)."
                    )
                if n_in < 0 or n_out < 0:
                    raise ValueError(
                        f"Invalid token counts at row {row_idx}: n_in={n_in}, n_out={n_out} (must be >= 0)."
                    )

                node_id = int(rng.integers(0, n_nodes))
                writers[node_id].writerow(
                    {
                        "arrival_time": f"{arrival_time:.9f}",
                        "n_in": int(n_in),
                        "n_out": int(n_out),
                    }
                )

                count_by_node[node_id] += 1
                sum_in_by_node[node_id] += int(n_in)
                sum_out_by_node[node_id] += int(n_out)
                if arrival_time < min_arrival_by_node[node_id]:
                    min_arrival_by_node[node_id] = float(arrival_time)
                if arrival_time > max_arrival_by_node[node_id]:
                    max_arrival_by_node[node_id] = float(arrival_time)

                global_count += 1
                global_sum_in += int(n_in)
                global_sum_out += int(n_out)
                if arrival_time < global_min_arrival:
                    global_min_arrival = float(arrival_time)
                if arrival_time > global_max_arrival:
                    global_max_arrival = float(arrival_time)
    finally:
        for f in file_handles.values():
            f.close()

    if global_count <= 0:
        raise ValueError("Input CSV contained no request rows.")

    # Per-node manifest.
    manifest_rows: List[Dict[str, object]] = []
    for node_id in layout.iter_node_ids():
        node_id = int(node_id)
        row, rack, node = layout.node_id_to_coords(node_id)
        node_count = int(count_by_node[node_id])
        if node_count > 0:
            min_arrival = float(min_arrival_by_node[node_id])
            max_arrival = float(max_arrival_by_node[node_id])
            span = float(max_arrival - min_arrival)
        else:
            min_arrival = float("nan")
            max_arrival = float("nan")
            span = float("nan")

        manifest_rows.append(
            {
                "node_id": node_id,
                "row": int(row),
                "rack": int(rack),
                "node": int(node),
                "file": os.path.basename(node_paths[node_id]),
                "num_requests": node_count,
                "sum_n_in": int(sum_in_by_node[node_id]),
                "sum_n_out": int(sum_out_by_node[node_id]),
                "min_arrival_time": min_arrival,
                "max_arrival_time": max_arrival,
                "span_seconds": span,
            }
        )

    manifest_rows.sort(key=lambda r: int(r["node_id"]))
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_id",
                "row",
                "rack",
                "node",
                "file",
                "num_requests",
                "sum_n_in",
                "sum_n_out",
                "min_arrival_time",
                "max_arrival_time",
                "span_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Hard invariants.
    union_min = float(np.min(min_arrival_by_node[np.isfinite(min_arrival_by_node)]))
    union_max = float(np.max(max_arrival_by_node[np.isfinite(max_arrival_by_node)]))
    checks = {
        "count_conservation": int(np.sum(count_by_node)) == int(global_count),
        "input_token_conservation": int(np.sum(sum_in_by_node)) == int(global_sum_in),
        "output_token_conservation": int(np.sum(sum_out_by_node)) == int(global_sum_out),
        "arrival_min_match": math.isclose(union_min, global_min_arrival, rel_tol=0.0, abs_tol=1e-9),
        "arrival_max_match": math.isclose(union_max, global_max_arrival, rel_tol=0.0, abs_tol=1e-9),
    }
    all_passed = bool(all(checks.values()))

    summary = {
        "status": "ok" if all_passed else "failed",
        "input_csv": input_csv,
        "output_dir": output_dir,
        "manifest_csv": manifest_csv,
        "layout": {
            "rows": int(layout.rows),
            "racks_per_row": int(layout.racks_per_row),
            "nodes_per_rack": int(layout.nodes_per_rack),
            "n_nodes": int(layout.n_nodes),
        },
        "seed": int(seed),
        "global": {
            "num_requests": int(global_count),
            "sum_n_in": int(global_sum_in),
            "sum_n_out": int(global_sum_out),
            "min_arrival_time": float(global_min_arrival),
            "max_arrival_time": float(global_max_arrival),
            "span_seconds": float(global_max_arrival - global_min_arrival),
        },
        "node_stats": {
            "min_requests_per_node": int(np.min(count_by_node)),
            "max_requests_per_node": int(np.max(count_by_node)),
            "mean_requests_per_node": float(np.mean(count_by_node)),
            "std_requests_per_node": float(np.std(count_by_node)),
        },
        "checks": checks,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if not all_passed:
        raise RuntimeError(f"Invariant checks failed for node stream split: {checks}")
    return summary


def main() -> None:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(description="Experiment 2a: Azure aggregate -> per-node streams.")
    parser.add_argument("--input-csv", default=defaults["input_csv"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--manifest-csv", default=defaults["manifest_csv"])
    parser.add_argument("--summary-json", default=defaults["summary_json"])
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--racks-per-row", type=int, default=6)
    parser.add_argument("--nodes-per-rack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layout = FacilityLayout(
        rows=int(args.rows),
        racks_per_row=int(args.racks_per_row),
        nodes_per_rack=int(args.nodes_per_rack),
    )
    summary = split_azure_requests_to_nodes(
        input_csv=str(args.input_csv),
        output_dir=str(args.output_dir),
        manifest_csv=str(args.manifest_csv),
        summary_json=str(args.summary_json),
        layout=layout,
        seed=int(args.seed),
    )

    print("=" * 72)
    print("Azure -> Node Streams (Experiment 2a)")
    print("=" * 72)
    print(f"Input CSV          : {summary['input_csv']}")
    print(f"Output dir         : {summary['output_dir']}")
    print(f"Manifest CSV       : {summary['manifest_csv']}")
    print(f"Nodes              : {summary['layout']['n_nodes']}")
    print(f"Requests           : {summary['global']['num_requests']:,}")
    print(
        f"Per-node reqs      : min={summary['node_stats']['min_requests_per_node']:,} "
        f"max={summary['node_stats']['max_requests_per_node']:,} "
        f"mean={summary['node_stats']['mean_requests_per_node']:.2f}"
    )
    print(f"Checks passed      : {all(summary['checks'].values())}")
    print("=" * 72)


if __name__ == "__main__":
    main()
