"""
Tests for scripts/eval/azure_to_node_streams.py.
"""

import csv
import os
import tempfile

from scripts.eval.azure_to_node_streams import split_azure_requests_to_nodes  # noqa: E402
from scripts.eval.facility import FacilityLayout  # noqa: E402


def _write_parsed_csv(path: str, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["request_id", "timestamp_utc", "arrival_time", "n_in", "n_out"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_split_conservation_and_invariants():
    rows = []
    for i in range(20):
        rows.append(
            {
                "request_id": i,
                "timestamp_utc": f"2024-05-16 00:00:{i:02d}+00:00",
                "arrival_time": float(i),
                "n_in": 100 + i,
                "n_out": 10 + (i % 3),
            }
        )

    with tempfile.TemporaryDirectory() as td:
        in_csv = os.path.join(td, "day_requests.csv")
        out_dir = os.path.join(td, "node_streams")
        manifest_csv = os.path.join(out_dir, "stream_manifest.csv")
        summary_json = os.path.join(out_dir, "stream_summary.json")
        _write_parsed_csv(in_csv, rows)

        layout = FacilityLayout(rows=2, racks_per_row=2, nodes_per_rack=2)
        summary = split_azure_requests_to_nodes(
            input_csv=in_csv,
            output_dir=out_dir,
            manifest_csv=manifest_csv,
            summary_json=summary_json,
            layout=layout,
            seed=123,
        )

        assert summary["status"] == "ok"
        assert summary["global"]["num_requests"] == 20
        assert all(summary["checks"].values())
        assert os.path.exists(manifest_csv)
        assert os.path.exists(summary_json)

        count_sum = 0
        n_in_sum = 0
        n_out_sum = 0
        union_min = None
        union_max = None
        for row_i in range(layout.rows):
            for rack_j in range(layout.racks_per_row):
                for node_k in range(layout.nodes_per_rack):
                    path = os.path.join(out_dir, f"node_{row_i}_{rack_j}_{node_k}.csv")
                    assert os.path.exists(path)
                    with open(path, "r", newline="") as f:
                        node_rows = list(csv.DictReader(f))
                    count_sum += len(node_rows)
                    for r in node_rows:
                        n_in_sum += int(r["n_in"])
                        n_out_sum += int(r["n_out"])
                        t = float(r["arrival_time"])
                        union_min = t if union_min is None else min(union_min, t)
                        union_max = t if union_max is None else max(union_max, t)

        assert count_sum == 20
        assert n_in_sum == sum(int(r["n_in"]) for r in rows)
        assert n_out_sum == sum(int(r["n_out"]) for r in rows)
        assert union_min == 0.0
        assert union_max == 19.0


def test_split_is_reproducible_for_fixed_seed():
    rows = []
    for i in range(30):
        rows.append(
            {
                "request_id": i,
                "timestamp_utc": f"2024-05-16 00:00:{i:02d}+00:00",
                "arrival_time": float(i) * 0.5,
                "n_in": 10 + i,
                "n_out": 5 + (i % 4),
            }
        )

    with tempfile.TemporaryDirectory() as td:
        in_csv = os.path.join(td, "day_requests.csv")
        _write_parsed_csv(in_csv, rows)

        layout = FacilityLayout(rows=2, racks_per_row=1, nodes_per_rack=3)
        out_a = os.path.join(td, "out_a")
        out_b = os.path.join(td, "out_b")
        split_azure_requests_to_nodes(
            input_csv=in_csv,
            output_dir=out_a,
            manifest_csv=os.path.join(out_a, "stream_manifest.csv"),
            summary_json=os.path.join(out_a, "stream_summary.json"),
            layout=layout,
            seed=999,
        )
        split_azure_requests_to_nodes(
            input_csv=in_csv,
            output_dir=out_b,
            manifest_csv=os.path.join(out_b, "stream_manifest.csv"),
            summary_json=os.path.join(out_b, "stream_summary.json"),
            layout=layout,
            seed=999,
        )

        for row_i in range(layout.rows):
            for rack_j in range(layout.racks_per_row):
                for node_k in range(layout.nodes_per_rack):
                    p_a = os.path.join(out_a, f"node_{row_i}_{rack_j}_{node_k}.csv")
                    p_b = os.path.join(out_b, f"node_{row_i}_{rack_j}_{node_k}.csv")
                    with open(p_a, "r") as fa, open(p_b, "r") as fb:
                        assert fa.read() == fb.read()
