"""
Tests for scripts/eval/azure_generate_traces.py.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from azure_generate_traces import generate_node_traces  # noqa: E402
from model.tests.test_eval_baselines_scripts import _build_toy_fixture  # noqa: E402


def _write_node_stream(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arrival_time", "n_in", "n_out"])
        writer.writeheader()
        writer.writerows(rows)


def test_generate_node_traces_smoke_and_dense_policy():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _build_toy_fixture(root)

        node_stream_dir = root / "data" / "azure_facility" / "node_streams"
        _write_node_stream(
            str(node_stream_dir / "node_0_0_0.csv"),
            [
                {"arrival_time": 0.0, "n_in": 32, "n_out": 8},
                {"arrival_time": 2.0, "n_in": 48, "n_out": 12},
            ],
        )
        _write_node_stream(
            str(node_stream_dir / "node_0_0_1.csv"),
            [
                {"arrival_time": 1.0, "n_in": 16, "n_out": 4},
                {"arrival_time": 3.0, "n_in": 24, "n_out": 6},
            ],
        )

        out_dir = root / "results" / "azure_facility" / "node_traces"
        summary = generate_node_traces(
            run_manifest=str(fx["run_manifest"]),
            experimental_manifest=str(fx["experimental_manifest"]),
            throughput_db=str(fx["throughput_db"]),
            ar1_params_dir=str(fx["ar1_params_dir"]),
            node_stream_dir=str(node_stream_dir),
            out_dir=str(out_dir),
            config_id=str(fx["config_id"]),
            duration_s=10.0,
            dt=0.25,
            rows=1,
            racks_per_row=1,
            nodes_per_rack=2,
            batch_size=2,
            base_seed=123,
            device="cpu",
            decode_mode="stochastic",
            median_filter_window=1,
        )

        assert summary["status"] == "ok"
        assert summary["counts"]["evaluated_nodes"] == 2
        assert summary["generation"]["uses_ar1"] is False

        p0 = out_dir / "node_0_0_0.npy"
        p1 = out_dir / "node_0_0_1.npy"
        assert p0.exists()
        assert p1.exists()
        arr0 = np.asarray(np.load(p0), dtype=np.float64).reshape(-1)
        arr1 = np.asarray(np.load(p1), dtype=np.float64).reshape(-1)
        assert arr0.shape == (40,)
        assert arr1.shape == (40,)
        assert np.all(np.isfinite(arr0))
        assert np.all(np.isfinite(arr1))

        manifest_csv = out_dir / "trace_manifest.csv"
        assert manifest_csv.exists()
        with open(manifest_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert {r["status"] for r in rows} == {"evaluated"}
        assert {r["uses_ar1"] for r in rows} == {"False"}
