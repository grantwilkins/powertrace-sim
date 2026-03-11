"""Tests for isolated Azure baselines-included trace generation."""

import csv
import tempfile
from pathlib import Path

import numpy as np

from scripts.eval.azure_scripts_baselines_included.azure_generate_traces import (
    generate_node_traces,
)
from model.tests.test_eval_baselines_scripts import _build_toy_fixture


def _write_node_stream(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arrival_time", "n_in", "n_out"])
        writer.writeheader()
        writer.writerows(rows)


def test_generate_node_traces_with_splitwise_isolated_outputs() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _build_toy_fixture(root)

        node_stream_dir = root / "data" / "azure_facility" / "node_streams"
        _write_node_stream(
            node_stream_dir / "node_0_0_0.csv",
            [
                {"arrival_time": 0.0, "n_in": 32, "n_out": 8},
                {"arrival_time": 2.0, "n_in": 48, "n_out": 12},
            ],
        )
        _write_node_stream(
            node_stream_dir / "node_0_0_1.csv",
            [
                {"arrival_time": 1.0, "n_in": 16, "n_out": 4},
                {"arrival_time": 3.0, "n_in": 24, "n_out": 6},
            ],
        )

        out_root = root / "results" / "azure_facility_baselines_included" / "node_traces"
        summary = generate_node_traces(
            run_manifest=str(fx["run_manifest"]),
            experimental_manifest=str(fx["experimental_manifest"]),
            throughput_db=str(fx["throughput_db"]),
            pair_manifest_csv=str(fx["pair_manifest"]),
            splitwise_perf_model_csv=str(fx["perf_model_csv"]),
            ar1_params_dir=str(fx["ar1_params_dir"]),
            node_stream_dir=str(node_stream_dir),
            out_root=str(out_root),
            config_id=str(fx["config_id"]),
            methods="ours,splitwise_lut,splitwise_strict",
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
            tp_gpus=4,
            n_gpus_per_node=4,
        )

        assert summary["status"] == "ok"
        assert summary["methods"] == ["ours", "splitwise_lut", "splitwise_strict"]
        assert summary["counts"]["evaluated_by_method"]["ours"] == 2
        assert summary["counts"]["evaluated_by_method"]["splitwise_lut"] == 2
        assert summary["counts"]["evaluated_by_method"]["splitwise_strict"] == 2

        for method in ["ours", "splitwise_lut", "splitwise_strict"]:
            p0 = out_root / method / "node_0_0_0.npy"
            p1 = out_root / method / "node_0_0_1.npy"
            assert p0.exists()
            assert p1.exists()
            arr0 = np.asarray(np.load(p0), dtype=np.float64).reshape(-1)
            arr1 = np.asarray(np.load(p1), dtype=np.float64).reshape(-1)
            assert arr0.shape == (40,)
            assert arr1.shape == (40,)
            assert np.all(np.isfinite(arr0))
            assert np.all(np.isfinite(arr1))

        manifest_csv = out_root / "trace_manifest.csv"
        assert manifest_csv.exists()
        with open(manifest_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 6
        assert {r["status"] for r in rows} == {"evaluated"}
        assert {r["method"] for r in rows} == {"ours", "splitwise_lut", "splitwise_strict"}
