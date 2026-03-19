"""
Tests for scripts/eval/azure_generate_traces.py.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from azure_defaults import (  # noqa: E402
    DEFAULT_CONFIG_ID,
    DEFAULT_SPLITWISE_SOURCE_HARDWARE,
    DEFAULT_SPLITWISE_SOURCE_MODEL,
    DEFAULT_SPLITWISE_SOURCE_TP,
)
from azure_generate_traces import generate_node_traces  # noqa: E402

from model.tests.test_eval_baselines_scripts import _build_toy_fixture  # noqa: E402


def _write_node_stream(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arrival_time", "n_in", "n_out"])
        writer.writeheader()
        writer.writerows(rows)


def test_default_config_and_splitwise_defaults_match_llama70b_a100_tp8() -> None:
    assert DEFAULT_CONFIG_ID == "llama-3-70b_A100_tp8"
    assert DEFAULT_SPLITWISE_SOURCE_MODEL == "llama-3-70b"
    assert DEFAULT_SPLITWISE_SOURCE_HARDWARE == "a100-80gb"
    assert DEFAULT_SPLITWISE_SOURCE_TP == 8


def test_generate_node_traces_with_splitwise_outputs() -> None:
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

        out_root = root / "results" / "azure_facility" / "node_traces"
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
        assert summary["methods"] == ["ours", "splitwise_strict"]
        assert summary["counts"]["evaluated_by_method"]["ours"] == 2
        assert summary["counts"]["evaluated_by_method"]["splitwise_strict"] == 2
        assert summary["generation"]["uses_ar1"] is False
        assert (
            summary["splitwise"]["splitwise_source_model"]
            == DEFAULT_SPLITWISE_SOURCE_MODEL
        )
        assert (
            summary["splitwise"]["splitwise_source_hardware"]
            == DEFAULT_SPLITWISE_SOURCE_HARDWARE
        )
        assert summary["splitwise"]["splitwise_source_tp"] == 4
        assert (
            summary["splitwise"]["meta"]["splitwise_source_resolved_model"]
            == "llama2-70b"
        )
        assert (
            summary["splitwise"]["meta"]["splitwise_source_match_status"]
            == "family_model_fallback"
        )

        for method in ["ours", "splitwise_strict"]:
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
        assert len(rows) == 4
        assert {row["status"] for row in rows} == {"evaluated"}
        assert {row["method"] for row in rows} == {"ours", "splitwise_strict"}


def test_splitwise_lut_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _build_toy_fixture(root)
        node_stream_dir = root / "data" / "azure_facility" / "node_streams"
        _write_node_stream(
            node_stream_dir / "node_0_0_0.csv",
            [{"arrival_time": 0.0, "n_in": 32, "n_out": 8}],
        )

        with pytest.raises(ValueError, match="splitwise_lut"):
            generate_node_traces(
                run_manifest=str(fx["run_manifest"]),
                experimental_manifest=str(fx["experimental_manifest"]),
                throughput_db=str(fx["throughput_db"]),
                ar1_params_dir=str(fx["ar1_params_dir"]),
                node_stream_dir=str(node_stream_dir),
                out_root=str(root / "results" / "azure_facility" / "node_traces"),
                config_id=str(fx["config_id"]),
                methods="splitwise_lut",
                duration_s=10.0,
                dt=0.25,
                rows=1,
                racks_per_row=1,
                nodes_per_rack=1,
                batch_size=1,
                base_seed=1,
                device="cpu",
            )
