"""
Tests for scripts/eval/aggregation_variance.py.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from aggregation_variance import compute_aggregation_variance  # noqa: E402


def _write_node_traces(
    node_trace_dir: Path,
    traces: np.ndarray,
    *,
    rows: int,
    racks_per_row: int,
    nodes_per_rack: int,
) -> None:
    node_trace_dir.mkdir(parents=True, exist_ok=True)
    n_nodes = int(rows) * int(racks_per_row) * int(nodes_per_rack)
    assert traces.shape[0] == n_nodes
    idx = 0
    for row in range(rows):
        for rack in range(racks_per_row):
            for node in range(nodes_per_rack):
                path = node_trace_dir / f"node_{row}_{rack}_{node}.npy"
                np.save(path, np.asarray(traces[idx], dtype=np.float32))
                idx += 1


def test_independent_traces_follow_inverse_sqrt_scaling():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_dir = root / "node_traces"
        out_plot = root / "cov_plot.pdf"
        out_csv = root / "cov_rows.csv"
        out_json = root / "cov_summary.json"

        rows, racks_per_row, nodes_per_rack = 1, 4, 4  # 16 nodes
        n_nodes = rows * racks_per_row * nodes_per_rack
        t = 3000
        rng = np.random.default_rng(0)
        traces = rng.normal(loc=100.0, scale=10.0, size=(n_nodes, t))
        traces = np.clip(traces, a_min=1.0, a_max=None)
        _write_node_traces(
            node_dir,
            traces,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
        )

        result = compute_aggregation_variance(
            node_trace_dir=str(node_dir),
            out_plot=str(out_plot),
            out_csv=str(out_csv),
            out_json=str(out_json),
            subset_sizes=(1, 4, 16),
            repeats=80,
            seed=7,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
        )
        summary = result["summary_by_subset"]
        cov_1 = float(summary["1"]["cov_mean"])
        cov_4 = float(summary["4"]["cov_mean"])
        cov_16 = float(summary["16"]["cov_mean"])

        assert np.isclose(cov_4 / cov_1, 1.0 / np.sqrt(4.0), atol=0.08)
        assert np.isclose(cov_16 / cov_1, 1.0 / np.sqrt(16.0), atol=0.06)
        assert -0.65 <= float(result["fit_loglog_cov_vs_n"]["slope"]) <= -0.35

        assert out_plot.exists()
        assert out_csv.exists()
        assert out_json.exists()


def test_correlated_traces_collapse_slower_than_independence():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_dir = root / "node_traces"
        out_plot = root / "cov_plot.pdf"
        out_csv = root / "cov_rows.csv"
        out_json = root / "cov_summary.json"

        rows, racks_per_row, nodes_per_rack = 1, 4, 4  # 16 nodes
        n_nodes = rows * racks_per_row * nodes_per_rack
        t = 3000
        rng = np.random.default_rng(1)
        common = rng.normal(loc=100.0, scale=12.0, size=(t,))
        traces = common.reshape(1, -1) + rng.normal(loc=0.0, scale=1.0, size=(n_nodes, t))
        traces = np.clip(traces, a_min=1.0, a_max=None)
        _write_node_traces(
            node_dir,
            traces,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
        )

        result = compute_aggregation_variance(
            node_trace_dir=str(node_dir),
            out_plot=str(out_plot),
            out_csv=str(out_csv),
            out_json=str(out_json),
            subset_sizes=(1, 4, 16),
            repeats=60,
            seed=11,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
        )

        summary = result["summary_by_subset"]
        cov_16 = float(summary["16"]["cov_mean"])
        cov_theory_16 = float(summary["16"]["cov_theory"])
        assert cov_16 > (cov_theory_16 * 1.25)


def test_sampling_is_deterministic_for_fixed_seed():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_dir = root / "node_traces"
        rows, racks_per_row, nodes_per_rack = 1, 4, 4  # 16 nodes
        n_nodes = rows * racks_per_row * nodes_per_rack
        t = 2048
        rng = np.random.default_rng(22)
        traces = rng.normal(loc=120.0, scale=5.0, size=(n_nodes, t))
        traces = np.clip(traces, a_min=1.0, a_max=None)
        _write_node_traces(
            node_dir,
            traces,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
        )

        out_a_plot = root / "a_plot.pdf"
        out_a_csv = root / "a_rows.csv"
        out_a_json = root / "a_summary.json"
        out_b_plot = root / "b_plot.pdf"
        out_b_csv = root / "b_rows.csv"
        out_b_json = root / "b_summary.json"

        compute_aggregation_variance(
            node_trace_dir=str(node_dir),
            out_plot=str(out_a_plot),
            out_csv=str(out_a_csv),
            out_json=str(out_a_json),
            subset_sizes=(1, 4, 16),
            repeats=25,
            seed=1234,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
        )
        compute_aggregation_variance(
            node_trace_dir=str(node_dir),
            out_plot=str(out_b_plot),
            out_csv=str(out_b_csv),
            out_json=str(out_b_json),
            subset_sizes=(1, 4, 16),
            repeats=25,
            seed=1234,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
        )

        assert out_a_csv.read_text() == out_b_csv.read_text()
