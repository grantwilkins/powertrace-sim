"""
Tests for scripts/eval/hierarchy_figure.py.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from hierarchy_figure import (  # noqa: E402
    generate_hierarchy_figure,
    generate_hierarchy_outputs,
    generate_hierarchy_panels,
)


def _downsample_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    assert x.size % factor == 0
    return np.mean(x.reshape(-1, factor), axis=1)


def _write_valid_fixture(node_dir: Path, agg_dir: Path) -> None:
    node_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    n_1s = 1800
    n_250ms = n_1s * 4
    t250 = np.arange(n_250ms, dtype=np.float64)
    server_250 = 500.0 + (40.0 * np.sin(2.0 * np.pi * t250 / n_250ms))
    server_1s = _downsample_mean(server_250, 4)
    rack_1s = (4.0 * server_1s) + 120.0
    row_1s = 6.0 * rack_1s
    site_1s = 10.0 * row_1s

    np.save(node_dir / "node_0_0_0.npy", np.asarray(server_250, dtype=np.float32))
    np.save(agg_dir / "rack_0_0.npy", np.asarray(rack_1s, dtype=np.float32))
    np.save(agg_dir / "row_0.npy", np.asarray(row_1s, dtype=np.float32))
    np.save(agg_dir / "site_it_1s.npy", np.asarray(site_1s, dtype=np.float32))


def test_hierarchy_figure_outputs_exist_and_levels_are_present():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_dir = root / "node_traces"
        agg_dir = root / "aggregated"
        _write_valid_fixture(node_dir, agg_dir)

        out_plot = root / "hierarchy.pdf"
        out_csv = root / "hierarchy.csv"
        out_json = root / "hierarchy.json"
        result = generate_hierarchy_figure(
            node_trace_dir=str(node_dir),
            aggregated_dir=str(agg_dir),
            out_plot=str(out_plot),
            out_csv=str(out_csv),
            out_json=str(out_json),
            node_id="0_0_0",
            rack_id="0_0",
            row_id="0",
            server_downsample_factor=4,
            trend_window_s=900,
        )

        assert result["status"] == "ok"
        assert out_plot.exists()
        assert out_csv.exists()
        assert out_json.exists()

        with open(out_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert {str(r["level"]) for r in rows} == {"server", "rack", "row", "site"}


def test_hierarchy_panels_separate_mode_creates_four_panel_files():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_dir = root / "node_traces"
        agg_dir = root / "aggregated"
        _write_valid_fixture(node_dir, agg_dir)

        panel_dir = root / "panels"
        out_csv = root / "hierarchy.csv"
        out_json = root / "hierarchy.json"
        result = generate_hierarchy_panels(
            node_trace_dir=str(node_dir),
            aggregated_dir=str(agg_dir),
            panel_out_dir=str(panel_dir),
            out_csv=str(out_csv),
            out_json=str(out_json),
            node_id="0_0_0",
            rack_id="0_0",
            row_id="0",
            server_downsample_factor=4,
            trend_window_s=900,
        )

        assert result["status"] == "ok"
        assert result["output_mode"] == "separate"
        assert out_csv.exists()
        assert out_json.exists()
        assert set(result["panel_files"].keys()) == {"server", "rack", "row", "site"}
        for level in ("server", "rack", "row", "site"):
            assert Path(result["panel_files"][level]).exists()


def test_hierarchy_outputs_both_mode_creates_combined_and_panels():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_dir = root / "node_traces"
        agg_dir = root / "aggregated"
        _write_valid_fixture(node_dir, agg_dir)

        out_plot = root / "combined.pdf"
        panel_dir = root / "panels"
        out_csv = root / "hierarchy.csv"
        out_json = root / "hierarchy.json"
        result = generate_hierarchy_outputs(
            node_trace_dir=str(node_dir),
            aggregated_dir=str(agg_dir),
            out_plot=str(out_plot),
            panel_out_dir=str(panel_dir),
            out_csv=str(out_csv),
            out_json=str(out_json),
            output_mode="both",
            node_id="0_0_0",
            rack_id="0_0",
            row_id="0",
            server_downsample_factor=4,
            trend_window_s=900,
        )

        assert result["status"] == "ok"
        assert result["output_mode"] == "both"
        assert out_plot.exists()
        assert out_csv.exists()
        assert out_json.exists()
        for level in ("server", "rack", "row", "site"):
            assert Path(result["panel_files"][level]).exists()


def test_hierarchy_figure_rejects_invalid_downsample_divisibility():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_dir = root / "node_traces"
        agg_dir = root / "aggregated"
        node_dir.mkdir(parents=True, exist_ok=True)
        agg_dir.mkdir(parents=True, exist_ok=True)

        # Not divisible by downsample factor 4.
        np.save(node_dir / "node_0_0_0.npy", np.asarray(np.ones((10,), dtype=np.float32)))
        np.save(agg_dir / "rack_0_0.npy", np.asarray(np.ones((2,), dtype=np.float32)))
        np.save(agg_dir / "row_0.npy", np.asarray(np.ones((2,), dtype=np.float32)))
        np.save(agg_dir / "site_it_1s.npy", np.asarray(np.ones((2,), dtype=np.float32)))

        with pytest.raises(ValueError, match="not divisible"):
            generate_hierarchy_figure(
                node_trace_dir=str(node_dir),
                aggregated_dir=str(agg_dir),
                out_plot=str(root / "plot.pdf"),
                out_csv=str(root / "rows.csv"),
                out_json=str(root / "summary.json"),
                node_id="0_0_0",
                rack_id="0_0",
                row_id="0",
                server_downsample_factor=4,
                trend_window_s=1,
            )


def test_hierarchy_figure_rejects_length_mismatch():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_dir = root / "node_traces"
        agg_dir = root / "aggregated"
        node_dir.mkdir(parents=True, exist_ok=True)
        agg_dir.mkdir(parents=True, exist_ok=True)

        np.save(node_dir / "node_0_0_0.npy", np.asarray(np.ones((16,), dtype=np.float32)))
        # server_1s length after /4 is 4, but others are 5.
        np.save(agg_dir / "rack_0_0.npy", np.asarray(np.ones((5,), dtype=np.float32)))
        np.save(agg_dir / "row_0.npy", np.asarray(np.ones((5,), dtype=np.float32)))
        np.save(agg_dir / "site_it_1s.npy", np.asarray(np.ones((5,), dtype=np.float32)))

        with pytest.raises(ValueError, match="matching 1s lengths"):
            generate_hierarchy_figure(
                node_trace_dir=str(node_dir),
                aggregated_dir=str(agg_dir),
                out_plot=str(root / "plot.pdf"),
                out_csv=str(root / "rows.csv"),
                out_json=str(root / "summary.json"),
                node_id="0_0_0",
                rack_id="0_0",
                row_id="0",
                server_downsample_factor=4,
                trend_window_s=1,
            )
