"""
Tests for scripts/eval/oversubscription_figure.py.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from oversubscription_figure import compute_oversubscription_capacity  # noqa: E402


def _write_constant_rack_fixture(
    aggregated_dir: Path,
    *,
    n_racks: int = 8,
    timesteps: int = 3600,
    rack_kw: float = 10.0,
) -> None:
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(int(n_racks)):
        row = idx // 4
        rack = idx % 4
        trace_w = np.full((timesteps,), float(rack_kw) * 1000.0, dtype=np.float32)
        np.save(aggregated_dir / f"rack_{row}_{rack}.npy", trace_w)


def _write_variable_rack_fixture(
    aggregated_dir: Path,
    *,
    n_racks: int = 10,
    timesteps: int = 7200,
) -> None:
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(timesteps, dtype=np.float64)
    for idx in range(int(n_racks)):
        row = idx // 5
        rack = idx % 5
        base_kw = 7.5 + (0.45 * float(idx))
        wave_kw = 0.8 * np.sin((2.0 * np.pi * t / 1800.0) + (0.15 * idx))
        ripple_kw = 0.25 * np.cos((2.0 * np.pi * t / 300.0) + (0.31 * idx))
        trace_kw = np.clip(base_kw + wave_kw + ripple_kw, a_min=0.5, a_max=None)
        np.save(aggregated_dir / f"rack_{row}_{rack}.npy", np.asarray(trace_kw * 1000.0, dtype=np.float32))


def test_auto_derived_counts_and_outputs_are_valid():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        aggregated_dir = root / "aggregated"
        _write_constant_rack_fixture(aggregated_dir, n_racks=8, rack_kw=10.0)

        out_capacity_plot = root / "capacity.pdf"
        out_lines_plot = root / "lines.pdf"
        out_csv = root / "capacity.csv"
        out_json = root / "capacity.json"
        result = compute_oversubscription_capacity(
            aggregated_dir=str(aggregated_dir),
            out_capacity_plot=str(out_capacity_plot),
            out_lines_plot=str(out_lines_plot),
            out_csv=str(out_csv),
            out_json=str(out_json),
            row_limit_kw=60.0,
            rack_tdp_kw=12.0,
            risk_percentile=95.0,
            seed=17,
            samples_per_count=40,
            trace_samples=20,
        )

        assert result["status"] == "ok"
        assert int(result["selection"]["tdp_racks"]) == 5
        assert int(result["selection"]["oversub_racks"]) == 6
        assert out_capacity_plot.exists()
        assert out_lines_plot.exists()
        assert out_csv.exists()
        assert out_json.exists()

        rows_by_n = {int(r["n_racks"]): r for r in result["summary_rows"]}
        assert float(rows_by_n[6]["peak_prisk_kw"]) <= 60.0 + 1e-12
        assert float(rows_by_n[7]["peak_prisk_kw"]) > 60.0


def test_capacity_csv_has_expected_columns_and_candidate_range():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        aggregated_dir = root / "aggregated"
        _write_variable_rack_fixture(aggregated_dir, n_racks=6, timesteps=3600)

        out_capacity_plot = root / "capacity.pdf"
        out_lines_plot = root / "lines.pdf"
        out_csv = root / "capacity.csv"
        out_json = root / "capacity.json"
        result = compute_oversubscription_capacity(
            aggregated_dir=str(aggregated_dir),
            out_capacity_plot=str(out_capacity_plot),
            out_lines_plot=str(out_lines_plot),
            out_csv=str(out_csv),
            out_json=str(out_json),
            row_limit_kw=400.0,
            rack_tdp_kw=90.0,
            risk_percentile=95.0,
            seed=123,
            samples_per_count=30,
            trace_samples=12,
        )

        assert int(result["dataset"]["n_racks_available"]) == 6
        with open(out_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 6
        assert list(int(r["n_racks"]) for r in rows) == [1, 2, 3, 4, 5, 6]
        expected_cols = {
            "n_racks",
            "n_samples",
            "peak_mean_kw",
            "peak_p05_kw",
            "peak_p50_kw",
            "peak_p95_kw",
            "peak_prisk_kw",
            "peak_max_kw",
            "exceed_prob",
        }
        assert expected_cols.issubset(set(rows[0].keys()))


def test_fixed_seed_reproducibility():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        aggregated_dir = root / "aggregated"
        _write_variable_rack_fixture(aggregated_dir, n_racks=9, timesteps=3600)

        kwargs = dict(
            aggregated_dir=str(aggregated_dir),
            row_limit_kw=300.0,
            rack_tdp_kw=55.0,
            risk_percentile=95.0,
            seed=999,
            samples_per_count=25,
            trace_samples=10,
        )

        compute_oversubscription_capacity(
            out_capacity_plot=str(root / "a_capacity.pdf"),
            out_lines_plot=str(root / "a_lines.pdf"),
            out_csv=str(root / "a.csv"),
            out_json=str(root / "a.json"),
            **kwargs,
        )
        second = compute_oversubscription_capacity(
            out_capacity_plot=str(root / "b_capacity.pdf"),
            out_lines_plot=str(root / "b_lines.pdf"),
            out_csv=str(root / "b.csv"),
            out_json=str(root / "b.json"),
            **kwargs,
        )

        assert (root / "a.csv").read_text() == (root / "b.csv").read_text()
        assert int(second["selection"]["tdp_racks"]) >= 1
        assert int(second["selection"]["oversub_racks"]) >= 1


def test_invalid_inputs_raise_clear_errors():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        empty_agg = root / "empty_aggregated"
        empty_agg.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="No rack traces found"):
            compute_oversubscription_capacity(
                aggregated_dir=str(empty_agg),
                out_capacity_plot=str(root / "capacity.pdf"),
                out_lines_plot=str(root / "lines.pdf"),
                out_csv=str(root / "rows.csv"),
                out_json=str(root / "summary.json"),
            )

        aggregated_dir = root / "aggregated"
        _write_constant_rack_fixture(aggregated_dir, n_racks=4, rack_kw=8.0)

        with pytest.raises(ValueError, match="row_limit_kw must be positive"):
            compute_oversubscription_capacity(
                aggregated_dir=str(aggregated_dir),
                out_capacity_plot=str(root / "capacity2.pdf"),
                out_lines_plot=str(root / "lines2.pdf"),
                out_csv=str(root / "rows2.csv"),
                out_json=str(root / "summary2.json"),
                row_limit_kw=0.0,
            )

        with pytest.raises(ValueError, match="rack_tdp_kw must be positive"):
            compute_oversubscription_capacity(
                aggregated_dir=str(aggregated_dir),
                out_capacity_plot=str(root / "capacity3.pdf"),
                out_lines_plot=str(root / "lines3.pdf"),
                out_csv=str(root / "rows3.csv"),
                out_json=str(root / "summary3.json"),
                rack_tdp_kw=-1.0,
            )

        with pytest.raises(ValueError, match="risk_percentile must be in"):
            compute_oversubscription_capacity(
                aggregated_dir=str(aggregated_dir),
                out_capacity_plot=str(root / "capacity4.pdf"),
                out_lines_plot=str(root / "lines4.pdf"),
                out_csv=str(root / "rows4.csv"),
                out_json=str(root / "summary4.json"),
                risk_percentile=0.0,
            )
