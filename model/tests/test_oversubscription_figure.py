"""
Tests for scripts/eval/oversubscription_figure.py.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from oversubscription_figure import METHOD_LABEL, compute_oversubscription_capacity  # noqa: E402


def _write_constant_method_fixture(
    method_dir: Path,
    *,
    n_racks: int,
    timesteps: int,
    rack_kw: float,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(int(n_racks)):
        row = idx // 4
        rack = idx % 4
        trace_w = np.full((timesteps,), float(rack_kw) * 1000.0, dtype=np.float32)
        np.save(method_dir / f"rack_{row}_{rack}.npy", trace_w)


def _write_metrics_csv(path: Path, *, mean_avg_kw: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_kind", "resolution_s", "avg_kw"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trace_kind": "mean_baseline",
                "resolution_s": 900.0,
                "avg_kw": float(mean_avg_kw),
            }
        )


def test_oversubscription_selection_by_method() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        aggregated_root = root / "results" / "azure_facility" / "aggregated"
        _write_constant_method_fixture(
            aggregated_root / "ours",
            n_racks=8,
            timesteps=3600,
            rack_kw=9.0,
        )
        _write_constant_method_fixture(
            aggregated_root / "splitwise_strict",
            n_racks=8,
            timesteps=3600,
            rack_kw=12.0,
        )

        metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        _write_metrics_csv(metrics_csv, mean_avg_kw=80.0)

        out_capacity_plot = root / "figures" / "azure_oversubscription_capacity.pdf"
        out_lines_plot = root / "figures" / "azure_oversubscription_lines.pdf"
        out_csv = root / "results" / "eval_paper" / "azure_oversubscription_capacity.csv"
        out_json = root / "results" / "eval_paper" / "azure_oversubscription_capacity.json"

        result = compute_oversubscription_capacity(
            aggregated_root=str(aggregated_root),
            metrics_csv=str(metrics_csv),
            out_capacity_plot=str(out_capacity_plot),
            out_lines_plot=str(out_lines_plot),
            out_csv=str(out_csv),
            out_json=str(out_json),
            row_limit_kw=60.0,
            rack_tdp_kw=12.0,
            risk_percentile=95.0,
            seed=11,
            samples_per_count=25,
            trace_samples=10,
        )

        assert result["status"] == "ok"
        assert METHOD_LABEL["splitwise_strict"] == "Splitwise"
        assert out_capacity_plot.exists()
        assert out_lines_plot.exists()
        assert out_csv.exists()
        assert out_json.exists()

        selection = result["selection"]
        assert int(selection["tdp_racks"]) == 5
        assert np.isclose(float(selection["tdp_row_kw"]), 60.0)
        assert int(selection["selection_by_method"]["ours"]["oversub_racks"]) == 6
        assert int(selection["selection_by_method"]["mean_baseline"]["oversub_racks"]) == 6
        assert int(selection["selection_by_method"]["splitwise_strict"]["oversub_racks"]) == 5
        assert np.isclose(
            float(selection["selection_by_method"]["mean_baseline"]["peak_prisk_kw"]),
            60.0,
        )

        with open(out_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 24
        assert {row["method"] for row in rows} == {
            "mean_baseline",
            "splitwise_strict",
            "ours",
        }

        rows_by_key = {(row["method"], int(row["n_racks"])): row for row in rows}
        assert float(rows_by_key[("splitwise_strict", 5)]["peak_prisk_kw"]) <= 60.0 + 1e-12
        assert float(rows_by_key[("splitwise_strict", 6)]["peak_prisk_kw"]) > 60.0

        with open(out_json, "r") as f:
            payload = json.load(f)
        assert payload["dataset"]["reference_method"] == "ours"
        assert payload["inputs"]["methods"] == [
            "mean_baseline",
            "splitwise_strict",
            "ours",
        ]


def test_oversubscription_extrapolates_beyond_observed_racks() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        aggregated_root = root / "results" / "azure_facility" / "aggregated"
        _write_constant_method_fixture(
            aggregated_root / "ours",
            n_racks=8,
            timesteps=3600,
            rack_kw=5.0,
        )
        _write_constant_method_fixture(
            aggregated_root / "splitwise_strict",
            n_racks=8,
            timesteps=3600,
            rack_kw=10.0,
        )

        metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        _write_metrics_csv(metrics_csv, mean_avg_kw=64.0)

        result = compute_oversubscription_capacity(
            aggregated_root=str(aggregated_root),
            metrics_csv=str(metrics_csv),
            out_capacity_plot=str(root / "capacity.pdf"),
            out_lines_plot=str(root / "lines.pdf"),
            out_csv=str(root / "capacity.csv"),
            out_json=str(root / "capacity.json"),
            row_limit_kw=60.0,
            rack_tdp_kw=12.0,
            risk_percentile=95.0,
            seed=3,
            samples_per_count=20,
            trace_samples=8,
            max_racks_to_evaluate=16,
        )

        selection = result["selection"]["selection_by_method"]
        assert int(result["dataset"]["n_racks_available"]) == 8
        assert int(result["dataset"]["max_racks_evaluated"]) == 13
        assert int(selection["ours"]["oversub_racks"]) == 12
        assert bool(selection["ours"]["used_extrapolation"]) is True
        assert int(selection["splitwise_strict"]["oversub_racks"]) == 6
        assert bool(selection["splitwise_strict"]["used_extrapolation"]) is False


def test_fixed_seed_reproducibility() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        aggregated_root = root / "results" / "azure_facility" / "aggregated"
        _write_constant_method_fixture(
            aggregated_root / "ours",
            n_racks=8,
            timesteps=3600,
            rack_kw=7.0,
        )
        _write_constant_method_fixture(
            aggregated_root / "splitwise_strict",
            n_racks=8,
            timesteps=3600,
            rack_kw=8.0,
        )
        metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        _write_metrics_csv(metrics_csv, mean_avg_kw=60.0)

        kwargs = dict(
            aggregated_root=str(aggregated_root),
            metrics_csv=str(metrics_csv),
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
        assert int(second["selection"]["selection_by_method"]["ours"]["oversub_racks"]) >= 1


def test_invalid_inputs_raise_clear_errors() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        empty_agg = root / "empty_aggregated"
        empty_agg.mkdir(parents=True, exist_ok=True)

        metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        _write_metrics_csv(metrics_csv, mean_avg_kw=50.0)

        with pytest.raises(ValueError, match="At least one trace-backed method"):
            compute_oversubscription_capacity(
                aggregated_root=str(empty_agg),
                metrics_csv=str(metrics_csv),
                out_capacity_plot=str(root / "capacity.pdf"),
                out_lines_plot=str(root / "lines.pdf"),
                out_csv=str(root / "rows.csv"),
                out_json=str(root / "summary.json"),
            )

        aggregated_root = root / "aggregated"
        _write_constant_method_fixture(
            aggregated_root / "ours",
            n_racks=4,
            timesteps=3600,
            rack_kw=8.0,
        )
        _write_constant_method_fixture(
            aggregated_root / "splitwise_strict",
            n_racks=4,
            timesteps=3600,
            rack_kw=9.0,
        )

        with pytest.raises(ValueError, match="row_limit_kw must be positive"):
            compute_oversubscription_capacity(
                aggregated_root=str(aggregated_root),
                metrics_csv=str(metrics_csv),
                out_capacity_plot=str(root / "capacity2.pdf"),
                out_lines_plot=str(root / "lines2.pdf"),
                out_csv=str(root / "rows2.csv"),
                out_json=str(root / "summary2.json"),
                row_limit_kw=0.0,
            )

        with pytest.raises(ValueError, match="rack_tdp_kw must be positive"):
            compute_oversubscription_capacity(
                aggregated_root=str(aggregated_root),
                metrics_csv=str(metrics_csv),
                out_capacity_plot=str(root / "capacity3.pdf"),
                out_lines_plot=str(root / "lines3.pdf"),
                out_csv=str(root / "rows3.csv"),
                out_json=str(root / "summary3.json"),
                rack_tdp_kw=-1.0,
            )

        with pytest.raises(ValueError, match="risk_percentile must be in"):
            compute_oversubscription_capacity(
                aggregated_root=str(aggregated_root),
                metrics_csv=str(metrics_csv),
                out_capacity_plot=str(root / "capacity4.pdf"),
                out_lines_plot=str(root / "lines4.pdf"),
                out_csv=str(root / "rows4.csv"),
                out_json=str(root / "summary4.json"),
                risk_percentile=0.0,
            )
