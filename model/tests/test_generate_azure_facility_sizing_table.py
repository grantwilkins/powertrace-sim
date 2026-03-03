"""
Tests for scripts/eval/generate_azure_facility_sizing_table.py.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from generate_azure_facility_sizing_table import (  # noqa: E402
    generate_azure_facility_sizing_table,
)


def _write_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_kind",
                "resolution_s",
                "peak_kw",
                "avg_kw",
                "par",
                "load_factor",
                "ramp_max_up_kw_per_step",
                "ramp_max_down_kw_per_step",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trace_kind": "ours",
                "resolution_s": 900.0,
                "peak_kw": 800.0,
                "avg_kw": 600.0,
                "par": 1.333333333333,
                "load_factor": 0.75,
                "ramp_max_up_kw_per_step": 20.0,
                "ramp_max_down_kw_per_step": -10.0,
            }
        )
        writer.writerow(
            {
                "trace_kind": "tdp_baseline",
                "resolution_s": 900.0,
                "peak_kw": 1000.0,
                "avg_kw": 1000.0,
                "par": 1.0,
                "load_factor": 1.0,
                "ramp_max_up_kw_per_step": 0.0,
                "ramp_max_down_kw_per_step": 0.0,
            }
        )
        writer.writerow(
            {
                "trace_kind": "mean_baseline",
                "resolution_s": 900.0,
                "peak_kw": 500.0,
                "avg_kw": 500.0,
                "par": 1.0,
                "load_factor": 1.0,
                "ramp_max_up_kw_per_step": 0.0,
                "ramp_max_down_kw_per_step": 0.0,
            }
        )


def test_interconnection_summary_formulas_rows_and_units():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        site_15min_npy = root / "results" / "azure_facility" / "aggregated" / "site_15min.npy"
        out_csv = root / "results" / "eval_paper" / "azure_facility_sizing_table.csv"
        out_json = root / "results" / "eval_paper" / "azure_facility_sizing_table.json"
        out_tex = root / "results" / "eval_paper" / "azure_facility_sizing_table.tex"
        out_ic_csv = root / "results" / "eval_paper" / "interconnection_summary_table.csv"
        out_ic_json = root / "results" / "eval_paper" / "interconnection_summary_table.json"
        out_ic_tex = root / "results" / "eval_paper" / "interconnection_summary_table.tex"

        _write_metrics_csv(metrics_csv)
        site_w = np.asarray([100000.0, 120000.0, 80000.0, 130000.0, 110000.0], dtype=np.float64)
        site_15min_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(site_15min_npy, np.asarray(site_w, dtype=np.float32))

        result = generate_azure_facility_sizing_table(
            metrics_csv=str(metrics_csv),
            out_csv=str(out_csv),
            out_json=str(out_json),
            out_tex=str(out_tex),
            site_15min_npy=str(site_15min_npy),
            out_interconnect_csv=str(out_ic_csv),
            out_interconnect_json=str(out_ic_json),
            out_interconnect_tex=str(out_ic_tex),
            decimals_interconnect_power=4,
            decimals_interconnect_ratio=4,
            decimals_interconnect_ramp=5,
            decimals_interconnect_energy=2,
        )

        assert out_ic_csv.exists()
        assert out_ic_json.exists()
        assert out_ic_tex.exists()

        interconnect = result["interconnection_summary"]
        rows = interconnect["table_rows"]
        assert len(rows) == 8

        expected_metrics = [
            "Peak facility power (P95 across seeds)",
            "Average facility power",
            "Load factor",
            "Peak-to-average ratio",
            "95th-percentile 15-min ramp (up)",
            "95th-percentile 15-min ramp (down)",
            "Max 15-min ramp (up)",
            "Annual energy (extrapolated)",
        ]
        expected_units = [
            "MW",
            "MW",
            "---",
            "---",
            "MW/15-min",
            "MW/15-min",
            "MW/15-min",
            "MWh/yr",
        ]
        assert [r["metric"] for r in rows] == expected_metrics
        assert [r["unit"] for r in rows] == expected_units

        raw = interconnect["raw_values"]
        assert np.isclose(float(raw["peak_p95_mw"]), 0.128)
        assert np.isclose(float(raw["average_mw"]), 0.6)
        assert np.isclose(float(raw["load_factor"]), 0.75)
        assert np.isclose(float(raw["peak_to_average_ratio"]), 1.333333333333)
        assert np.isclose(float(raw["ramp_p95_up_mw_per_15min"]), 0.0455)
        assert np.isclose(float(raw["ramp_p95_down_mw_per_15min"]), 0.037)
        assert np.isclose(float(raw["ramp_max_up_mw_per_15min"]), 0.05)
        assert np.isclose(float(raw["annual_energy_mwh_per_year"]), 5256.0)

        assert rows[0]["value"] == "0.1280"
        assert rows[1]["value"] == "0.6000"
        assert rows[2]["value"] == "0.7500"
        assert rows[3]["value"] == "1.3333"
        assert rows[4]["value"] == "0.04550"
        assert rows[5]["value"] == "0.03700"
        assert rows[6]["value"] == "0.05000"
        assert rows[7]["value"] == "5256.00"

        with open(out_ic_json, "r") as f:
            payload = json.load(f)
        assert payload["inputs"]["metric_basis"]["trace_kind"] == "ours"
        assert float(payload["inputs"]["metric_basis"]["resolution_s"]) == 900.0
        assert "definitions" in payload
        assert len(payload["table_rows"]) == 8


def test_existing_sizing_table_rows_unchanged():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        site_15min_npy = root / "results" / "azure_facility" / "aggregated" / "site_15min.npy"
        out_csv = root / "results" / "eval_paper" / "azure_facility_sizing_table.csv"
        out_json = root / "results" / "eval_paper" / "azure_facility_sizing_table.json"
        out_tex = root / "results" / "eval_paper" / "azure_facility_sizing_table.tex"
        out_ic_csv = root / "results" / "eval_paper" / "interconnection_summary_table.csv"
        out_ic_json = root / "results" / "eval_paper" / "interconnection_summary_table.json"
        out_ic_tex = root / "results" / "eval_paper" / "interconnection_summary_table.tex"

        _write_metrics_csv(metrics_csv)
        site_15min_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(site_15min_npy, np.asarray([1.0e6, 1.0e6], dtype=np.float32))

        result = generate_azure_facility_sizing_table(
            metrics_csv=str(metrics_csv),
            out_csv=str(out_csv),
            out_json=str(out_json),
            out_tex=str(out_tex),
            site_15min_npy=str(site_15min_npy),
            out_interconnect_csv=str(out_ic_csv),
            out_interconnect_json=str(out_ic_json),
            out_interconnect_tex=str(out_ic_tex),
            decimals_power=2,
            decimals_ratio=2,
            decimals_ramp=2,
            hide_redundant_cells=True,
        )

        expected_rows = [
            {
                "metric": "Peak facility power (MW)",
                "tdp": "1.00",
                "mean": "---",
                "ours": "0.80",
            },
            {
                "metric": "Average facility power (MW)",
                "tdp": "---",
                "mean": "0.50",
                "ours": "0.60",
            },
            {
                "metric": "Peak-to-average ratio",
                "tdp": "1.00",
                "mean": "1.00",
                "ours": "1.33",
            },
            {
                "metric": "Max 15-min ramp rate (MW/hr)",
                "tdp": "0.00",
                "mean": "0.00",
                "ours": "0.08",
            },
            {
                "metric": "Load factor",
                "tdp": "1.00",
                "mean": "1.00",
                "ours": "0.75",
            },
        ]
        assert result["table_rows"] == expected_rows

        with open(out_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows == expected_rows

