"""Tests for scripts/eval/generate_azure_facility_sizing_table.py."""

import csv
import tempfile
from pathlib import Path

from scripts.eval.generate_azure_facility_sizing_table import (
    generate_azure_facility_sizing_table,
)


def _write_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("tdp_baseline", 1000.0, 1000.0, 1.0, 0.0, 0.0),
        ("mean_baseline", 650.0, 650.0, 1.0, 0.0, 0.0),
        ("splitwise_strict", 820.0, 710.0, 1.1549296, 25.0, -21.0),
        ("ours", 760.0, 680.0, 1.1176471, 18.0, -15.0),
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_kind",
                "resolution_s",
                "n_samples",
                "peak_kw",
                "avg_kw",
                "par",
                "load_factor",
                "ramp_p50_kw_per_step",
                "ramp_p95_abs_kw_per_step",
                "ramp_p99_abs_kw_per_step",
                "ramp_max_up_kw_per_step",
                "ramp_max_down_kw_per_step",
                "ramp_p95_abs_kw_per_s",
                "ldc_p95_kw",
                "ldc_p99_kw",
                "diversity_factor_it",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        for kind, peak, avg, par, up, down in rows:
            writer.writerow(
                {
                    "trace_kind": kind,
                    "resolution_s": 900.0,
                    "n_samples": 96,
                    "peak_kw": peak,
                    "avg_kw": avg,
                    "par": par,
                    "load_factor": avg / peak,
                    "ramp_p50_kw_per_step": 0.0,
                    "ramp_p95_abs_kw_per_step": abs(up),
                    "ramp_p99_abs_kw_per_step": abs(up),
                    "ramp_max_up_kw_per_step": up,
                    "ramp_max_down_kw_per_step": down,
                    "ramp_p95_abs_kw_per_s": abs(up) / 900.0,
                    "ldc_p95_kw": peak,
                    "ldc_p99_kw": peak,
                    "diversity_factor_it": 0.8,
                    "status": "evaluated",
                    "notes": "constant_trace" if "baseline" in kind else "",
                }
            )


def test_generate_sizing_table_dynamic_methods() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        out_csv = root / "results" / "eval_paper" / "azure_facility_sizing_table.csv"
        out_json = root / "results" / "eval_paper" / "azure_facility_sizing_table.json"
        out_tex = root / "results" / "eval_paper" / "azure_facility_sizing_table.tex"
        _write_metrics_csv(metrics_csv)

        result = generate_azure_facility_sizing_table(
            metrics_csv=str(metrics_csv),
            out_csv=str(out_csv),
            out_json=str(out_json),
            out_tex=str(out_tex),
            method_order="tdp_baseline,mean_baseline,splitwise_strict,ours",
            power_factor=0.9,
        )

        assert out_csv.exists()
        assert out_json.exists()
        assert out_tex.exists()

        with open(out_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            rows = list(reader)

        assert headers == [
            "metric",
            "tdp_baseline",
            "mean_baseline",
            "splitwise_strict",
            "ours",
        ]
        assert len(rows) == 5
        assert result["method_order"] == [
            "tdp_baseline",
            "mean_baseline",
            "splitwise_strict",
            "ours",
        ]

        tex_text = out_tex.read_text()
        assert "Splitwise" in tex_text
        assert "Splitwise (Strict Emulation)" not in tex_text
