"""
Tests for scripts/eval/azure_figures.py.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from azure_figures import (  # noqa: E402
    TRACE_KIND_LABEL,
    _compute_baseline_comparison_stats,
    _compute_sizing_metrics_from_rows,
    _load_arrival_rate_binned,
    _load_rack_matrix_15min_kw,
    generate_azure_figures,
)


def _write_parsed_requests_csv(path: Path, bin_seconds: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["request_id", "timestamp_utc", "arrival_time", "n_in", "n_out"],
        )
        writer.writeheader()
        req_id = 0
        for bucket in range(288):
            n_req = (bucket % 3) + 1
            t = (bucket * bin_seconds) + 1.0
            for _ in range(n_req):
                writer.writerow(
                    {
                        "request_id": req_id,
                        "timestamp_utc": "2024-05-16 00:00:00+00:00",
                        "arrival_time": f"{t:.6f}",
                        "n_in": 100,
                        "n_out": 10,
                    }
                )
                req_id += 1


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        writer.writerows(rows)


def _write_ldc_csv(path: Path, series_kw: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_kind", "resolution_s", "rank", "fraction_exceeded", "power_kw"],
        )
        writer.writeheader()
        for trace_kind, arr in series_kw.items():
            sorted_kw = np.sort(np.asarray(arr, dtype=np.float64))[::-1]
            n = int(sorted_kw.size)
            for rank in range(n):
                writer.writerow(
                    {
                        "trace_kind": trace_kind,
                        "resolution_s": 900.0,
                        "rank": rank,
                        "fraction_exceeded": rank / float(max(1, n)),
                        "power_kw": float(sorted_kw[rank]),
                    }
                )


def _write_site_traces_csv(path: Path, series_kw: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_kind", "bin_idx", "hour", "power_kw", "power_mw"],
        )
        writer.writeheader()
        for trace_kind, arr in series_kw.items():
            x = np.asarray(arr, dtype=np.float64).reshape(-1)
            for idx in range(x.size):
                hour = (idx + 0.5) * (900.0 / 3600.0)
                writer.writerow(
                    {
                        "trace_kind": trace_kind,
                        "bin_idx": idx,
                        "hour": float(hour),
                        "power_kw": float(x[idx]),
                        "power_mw": float(x[idx] / 1000.0),
                    }
                )


def _build_fixture(root: Path) -> dict:
    parsed_csv = root / "data" / "azure_trace" / "parsed" / "day_2024-05-16_requests.csv"
    _write_parsed_requests_csv(parsed_csv)

    aggregated_root = root / "results" / "azure_facility" / "aggregated"
    ours_agg = aggregated_root / "ours"
    ours_agg.mkdir(parents=True, exist_ok=True)

    x = np.arange(96, dtype=np.float64)
    ours_kw = 700.0 + (60.0 * np.sin(2.0 * np.pi * x / 96.0))
    splitwise_kw = ours_kw + 35.0
    mean_kw = np.full_like(ours_kw, 640.0)
    tdp_kw = np.full_like(ours_kw, 1000.0)
    series_kw = {
        "ours": ours_kw,
        "splitwise_strict": splitwise_kw,
        "mean_baseline": mean_kw,
        "tdp_baseline": tdp_kw,
    }

    t = np.arange(86400, dtype=np.float64)
    for row_i in range(10):
        for rack_j in range(6):
            rack_id = row_i * 6 + rack_j
            rack_w = 7000.0 + (rack_id * 10.0) + (
                500.0 * np.sin(2.0 * np.pi * t / 86400.0 + rack_id * 0.03)
            )
            np.save(ours_agg / f"rack_{row_i}_{rack_j}.npy", np.asarray(rack_w, dtype=np.float32))

    metrics_rows = []
    for kind, arr in series_kw.items():
        peak = float(np.max(arr))
        avg = float(np.mean(arr))
        par = peak / avg
        ramp = float(np.max(np.diff(arr))) if arr.size > 1 else 0.0
        ramp_down = float(np.min(np.diff(arr))) if arr.size > 1 else 0.0
        metrics_rows.append(
            {
                "trace_kind": kind,
                "resolution_s": 900.0,
                "n_samples": 96,
                "peak_kw": peak,
                "avg_kw": avg,
                "par": par,
                "load_factor": avg / peak,
                "ramp_p50_kw_per_step": 0.0,
                "ramp_p95_abs_kw_per_step": abs(ramp),
                "ramp_p99_abs_kw_per_step": abs(ramp),
                "ramp_max_up_kw_per_step": ramp,
                "ramp_max_down_kw_per_step": ramp_down,
                "ramp_p95_abs_kw_per_s": abs(ramp) / 900.0,
                "ldc_p95_kw": float(np.percentile(arr, 95)),
                "ldc_p99_kw": float(np.percentile(arr, 99)),
                "diversity_factor_it": 0.8,
                "status": "evaluated",
                "notes": "constant_trace" if "baseline" in kind else "",
            }
        )

    metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
    ldc_csv = root / "results" / "eval_paper" / "azure_facility_ldc_15min.csv"
    site_csv = root / "results" / "eval_paper" / "azure_facility_site_traces_15min.csv"
    _write_metrics_csv(metrics_csv, metrics_rows)
    _write_ldc_csv(ldc_csv, series_kw)
    _write_site_traces_csv(site_csv, series_kw)

    return {
        "parsed_csv": parsed_csv,
        "aggregated_root": aggregated_root,
        "metrics_csv": metrics_csv,
        "ldc_csv": ldc_csv,
        "site_csv": site_csv,
        "out_dir": root / "figures",
    }


def test_arrival_binning_correctness() -> None:
    with tempfile.TemporaryDirectory() as td:
        parsed_csv = Path(td) / "parsed.csv"
        _write_parsed_requests_csv(parsed_csv)
        out = _load_arrival_rate_binned(str(parsed_csv), bin_seconds=300, day_seconds=86400)
        rates = np.asarray(out["rate_req_per_s"], dtype=np.float64)
        assert rates.shape == (288,)
        assert np.isclose(rates[0], 1.0 / 300.0)
        assert np.isclose(rates[1], 2.0 / 300.0)
        assert np.isclose(rates[2], 3.0 / 300.0)
        assert np.isclose(np.mean(rates), 2.0 / 300.0)


def test_baseline_annotation_and_ramp_conversion() -> None:
    metrics_rows = {
        "ours": {
            "peak_kw": 760.0,
            "avg_kw": 600.0,
            "par": 1.2666667,
            "ramp_max_up_kw_per_step": 40.0,
            "ramp_max_down_kw_per_step": -25.0,
            "ldc_p99_kw": 750.0,
        },
        "tdp_baseline": {
            "peak_kw": 1000.0,
            "avg_kw": 1000.0,
            "par": 1.0,
            "ramp_max_up_kw_per_step": 0.0,
            "ramp_max_down_kw_per_step": 0.0,
            "ldc_p99_kw": 1000.0,
        },
        "mean_baseline": {
            "peak_kw": 580.0,
            "avg_kw": 580.0,
            "par": 1.0,
            "ramp_max_up_kw_per_step": 0.0,
            "ramp_max_down_kw_per_step": 0.0,
            "ldc_p99_kw": 580.0,
        },
        "splitwise_strict": {
            "peak_kw": 790.0,
            "avg_kw": 640.0,
            "par": 1.234375,
            "ramp_max_up_kw_per_step": 28.0,
            "ramp_max_down_kw_per_step": -22.0,
            "ldc_p99_kw": 780.0,
        },
    }
    stats = _compute_baseline_comparison_stats(metrics_rows)
    assert np.isclose(stats["tdp_over_peak_pct"], (1000.0 - 760.0) / 760.0 * 100.0)
    assert np.isclose(stats["tdp_over_avg_pct"], (1000.0 - 600.0) / 600.0 * 100.0)

    sizing = _compute_sizing_metrics_from_rows(
        metrics_rows,
        ["tdp_baseline", "mean_baseline", "splitwise_strict", "ours"],
        power_resolution_seconds=900,
    )
    assert np.isclose(sizing["ours"]["max_ramp_mw_per_hr"], (40.0 / 1000.0) * 4.0)
    assert np.isclose(sizing["splitwise_strict"]["max_ramp_mw_per_hr"], (28.0 / 1000.0) * 4.0)
    assert TRACE_KIND_LABEL["splitwise_strict"] == "Splitwise"


def test_heatmap_matrix_shape() -> None:
    with tempfile.TemporaryDirectory() as td:
        aggregated_root = Path(td) / "aggregated" / "ours"
        aggregated_root.mkdir(parents=True, exist_ok=True)
        for row_i in range(10):
            for rack_j in range(6):
                arr = np.full((86400,), fill_value=7000.0 + row_i + rack_j, dtype=np.float32)
                np.save(aggregated_root / f"rack_{row_i}_{rack_j}.npy", arr)

        out = _load_rack_matrix_15min_kw(
            aggregated_root=str(Path(td) / "aggregated"),
            rows=10,
            racks_per_row=6,
            heatmap_downsample_seconds=900,
            method="ours",
        )
        mat = np.asarray(out["matrix_kw"], dtype=np.float64)
        assert mat.shape == (60, 96)


def test_generate_azure_figures_smoke_outputs_and_manifest() -> None:
    with tempfile.TemporaryDirectory() as td:
        fx = _build_fixture(Path(td))
        manifest = generate_azure_figures(
            parsed_requests_csv=str(fx["parsed_csv"]),
            aggregated_root=str(fx["aggregated_root"]),
            metrics_csv=str(fx["metrics_csv"]),
            ldc_csv=str(fx["ldc_csv"]),
            site_traces_15min_csv=str(fx["site_csv"]),
            out_dir=str(fx["out_dir"]),
            trace_kinds="tdp_baseline,mean_baseline,splitwise_strict,ours",
            arrival_bin_seconds=300,
            power_resolution_seconds=900,
            heatmap_downsample_seconds=900,
        )

        out_paths = manifest["output_paths"]
        for key in [
            "figure_1_diurnal_profile",
            "figure_2_baseline_comparison_15min",
            "figure_3_load_duration_curve",
            "figure_4_rack_heatmap",
            "figure_5_sizing_metrics",
            "manifest",
        ]:
            assert os.path.exists(out_paths[key])
            assert os.path.getsize(out_paths[key]) > 0

        assert manifest["schema_version"] == "azure-figures-v2"
        assert manifest["config"]["trace_kinds"] == [
            "tdp_baseline",
            "mean_baseline",
            "splitwise_strict",
            "ours",
        ]
        assert "splitwise_strict_over_peak_pct_vs_ours" in manifest["derived_metrics"]["splitwise_vs_ours"]


def test_missing_input_validation() -> None:
    with tempfile.TemporaryDirectory() as td:
        fx = _build_fixture(Path(td))
        os.remove(os.path.join(fx["aggregated_root"], "ours", "rack_0_0.npy"))
        try:
            generate_azure_figures(
                parsed_requests_csv=str(fx["parsed_csv"]),
                aggregated_root=str(fx["aggregated_root"]),
                metrics_csv=str(fx["metrics_csv"]),
                ldc_csv=str(fx["ldc_csv"]),
                site_traces_15min_csv=str(fx["site_csv"]),
                out_dir=str(fx["out_dir"]),
            )
            assert False, "Expected FileNotFoundError due to missing rack file"
        except FileNotFoundError:
            pass
