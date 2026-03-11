"""Tests for isolated Azure baselines-included figures."""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np

from scripts.eval.azure_scripts_baselines_included.azure_figures import generate_azure_figures


def _write_parsed_requests_csv(path: Path, bin_seconds: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["request_id", "timestamp_utc", "arrival_time", "n_in", "n_out"],
        )
        writer.writeheader()
        req_id = 0
        for b in range(288):
            n_req = (b % 3) + 1
            t = (b * bin_seconds) + 1.0
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
            for i in range(x.size):
                hour = (i + 0.5) * (900.0 / 3600.0)
                writer.writerow(
                    {
                        "trace_kind": trace_kind,
                        "bin_idx": i,
                        "hour": float(hour),
                        "power_kw": float(x[i]),
                        "power_mw": float(x[i] / 1000.0),
                    }
                )


def test_generate_azure_bi_figures_smoke() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        parsed_csv = root / "data" / "azure_trace" / "parsed" / "day_2024-05-16_requests.csv"
        _write_parsed_requests_csv(parsed_csv)

        aggregated_root = root / "results" / "azure_facility_baselines_included" / "aggregated"
        ours_agg = aggregated_root / "ours"
        ours_agg.mkdir(parents=True, exist_ok=True)

        # 96 bins = 24h @ 15-min
        x = np.arange(96, dtype=np.float64)
        ours_kw = 700.0 + (60.0 * np.sin(2.0 * np.pi * x / 96.0))
        swt_kw = ours_kw + 20.0
        sws_kw = ours_kw + 35.0
        mean_kw = np.full_like(ours_kw, 640.0)
        tdp_kw = np.full_like(ours_kw, 1000.0)
        series_kw = {
            "ours": ours_kw,
            "splitwise_lut": swt_kw,
            "splitwise_strict": sws_kw,
            "mean_baseline": mean_kw,
            "tdp_baseline": tdp_kw,
        }

        # Rack traces at 1s (for heatmap), only needed for ours.
        t = np.arange(86400, dtype=np.float64)
        for row_i in range(10):
            for rack_j in range(6):
                rack_id = row_i * 6 + rack_j
                rack_w = 7000.0 + (rack_id * 10.0) + (500.0 * np.sin(2.0 * np.pi * t / 86400.0 + rack_id * 0.03))
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

        metrics_csv = root / "results" / "eval_paper_baselines_included" / "azure_facility_metrics.csv"
        ldc_csv = root / "results" / "eval_paper_baselines_included" / "azure_facility_ldc_15min.csv"
        site_csv = root / "results" / "eval_paper_baselines_included" / "azure_facility_site_traces_15min.csv"
        _write_metrics_csv(metrics_csv, metrics_rows)
        _write_ldc_csv(ldc_csv, series_kw)
        _write_site_traces_csv(site_csv, series_kw)

        out_dir = root / "figures" / "azure_baselines_included"
        manifest = generate_azure_figures(
            parsed_requests_csv=str(parsed_csv),
            aggregated_root=str(aggregated_root),
            metrics_csv=str(metrics_csv),
            ldc_csv=str(ldc_csv),
            site_traces_15min_csv=str(site_csv),
            out_dir=str(out_dir),
            trace_kinds="tdp_baseline,mean_baseline,splitwise_lut,splitwise_strict,ours",
            arrival_bin_seconds=300,
            power_resolution_seconds=900,
            heatmap_downsample_seconds=900,
            rows=10,
            racks_per_row=6,
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

        assert manifest["schema_version"] == "azure-figures-baselines-included-v1"
        assert "splitwise_vs_ours" in manifest["derived_metrics"]
