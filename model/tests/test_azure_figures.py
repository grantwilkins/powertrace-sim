"""
Tests for scripts/eval/azure_figures.py.
"""

import csv
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from azure_figures import (  # noqa: E402
    _compute_ldc_band_from_seed_site_mw,
    _compute_baseline_comparison_stats,
    _compute_sizing_metrics_from_rows,
    _load_arrival_rate_binned,
    _load_rack_matrix_15min_kw,
    generate_azure_figures,
)


def _write_parsed_requests_csv(path: str, bin_seconds: int = 300) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["request_id", "timestamp_utc", "arrival_time", "n_in", "n_out"],
        )
        writer.writeheader()
        req_id = 0
        for b in range(288):
            n_req = (b % 3) + 1  # 1,2,3 repeating.
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


def _write_metrics_csv(path: str) -> None:
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
        writer.writerow(
            {
                "trace_kind": "ours",
                "resolution_s": 900.0,
                "n_samples": 96,
                "peak_kw": 760.0,
                "avg_kw": 600.0,
                "par": 1.2666667,
                "load_factor": 0.7894737,
                "ramp_p50_kw_per_step": 0.0,
                "ramp_p95_abs_kw_per_step": 10.0,
                "ramp_p99_abs_kw_per_step": 20.0,
                "ramp_max_up_kw_per_step": 40.0,
                "ramp_max_down_kw_per_step": -25.0,
                "ramp_p95_abs_kw_per_s": 0.011111,
                "ldc_p95_kw": 720.0,
                "ldc_p99_kw": 750.0,
                "diversity_factor_it": 0.8,
                "status": "evaluated",
                "notes": "",
            }
        )
        writer.writerow(
            {
                "trace_kind": "tdp_baseline",
                "resolution_s": 900.0,
                "n_samples": 96,
                "peak_kw": 1000.0,
                "avg_kw": 1000.0,
                "par": 1.0,
                "load_factor": 1.0,
                "ramp_p50_kw_per_step": 0.0,
                "ramp_p95_abs_kw_per_step": 0.0,
                "ramp_p99_abs_kw_per_step": 0.0,
                "ramp_max_up_kw_per_step": 0.0,
                "ramp_max_down_kw_per_step": 0.0,
                "ramp_p95_abs_kw_per_s": 0.0,
                "ldc_p95_kw": 1000.0,
                "ldc_p99_kw": 1000.0,
                "diversity_factor_it": 1.0,
                "status": "evaluated",
                "notes": "constant_trace",
            }
        )
        writer.writerow(
            {
                "trace_kind": "mean_baseline",
                "resolution_s": 900.0,
                "n_samples": 96,
                "peak_kw": 580.0,
                "avg_kw": 580.0,
                "par": 1.0,
                "load_factor": 1.0,
                "ramp_p50_kw_per_step": 0.0,
                "ramp_p95_abs_kw_per_step": 0.0,
                "ramp_p99_abs_kw_per_step": 0.0,
                "ramp_max_up_kw_per_step": 0.0,
                "ramp_max_down_kw_per_step": 0.0,
                "ramp_p95_abs_kw_per_s": 0.0,
                "ldc_p95_kw": 580.0,
                "ldc_p99_kw": 580.0,
                "diversity_factor_it": 1.0,
                "status": "evaluated",
                "notes": "constant_trace",
            }
        )


def _write_ldc_csv(path: str, ours_kw: np.ndarray) -> None:
    ours_sorted = np.sort(np.asarray(ours_kw, dtype=np.float64))[::-1]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trace_kind", "resolution_s", "rank", "fraction_exceeded", "power_kw"],
        )
        writer.writeheader()
        n = int(ours_sorted.size)
        for rank in range(n):
            frac = rank / float(max(1, n))
            writer.writerow(
                {
                    "trace_kind": "ours",
                    "resolution_s": 900.0,
                    "rank": rank,
                    "fraction_exceeded": frac,
                    "power_kw": float(ours_sorted[rank]),
                }
            )
            writer.writerow(
                {
                    "trace_kind": "tdp_baseline",
                    "resolution_s": 900.0,
                    "rank": rank,
                    "fraction_exceeded": frac,
                    "power_kw": 1000.0,
                }
            )
            writer.writerow(
                {
                    "trace_kind": "mean_baseline",
                    "resolution_s": 900.0,
                    "rank": rank,
                    "fraction_exceeded": frac,
                    "power_kw": 580.0,
                }
            )


def _build_fixture(root: str) -> dict:
    parsed_csv = os.path.join(root, "data", "azure_trace", "parsed", "day_2024-05-16_requests.csv")
    os.makedirs(os.path.dirname(parsed_csv), exist_ok=True)
    _write_parsed_requests_csv(parsed_csv)

    aggregated_dir = os.path.join(root, "results", "azure_facility", "aggregated")
    os.makedirs(aggregated_dir, exist_ok=True)

    # 96 bins = 24 hours @ 15-min.
    x = np.arange(96, dtype=np.float64)
    site_15min_mw = 0.6 + (0.08 * np.sin(2.0 * np.pi * x / 96.0))
    np.save(os.path.join(aggregated_dir, "site_15min.npy"), np.asarray(site_15min_mw * 1e6, dtype=np.float32))

    # 60 racks, 86400 samples (1s for 24h), modest variation per rack.
    t = np.arange(86400, dtype=np.float64)
    for row_i in range(10):
        for rack_j in range(6):
            rack_id = row_i * 6 + rack_j
            rack_w = 7000.0 + (rack_id * 10.0) + (500.0 * np.sin(2.0 * np.pi * t / 86400.0 + rack_id * 0.03))
            np.save(
                os.path.join(aggregated_dir, f"rack_{row_i}_{rack_j}.npy"),
                np.asarray(rack_w, dtype=np.float32),
            )

    eval_dir = os.path.join(root, "results", "eval_paper")
    os.makedirs(eval_dir, exist_ok=True)
    metrics_csv = os.path.join(eval_dir, "azure_facility_metrics.csv")
    _write_metrics_csv(metrics_csv)
    ldc_csv = os.path.join(eval_dir, "azure_facility_ldc_15min.csv")
    _write_ldc_csv(ldc_csv, ours_kw=site_15min_mw * 1000.0)

    out_dir = os.path.join(root, "figures")
    return {
        "parsed_csv": parsed_csv,
        "aggregated_dir": aggregated_dir,
        "metrics_csv": metrics_csv,
        "ldc_csv": ldc_csv,
        "out_dir": out_dir,
    }


def _write_b2_seed_cache(
    seed_cache_dir: str,
    seeds: list[int],
    *,
    n_points: int = 96,
) -> None:
    x = np.arange(n_points, dtype=np.float64)
    base_mw = 0.58 + (0.06 * np.sin(2.0 * np.pi * x / float(max(1, n_points))))
    for idx, seed in enumerate(seeds):
        seed_dir = os.path.join(seed_cache_dir, f"seed_{int(seed)}", "aggregated")
        os.makedirs(seed_dir, exist_ok=True)
        site_mw = base_mw + (0.0025 * float(idx))
        site_w = np.asarray(site_mw * 1e6, dtype=np.float32)
        np.save(os.path.join(seed_dir, "site_15min.npy"), site_w)


def test_arrival_binning_correctness():
    with tempfile.TemporaryDirectory() as td:
        parsed_csv = os.path.join(td, "parsed.csv")
        _write_parsed_requests_csv(parsed_csv)
        out = _load_arrival_rate_binned(parsed_csv, bin_seconds=300, day_seconds=86400)
        rates = np.asarray(out["rate_req_per_s"], dtype=np.float64)
        assert rates.shape == (288,)
        assert np.isclose(rates[0], 1.0 / 300.0)
        assert np.isclose(rates[1], 2.0 / 300.0)
        assert np.isclose(rates[2], 3.0 / 300.0)
        assert np.isclose(np.mean(rates), 2.0 / 300.0)


def test_baseline_annotation_and_ramp_conversion():
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
    }
    stats = _compute_baseline_comparison_stats(metrics_rows)
    assert np.isclose(stats["tdp_over_peak_pct"], (1000.0 - 760.0) / 760.0 * 100.0)
    assert np.isclose(stats["tdp_over_avg_pct"], (1000.0 - 600.0) / 600.0 * 100.0)

    sizing = _compute_sizing_metrics_from_rows(metrics_rows, power_resolution_seconds=900)
    assert np.isclose(sizing["ours"]["max_ramp_mw_per_hr"], (40.0 / 1000.0) * 4.0)
    assert np.isclose(sizing["tdp_baseline"]["max_ramp_mw_per_hr"], 0.0)
    assert np.isclose(sizing["mean_baseline"]["max_ramp_mw_per_hr"], 0.0)


def test_heatmap_matrix_shape():
    with tempfile.TemporaryDirectory() as td:
        aggregated_dir = os.path.join(td, "aggregated")
        os.makedirs(aggregated_dir, exist_ok=True)
        for row_i in range(10):
            for rack_j in range(6):
                arr = np.full((86400,), fill_value=7000.0 + row_i + rack_j, dtype=np.float32)
                np.save(os.path.join(aggregated_dir, f"rack_{row_i}_{rack_j}.npy"), arr)

        out = _load_rack_matrix_15min_kw(
            aggregated_dir=aggregated_dir,
            rows=10,
            racks_per_row=6,
            heatmap_downsample_seconds=900,
        )
        mat = np.asarray(out["matrix_kw"], dtype=np.float64)
        assert mat.shape == (60, 96)


def test_generate_azure_figures_smoke_outputs_and_manifest():
    with tempfile.TemporaryDirectory() as td:
        fx = _build_fixture(td)
        manifest = generate_azure_figures(
            parsed_requests_csv=fx["parsed_csv"],
            aggregated_dir=fx["aggregated_dir"],
            metrics_csv=fx["metrics_csv"],
            ldc_csv=fx["ldc_csv"],
            out_dir=fx["out_dir"],
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

        assert manifest["schema_version"] == "azure-figures-v1"
        assert "derived_metrics" in manifest
        assert "figures" in manifest
        assert np.isfinite(float(manifest["derived_metrics"]["tdp_over_peak_pct"]))
        assert np.isfinite(float(manifest["derived_metrics"]["tdp_over_avg_pct"]))


def test_ldc_band_math_from_seed_site_series():
    values = np.asarray(
        [
            [1.00, 0.90, 0.80, 0.70],
            [1.10, 0.85, 0.75, 0.65],
            [0.95, 0.90, 0.85, 0.60],
        ],
        dtype=np.float64,
    )
    out = _compute_ldc_band_from_seed_site_mw(values)
    assert np.allclose(out["fraction_exceeded"], np.asarray([0.0, 0.25, 0.5, 0.75]))
    assert np.allclose(out["ldc_median_mw"], np.asarray([1.00, 0.90, 0.80, 0.65]))
    assert np.allclose(out["ldc_min_mw"], np.asarray([0.95, 0.85, 0.75, 0.60]))
    assert np.allclose(out["ldc_max_mw"], np.asarray([1.10, 0.90, 0.85, 0.70]))
    assert np.isclose(float(out["mean_mw"]), 0.8375)
    assert np.isclose(float(out["peak_mw"]), 1.0)
    assert np.isclose(float(out["load_factor"]), 0.8375)
    assert int(out["n_seeds"]) == 3
    assert int(out["n_points"]) == 4


def test_generate_azure_figures_b2_with_cached_seed_runs():
    with tempfile.TemporaryDirectory() as td:
        fx = _build_fixture(td)
        seed_cache_dir = os.path.join(
            td, "results", "azure_facility", "seed_runs"
        )
        seeds = [42, 43, 44, 45, 46]
        _write_b2_seed_cache(seed_cache_dir, seeds)

        manifest = generate_azure_figures(
            parsed_requests_csv=fx["parsed_csv"],
            aggregated_dir=fx["aggregated_dir"],
            metrics_csv=fx["metrics_csv"],
            ldc_csv=fx["ldc_csv"],
            out_dir=fx["out_dir"],
            arrival_bin_seconds=300,
            power_resolution_seconds=900,
            heatmap_downsample_seconds=900,
            include_figure_b2=True,
            b2_seeds=seeds,
            b2_seed_cache_dir=seed_cache_dir,
        )

        assert "figure_b2_load_duration_confidence" in manifest["figures"]
        assert "figure_b2_load_duration_confidence" in manifest["output_paths"]
        b2_path = manifest["output_paths"]["figure_b2_load_duration_confidence"]
        assert os.path.exists(b2_path)
        assert os.path.getsize(b2_path) > 0

        b2_meta = manifest["figures"]["figure_b2_load_duration_confidence"]
        assert b2_meta["band_type"] == "min_max"
        assert b2_meta["seeds"] == seeds
        assert int(b2_meta["n_points"]) == 96
        assert int(b2_meta["stats"]["n_seeds"]) == 5
        assert np.isfinite(float(b2_meta["stats"]["load_factor"]))


def test_generate_azure_figures_b2_bad_cached_shape_raises():
    with tempfile.TemporaryDirectory() as td:
        fx = _build_fixture(td)
        seed_cache_dir = os.path.join(
            td, "results", "azure_facility", "seed_runs"
        )
        seeds = [42, 43, 44, 45, 46]
        _write_b2_seed_cache(seed_cache_dir, seeds, n_points=95)
        try:
            generate_azure_figures(
                parsed_requests_csv=fx["parsed_csv"],
                aggregated_dir=fx["aggregated_dir"],
                metrics_csv=fx["metrics_csv"],
                ldc_csv=fx["ldc_csv"],
                out_dir=fx["out_dir"],
                include_figure_b2=True,
                b2_seeds=seeds,
                b2_seed_cache_dir=seed_cache_dir,
            )
            assert False, "Expected ValueError for invalid seed cache length"
        except ValueError as exc:
            msg = str(exc)
            assert "Expected 96 points" in msg
            assert "seed=42" in msg


def test_generate_azure_figures_b2_missing_cache_generation_failure_actionable():
    with tempfile.TemporaryDirectory() as td:
        fx = _build_fixture(td)
        seed_cache_dir = os.path.join(td, "results", "azure_facility", "seed_runs")
        os.makedirs(seed_cache_dir, exist_ok=True)
        with patch("azure_figures._generate_b2_seed_cache", side_effect=RuntimeError("missing inputs")):
            try:
                generate_azure_figures(
                    parsed_requests_csv=fx["parsed_csv"],
                    aggregated_dir=fx["aggregated_dir"],
                    metrics_csv=fx["metrics_csv"],
                    ldc_csv=fx["ldc_csv"],
                    out_dir=fx["out_dir"],
                    include_figure_b2=True,
                    b2_seeds=[42],
                    b2_seed_cache_dir=seed_cache_dir,
                )
                assert False, "Expected RuntimeError for B2 cache generation failure"
            except RuntimeError as exc:
                msg = str(exc)
                assert "Failed to generate seed cache for seed=42" in msg
                assert "pre-populate" in msg
                assert "Underlying error" in msg


def test_missing_input_validation():
    with tempfile.TemporaryDirectory() as td:
        fx = _build_fixture(td)
        os.remove(os.path.join(fx["aggregated_dir"], "rack_0_0.npy"))
        try:
            generate_azure_figures(
                parsed_requests_csv=fx["parsed_csv"],
                aggregated_dir=fx["aggregated_dir"],
                metrics_csv=fx["metrics_csv"],
                ldc_csv=fx["ldc_csv"],
                out_dir=fx["out_dir"],
            )
            assert False, "Expected FileNotFoundError due to missing rack file"
        except FileNotFoundError:
            pass
