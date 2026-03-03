"""
Tests for scripts/eval/appendix_interconnection_b1.py.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from appendix_interconnection_b1 import generate_appendix_interconnection_b1  # noqa: E402


def _expand_bins_to_250ms_w(bin_values_mw: Iterable[float], factor: int = 3600) -> np.ndarray:
    vals = np.asarray(list(bin_values_mw), dtype=np.float64).reshape(-1)
    if vals.size <= 0:
        raise ValueError("bin_values_mw cannot be empty")
    return np.repeat(vals * 1_000_000.0, int(factor)).astype(np.float64)


def _write_seed_export(
    *,
    seed_dir: Path,
    config_id: str,
    dt: float,
    duration_s: float,
    n_nodes: int,
    base_seed: int,
    method_series_w: Dict[str, np.ndarray],
) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    for method, arr in method_series_w.items():
        out_path = seed_dir / f"facility_power_250ms_{method}_w.npy"
        np.save(out_path, np.asarray(arr, dtype=np.float64))
        files[method] = str(out_path)

    manifest = {
        "schema_version": "facility-trace-export-v1",
        "config_id": config_id,
        "base_seed": int(base_seed),
        "dt": float(dt),
        "duration_s": float(duration_s),
        "n_nodes": int(n_nodes),
        "lambda_req_per_s_per_node": 0.25,
        "facility_power_mode": "gpu_sum_only",
        "pue": 1.0,
        "non_gpu_overhead_w": 0.0,
        "traffic": {
            "model": "poisson",
            "burst_rate_per_min": 2.0,
            "burst_mean_duration_s": 20.0,
            "burst_peak_scale": 6.0,
            "burst_background_sigma": 0.35,
            "burst_node_scale_sigma": 0.2,
        },
        "files": files,
        "missing_methods": [],
    }
    with open(seed_dir / "facility_trace_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def test_appendix_b1_smoke_outputs():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        seed_root = root / "seed_exports"
        seed_ids = [42, 43, 44, 45, 46]
        dt = 0.25
        n_bins = 4
        duration_s = float(n_bins * 900)
        factor = int(900.0 / dt)

        for i, seed in enumerate(seed_ids):
            ours_bins = np.asarray([0.8, 1.0, 0.95, 1.1], dtype=np.float64) + (0.01 * i)
            split_bins = np.asarray([0.8, 0.9, 0.88, 0.93], dtype=np.float64) + (0.005 * i)
            method_series = {
                "ours": _expand_bins_to_250ms_w(ours_bins, factor=factor),
                "splitwise_lut": _expand_bins_to_250ms_w(split_bins, factor=factor),
                "tdp": _expand_bins_to_250ms_w([1.2, 1.2, 1.2, 1.2], factor=factor),
                "mean": _expand_bins_to_250ms_w([0.9, 0.9, 0.9, 0.9], factor=factor),
            }
            _write_seed_export(
                seed_dir=seed_root / f"seed_{seed}",
                config_id="deepseek-r1-distill-70b_H100_tp4",
                dt=dt,
                duration_s=duration_s,
                n_nodes=240,
                base_seed=seed,
                method_series_w=method_series,
            )

        out_pdf = root / "figures" / "appendix_b1_ramp_rate_cdf_15min.pdf"
        out_cdf = root / "results" / "eval_paper" / "appendix_b1_ramp_cdf_points.csv"
        out_summary = root / "results" / "eval_paper" / "appendix_b1_ramp_summary.csv"
        out_manifest = root / "results" / "eval_paper" / "appendix_b1_manifest.json"
        run = generate_appendix_interconnection_b1(
            seed_trace_root=str(seed_root),
            seed_glob="seed_*",
            expected_seeds=5,
            out_figure_pdf=str(out_pdf),
            out_cdf_csv=str(out_cdf),
            out_summary_csv=str(out_summary),
            out_manifest_json=str(out_manifest),
            dry_run=False,
        )

        assert out_pdf.exists()
        assert out_cdf.exists()
        assert out_summary.exists()
        assert out_manifest.exists()
        assert run["schema_version"] == "appendix-b1-ramp-cdf-v1"

        with open(out_summary, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        methods = {row["method"] for row in rows}
        assert "ours" in methods
        assert "splitwise_lut" in methods
        assert "tdp" in methods
        assert "mean" in methods
        tdp_row = next(r for r in rows if r["method"] == "tdp")
        mean_row = next(r for r in rows if r["method"] == "mean")
        assert np.isclose(float(tdp_row["max_ramp_mw"]), 0.0)
        assert np.isclose(float(mean_row["max_ramp_mw"]), 0.0)


def test_appendix_b1_numeric_p95():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        seed_root = root / "seed_exports"
        dt = 0.25
        factor = int(900.0 / dt)
        ours_bins = [0.0, 10.0, 20.0, 30.0]
        split_bins = [0.0, 5.0, 15.0, 30.0]
        method_series = {
            "ours": _expand_bins_to_250ms_w(ours_bins, factor=factor),
            "splitwise_lut": _expand_bins_to_250ms_w(split_bins, factor=factor),
            "tdp": _expand_bins_to_250ms_w([2.0, 2.0, 2.0, 2.0], factor=factor),
            "mean": _expand_bins_to_250ms_w([1.0, 1.0, 1.0, 1.0], factor=factor),
        }
        _write_seed_export(
            seed_dir=seed_root / "seed_42",
            config_id="deepseek-r1-distill-70b_H100_tp4",
            dt=dt,
            duration_s=3600.0,
            n_nodes=240,
            base_seed=42,
            method_series_w=method_series,
        )

        out_pdf = root / "figures" / "appendix_b1_ramp_rate_cdf_15min.pdf"
        out_cdf = root / "results" / "eval_paper" / "appendix_b1_ramp_cdf_points.csv"
        out_summary = root / "results" / "eval_paper" / "appendix_b1_ramp_summary.csv"
        out_manifest = root / "results" / "eval_paper" / "appendix_b1_manifest.json"
        generate_appendix_interconnection_b1(
            seed_trace_root=str(seed_root),
            seed_glob="seed_*",
            expected_seeds=1,
            out_figure_pdf=str(out_pdf),
            out_cdf_csv=str(out_cdf),
            out_summary_csv=str(out_summary),
            out_manifest_json=str(out_manifest),
            dry_run=False,
        )

        with open(out_summary, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        ours_row = next(r for r in rows if r["method"] == "ours")
        split_row = next(r for r in rows if r["method"] == "splitwise_lut")
        assert np.isclose(float(ours_row["p95_ramp_mw"]), 10.0)
        assert np.isclose(float(split_row["p95_ramp_mw"]), 14.5)


def test_appendix_b1_missing_method_file_validation():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        seed_root = root / "seed_exports"
        dt = 0.25
        factor = int(900.0 / dt)
        method_series = {
            "ours": _expand_bins_to_250ms_w([1.0, 1.1, 1.2, 1.3], factor=factor),
        }
        _write_seed_export(
            seed_dir=seed_root / "seed_42",
            config_id="deepseek-r1-distill-70b_H100_tp4",
            dt=dt,
            duration_s=3600.0,
            n_nodes=240,
            base_seed=42,
            method_series_w=method_series,
        )

        out_manifest = root / "results" / "eval_paper" / "appendix_b1_manifest.json"
        try:
            generate_appendix_interconnection_b1(
                seed_trace_root=str(seed_root),
                seed_glob="seed_*",
                expected_seeds=1,
                out_figure_pdf=str(root / "figures" / "x.pdf"),
                out_cdf_csv=str(root / "results" / "eval_paper" / "x.csv"),
                out_summary_csv=str(root / "results" / "eval_paper" / "y.csv"),
                out_manifest_json=str(out_manifest),
                dry_run=False,
            )
            assert False, "Expected FileNotFoundError for missing splitwise_lut trace"
        except FileNotFoundError:
            pass


def test_appendix_b1_mismatched_dt_validation():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        seed_root = root / "seed_exports"
        factor_025 = int(900.0 / 0.25)
        factor_05 = int(900.0 / 0.5)
        method_series_025 = {
            "ours": _expand_bins_to_250ms_w([1.0, 1.1, 1.2, 1.3], factor=factor_025),
            "splitwise_lut": _expand_bins_to_250ms_w(
                [1.0, 1.05, 1.1, 1.15], factor=factor_025
            ),
        }
        method_series_05 = {
            "ours": _expand_bins_to_250ms_w([1.0, 1.1, 1.2, 1.3], factor=factor_05),
            "splitwise_lut": _expand_bins_to_250ms_w(
                [1.0, 1.05, 1.1, 1.15], factor=factor_05
            ),
        }
        _write_seed_export(
            seed_dir=seed_root / "seed_42",
            config_id="deepseek-r1-distill-70b_H100_tp4",
            dt=0.25,
            duration_s=3600.0,
            n_nodes=240,
            base_seed=42,
            method_series_w=method_series_025,
        )
        _write_seed_export(
            seed_dir=seed_root / "seed_43",
            config_id="deepseek-r1-distill-70b_H100_tp4",
            dt=0.5,
            duration_s=3600.0,
            n_nodes=240,
            base_seed=43,
            method_series_w=method_series_05,
        )

        try:
            generate_appendix_interconnection_b1(
                seed_trace_root=str(seed_root),
                seed_glob="seed_*",
                expected_seeds=2,
                out_figure_pdf=str(root / "figures" / "x.pdf"),
                out_cdf_csv=str(root / "results" / "eval_paper" / "x.csv"),
                out_summary_csv=str(root / "results" / "eval_paper" / "y.csv"),
                out_manifest_json=str(root / "results" / "eval_paper" / "z.json"),
                dry_run=False,
            )
            assert False, "Expected ValueError for inconsistent dt across seeds"
        except ValueError as exc:
            assert "Inconsistent dt" in str(exc)
