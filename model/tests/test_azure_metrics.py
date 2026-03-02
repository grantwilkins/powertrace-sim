"""
Tests for scripts/eval/azure_metrics.py.
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from azure_metrics import compute_azure_facility_metrics  # noqa: E402
from model.tests.test_eval_baselines_scripts import _build_toy_fixture  # noqa: E402


def _downsample_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    assert x.size % factor == 0
    return np.mean(x.reshape(-1, factor), axis=1)


def test_metrics_and_baselines_outputs():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _build_toy_fixture(root)

        aggregated_dir = root / "results" / "azure_facility" / "aggregated"
        node_trace_dir = root / "results" / "azure_facility" / "node_traces"
        aggregated_dir.mkdir(parents=True, exist_ok=True)
        node_trace_dir.mkdir(parents=True, exist_ok=True)

        pue = 1.3
        overhead = 1000.0
        t = 7200  # 30 minutes at 250ms.
        idx = np.arange(t, dtype=np.float64)
        node0_gpu = 100.0 + 10.0 * np.sin(2.0 * np.pi * idx / 200.0)
        node1_gpu = 120.0 + 5.0 * np.cos(2.0 * np.pi * idx / 150.0)

        np.save(node_trace_dir / "node_0_0_0.npy", np.asarray(node0_gpu, dtype=np.float32))
        np.save(node_trace_dir / "node_0_0_1.npy", np.asarray(node1_gpu, dtype=np.float32))

        site_it_250 = node0_gpu + node1_gpu + (2.0 * overhead)
        site_250 = site_it_250 * pue
        site_it_1s = _downsample_mean(site_it_250, 4)
        site_1s = site_it_1s * pue
        site_it_1min = _downsample_mean(site_it_1s, 60)
        site_1min = site_it_1min * pue
        site_it_15min = _downsample_mean(site_it_1s, 900)
        site_15min = site_it_15min * pue

        np.save(aggregated_dir / "site_it_250ms.npy", np.asarray(site_it_250, dtype=np.float32))
        np.save(aggregated_dir / "site_it_1s.npy", np.asarray(site_it_1s, dtype=np.float32))
        np.save(aggregated_dir / "site_it_1min.npy", np.asarray(site_it_1min, dtype=np.float32))
        np.save(aggregated_dir / "site_it_15min.npy", np.asarray(site_it_15min, dtype=np.float32))
        np.save(aggregated_dir / "site_250ms.npy", np.asarray(site_250, dtype=np.float32))
        np.save(aggregated_dir / "site_1s.npy", np.asarray(site_1s, dtype=np.float32))
        np.save(aggregated_dir / "site_1min.npy", np.asarray(site_1min, dtype=np.float32))
        np.save(aggregated_dir / "site_15min.npy", np.asarray(site_15min, dtype=np.float32))

        metrics_csv = root / "results" / "eval_paper" / "azure_facility_metrics.csv"
        ldc_csv = root / "results" / "eval_paper" / "azure_facility_ldc_15min.csv"
        summary = compute_azure_facility_metrics(
            aggregated_dir=str(aggregated_dir),
            node_trace_dir=str(node_trace_dir),
            experimental_manifest=str(fx["experimental_manifest"]),
            metrics_csv=str(metrics_csv),
            ldc_csv=str(ldc_csv),
            config_id=str(fx["config_id"]),
            rows=1,
            racks_per_row=1,
            nodes_per_rack=2,
            tp_gpus=4,
            gpu_tdp_w=700.0,
            non_gpu_overhead_w=overhead,
            pue=pue,
        )

        assert summary["status"] == "ok"
        assert metrics_csv.exists()
        assert ldc_csv.exists()

        with open(metrics_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 12  # 3 trace kinds x 4 resolutions
        assert {r["trace_kind"] for r in rows} == {"ours", "tdp_baseline", "mean_baseline"}
        assert {float(r["resolution_s"]) for r in rows} == {0.25, 1.0, 60.0, 900.0}

        ours_rows = [r for r in rows if r["trace_kind"] == "ours"]
        expected_div = float(np.max(site_it_250) / (2.0 * max(np.max(node0_gpu + overhead), np.max(node1_gpu + overhead))))
        for r in ours_rows:
            peak = float(r["peak_kw"])
            avg = float(r["avg_kw"])
            assert np.isfinite(float(r["par"]))
            assert np.isfinite(float(r["load_factor"]))
            assert np.isclose(float(r["load_factor"]), avg / peak)
            assert np.isclose(float(r["diversity_factor_it"]), expected_div)

        with open(ldc_csv, "r", newline="") as f:
            ldc_rows = list(csv.DictReader(f))
        # 30 minutes => two 15-min samples per trace kind.
        assert len(ldc_rows) == 6
        assert {r["trace_kind"] for r in ldc_rows} == {"ours", "tdp_baseline", "mean_baseline"}
