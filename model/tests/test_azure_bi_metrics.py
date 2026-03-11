"""Tests for isolated Azure baselines-included metrics."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from scripts.eval.azure_scripts_baselines_included.azure_metrics import (
    compute_azure_facility_metrics,
)


def _downsample_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    assert x.size % factor == 0
    return np.mean(x.reshape(-1, factor), axis=1)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _build_minimal_experimental_fixture(root: Path) -> dict:
    cfg = "toy-70b_H100_tp4"
    dataset_path = root / "results" / "experimental_continuous_v1" / "datasets" / "toy_H100_tp4.npz"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    power_train = np.array([1090.0, 1110.0, 1130.0, 1150.0], dtype=np.float64)
    power_test = np.array([1100.0, 1120.0, 1140.0, 1160.0], dtype=np.float64)
    np.savez(
        dataset_path,
        config_id=np.asarray([cfg, cfg], dtype=object),
        dt=np.asarray([0.25], dtype=np.float64),
        pair_key=np.asarray(["pk_train", "pk_test"], dtype=object),
        rate=np.asarray(["1", "1"], dtype=object),
        power_start_epoch_s=np.asarray([1000.0, 1000.0], dtype=np.float64),
        power=np.asarray([power_train, power_test], dtype=object),
    )

    split_path = root / "results" / "experimental_continuous_v1" / "splits" / "toy_H100_tp4.json"
    _write_json(split_path, {"train_indices": [0], "val_indices": [], "test_indices": [1]})

    manifest_path = root / "results" / "experimental_continuous_v1" / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "experimental-continuous-v1",
            "configs": {
                cfg: {
                    "dataset_npz": str(dataset_path),
                    "split_json": str(split_path),
                    "written": True,
                }
            },
        },
    )
    return {"config_id": cfg, "experimental_manifest": manifest_path}


def _write_method_aggregates(
    *,
    method_dir: Path,
    node0_gpu: np.ndarray,
    node1_gpu: np.ndarray,
    overhead: float,
    pue: float,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    site_it_250 = node0_gpu + node1_gpu + (2.0 * overhead)
    site_250 = site_it_250 * pue
    site_it_1s = _downsample_mean(site_it_250, 4)
    site_1s = site_it_1s * pue
    site_it_1min = _downsample_mean(site_it_1s, 60)
    site_1min = site_it_1min * pue
    site_it_15min = _downsample_mean(site_it_1s, 900)
    site_15min = site_it_15min * pue

    np.save(method_dir / "site_it_250ms.npy", np.asarray(site_it_250, dtype=np.float32))
    np.save(method_dir / "site_it_1s.npy", np.asarray(site_it_1s, dtype=np.float32))
    np.save(method_dir / "site_it_1min.npy", np.asarray(site_it_1min, dtype=np.float32))
    np.save(method_dir / "site_it_15min.npy", np.asarray(site_it_15min, dtype=np.float32))

    np.save(method_dir / "site_250ms.npy", np.asarray(site_250, dtype=np.float32))
    np.save(method_dir / "site_1s.npy", np.asarray(site_1s, dtype=np.float32))
    np.save(method_dir / "site_1min.npy", np.asarray(site_1min, dtype=np.float32))
    np.save(method_dir / "site_15min.npy", np.asarray(site_15min, dtype=np.float32))


def test_metrics_outputs_include_splitwise() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _build_minimal_experimental_fixture(root)

        aggregated_root = root / "results" / "azure_facility_baselines_included" / "aggregated"
        node_traces_root = root / "results" / "azure_facility_baselines_included" / "node_traces"

        pue = 1.3
        overhead = 1000.0
        t = 7200  # 30 minutes at 250ms
        idx = np.arange(t, dtype=np.float64)
        ours0 = 100.0 + 10.0 * np.sin(2.0 * np.pi * idx / 200.0)
        ours1 = 120.0 + 5.0 * np.cos(2.0 * np.pi * idx / 150.0)
        swt0 = ours0 + 8.0
        swt1 = ours1 + 8.0
        sws0 = ours0 + 15.0
        sws1 = ours1 + 15.0

        for method, n0, n1 in [
            ("ours", ours0, ours1),
            ("splitwise_lut", swt0, swt1),
            ("splitwise_strict", sws0, sws1),
        ]:
            method_node_dir = node_traces_root / method
            method_node_dir.mkdir(parents=True, exist_ok=True)
            np.save(method_node_dir / "node_0_0_0.npy", np.asarray(n0, dtype=np.float32))
            np.save(method_node_dir / "node_0_0_1.npy", np.asarray(n1, dtype=np.float32))
            _write_method_aggregates(
                method_dir=aggregated_root / method,
                node0_gpu=n0,
                node1_gpu=n1,
                overhead=overhead,
                pue=pue,
            )

        metrics_csv = root / "results" / "eval_paper_baselines_included" / "azure_facility_metrics.csv"
        ldc_csv = root / "results" / "eval_paper_baselines_included" / "azure_facility_ldc_15min.csv"
        site_csv = root / "results" / "eval_paper_baselines_included" / "azure_facility_site_traces_15min.csv"

        summary = compute_azure_facility_metrics(
            aggregated_root=str(aggregated_root),
            node_traces_root=str(node_traces_root),
            experimental_manifest=str(fx["experimental_manifest"]),
            metrics_csv=str(metrics_csv),
            ldc_csv=str(ldc_csv),
            site_traces_15min_csv=str(site_csv),
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
        assert site_csv.exists()

        with open(metrics_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 20  # 5 trace kinds x 4 resolutions
        assert {r["trace_kind"] for r in rows} == {
            "ours",
            "splitwise_lut",
            "splitwise_strict",
            "tdp_baseline",
            "mean_baseline",
        }

        with open(ldc_csv, "r", newline="") as f:
            ldc_rows = list(csv.DictReader(f))
        assert len(ldc_rows) == 10  # 2 samples x 5 kinds

        with open(site_csv, "r", newline="") as f:
            site_rows = list(csv.DictReader(f))
        assert len(site_rows) == 10
