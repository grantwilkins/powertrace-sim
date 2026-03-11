"""Tests for isolated Azure baselines-included aggregation."""

import tempfile
from pathlib import Path

import numpy as np

from scripts.eval.azure_scripts_baselines_included.azure_aggregate import aggregate_all_methods


def test_aggregate_all_methods_outputs() -> None:
    t = 3600  # 15 minutes at 250ms
    dt = 0.25
    overhead_w = 1000.0
    pue = 1.3

    methods = ["ours", "splitwise_lut", "splitwise_strict"]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        node_root = root / "results" / "azure_facility_baselines_included" / "node_traces"
        out_root = root / "results" / "azure_facility_baselines_included" / "aggregated"

        rows, racks_per_row, nodes_per_rack = 2, 2, 2
        for method_idx, method in enumerate(methods):
            method_dir = node_root / method
            method_dir.mkdir(parents=True, exist_ok=True)
            base_gpu = 100.0 + (10.0 * method_idx)
            for i in range(rows):
                for j in range(racks_per_row):
                    for k in range(nodes_per_rack):
                        path = method_dir / f"node_{i}_{j}_{k}.npy"
                        np.save(path, np.full((t,), base_gpu + i + j + k, dtype=np.float32))

        summary = aggregate_all_methods(
            node_traces_root=str(node_root),
            aggregated_root=str(out_root),
            methods=",".join(methods),
            dt=dt,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
            non_gpu_overhead_w=overhead_w,
            pue=pue,
        )

        assert summary["status"] == "ok"
        assert summary["methods"] == methods

        for method in methods:
            method_out = out_root / method
            assert (method_out / "site_250ms.npy").exists()
            assert (method_out / "site_1s.npy").exists()
            assert (method_out / "site_1min.npy").exists()
            assert (method_out / "site_15min.npy").exists()
            assert (method_out / "aggregation_metadata.json").exists()

        ours_site = np.asarray(np.load(out_root / "ours" / "site_250ms.npy"), dtype=np.float64).reshape(-1)
        # 8 nodes, each 100W base gpu + mean(i+j+k)=1.5W + overhead, then PUE
        expected = (8.0 * ((100.0 + 1.5) + overhead_w)) * pue
        assert np.allclose(np.mean(ours_site), expected, atol=5.0)
