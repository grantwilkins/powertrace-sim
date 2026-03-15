"""
Tests for scripts/eval/azure_aggregate.py.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from azure_aggregate import aggregate_all_methods, aggregate_facility_traces  # noqa: E402


def test_hierarchical_aggregation_and_power_placement():
    # 15 minutes at 250ms => 3600 samples.
    t = 3600
    dt = 0.25
    gpu_w = 100.0
    overhead_w = 1000.0
    pue = 1.3

    with tempfile.TemporaryDirectory() as td:
        node_dir = os.path.join(td, "node_traces")
        out_dir = os.path.join(td, "aggregated")
        os.makedirs(node_dir, exist_ok=True)

        rows, racks_per_row, nodes_per_rack = 2, 2, 2
        for i in range(rows):
            for j in range(racks_per_row):
                for k in range(nodes_per_rack):
                    path = os.path.join(node_dir, f"node_{i}_{j}_{k}.npy")
                    np.save(path, np.full((t,), gpu_w, dtype=np.float32))

        meta = aggregate_facility_traces(
            node_trace_dir=node_dir,
            out_dir=out_dir,
            dt=dt,
            rows=rows,
            racks_per_row=racks_per_row,
            nodes_per_rack=nodes_per_rack,
            non_gpu_overhead_w=overhead_w,
            pue=pue,
        )

        assert meta["status"] == "ok"
        assert meta["layout"]["n_nodes"] == 8

        rack = np.asarray(np.load(os.path.join(out_dir, "rack_0_0.npy")), dtype=np.float64).reshape(-1)
        row = np.asarray(np.load(os.path.join(out_dir, "row_0.npy")), dtype=np.float64).reshape(-1)
        site_it_250 = np.asarray(np.load(os.path.join(out_dir, "site_it_250ms.npy")), dtype=np.float64).reshape(-1)
        site_250 = np.asarray(np.load(os.path.join(out_dir, "site_250ms.npy")), dtype=np.float64).reshape(-1)
        site_1s = np.asarray(np.load(os.path.join(out_dir, "site_1s.npy")), dtype=np.float64).reshape(-1)
        site_1min = np.asarray(np.load(os.path.join(out_dir, "site_1min.npy")), dtype=np.float64).reshape(-1)
        site_15min = np.asarray(np.load(os.path.join(out_dir, "site_15min.npy")), dtype=np.float64).reshape(-1)

        # Rack: 2 nodes * (gpu + overhead)
        assert np.allclose(rack, 2.0 * (gpu_w + overhead_w))
        # Row: 2 racks
        assert np.allclose(row, 4.0 * (gpu_w + overhead_w))
        # Site IT: 8 nodes
        assert site_it_250.shape == (t,)
        assert np.allclose(site_it_250, 8.0 * (gpu_w + overhead_w))
        # Site facility: apply PUE only at site level.
        assert np.allclose(site_250, site_it_250 * pue)
        assert np.allclose(site_1s, (8.0 * (gpu_w + overhead_w)) * pue)
        assert np.allclose(site_1min, (8.0 * (gpu_w + overhead_w)) * pue)
        assert site_15min.shape == (1,)
        assert np.allclose(site_15min, (8.0 * (gpu_w + overhead_w)) * pue)


def test_aggregate_all_methods_outputs():
    t = 3600  # 15 minutes at 250ms
    dt = 0.25
    overhead_w = 1000.0
    pue = 1.3
    methods = ["ours", "splitwise_strict"]

    with tempfile.TemporaryDirectory() as td:
        node_root = os.path.join(td, "results", "azure_facility", "node_traces")
        out_root = os.path.join(td, "results", "azure_facility", "aggregated")

        rows, racks_per_row, nodes_per_rack = 2, 2, 2
        for method_idx, method in enumerate(methods):
            method_dir = os.path.join(node_root, method)
            os.makedirs(method_dir, exist_ok=True)
            base_gpu = 100.0 + (10.0 * method_idx)
            for i in range(rows):
                for j in range(racks_per_row):
                    for k in range(nodes_per_rack):
                        path = os.path.join(method_dir, f"node_{i}_{j}_{k}.npy")
                        np.save(path, np.full((t,), base_gpu + i + j + k, dtype=np.float32))

        summary = aggregate_all_methods(
            node_traces_root=node_root,
            aggregated_root=out_root,
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
        assert os.path.exists(os.path.join(out_root, "aggregation_summary.json"))

        for method in methods:
            method_out = os.path.join(out_root, method)
            assert os.path.exists(os.path.join(method_out, "site_250ms.npy"))
            assert os.path.exists(os.path.join(method_out, "site_1s.npy"))
            assert os.path.exists(os.path.join(method_out, "site_1min.npy"))
            assert os.path.exists(os.path.join(method_out, "site_15min.npy"))
            assert os.path.exists(os.path.join(method_out, "aggregation_metadata.json"))

        ours_site = np.asarray(
            np.load(os.path.join(out_root, "ours", "site_250ms.npy")),
            dtype=np.float64,
        ).reshape(-1)
        expected = (8.0 * ((100.0 + 1.5) + overhead_w)) * pue
        assert np.allclose(np.mean(ours_site), expected, atol=5.0)
