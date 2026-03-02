import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.classifiers.gru import GRUClassifier
from scripts.eval.baselines import (
    build_splitwise_lut_params,
    generate_marginal_gmm,
    generate_mean,
    generate_ours,
    generate_splitwise_lut,
    generate_tdp,
)
from scripts.eval.run_baselines_facility import run_baselines_facility
from scripts.eval.run_baselines_node_groundtruth import run_baselines_node_groundtruth
from scripts.eval.run_baselines_node import _is_moe_config, run_baselines_node


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_pair_manifest(path: Path, *, pair_key: str, json_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "dataset_dir",
                "dataset_name",
                "status",
                "model_name",
                "hardware",
                "tensor_parallelism",
                "rate",
                "iteration",
                "date_key",
                "pair_key",
                "power_csv_path",
                "json_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "family": "toy",
                "dataset_dir": "toy",
                "dataset_name": "toy",
                "status": "matched",
                "model_name": "toy",
                "hardware": "H100",
                "tensor_parallelism": "4",
                "rate": "1",
                "iteration": "0",
                "date_key": "20260101-000000",
                "pair_key": pair_key,
                "power_csv_path": "",
                "json_path": json_path,
            }
        )


def _write_perf_model_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["llama2-70b", "a100-80gb", 512, 1, 128, 1.02, 0.72, 196.0, 55.0, 7000.0, 4],
        ["llama2-70b", "a100-80gb", 512, 2, 128, 1.05, 0.74, 416.0, 60.0, 8000.0, 4],
        ["llama2-70b", "a100-80gb", 512, 4, 128, 1.01, 0.59, 845.0, 60.5, 8500.0, 4],
        ["llama2-70b", "a100-80gb", 512, 8, 128, 0.98, 0.43, 1600.0, 61.0, 9400.0, 4],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "hardware",
                "prompt_size",
                "batch_size",
                "token_size",
                "peak_power",
                "average_power",
                "prompt_time",
                "token_time",
                "e2e_time",
                "tensor_parallel",
            ]
        )
        for row in rows:
            writer.writerow(row)


def _build_toy_fixture(root: Path) -> dict:
    cfg = "toy-70b_H100_tp4"
    pair_key_train = "tp=1|rate=1|date=20260101-000001"
    pair_key_test = "tp=1|rate=1|date=20260101-000002"

    checkpoint_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "checkpoints" / "toy_H100_tp1_k3_f2_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = GRUClassifier(Dx=2, K=3, H=8, num_layers=1)
    torch.save(model.state_dict(), checkpoint_path)

    norm_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "norm_params" / "toy_H100_tp1.json"
    _write_json(
        norm_path,
        {
            "config_id": cfg,
            "dt": 0.25,
            "k": 3,
            "feature_set": "f2",
            "input_dim": 2,
            "hidden_dim": 8,
            "num_layers": 1,
            "active_mean": 0.0,
            "active_std": 1.0,
            "t_arrive_log_mean": 0.0,
            "t_arrive_log_std": 1.0,
            "delta_A_mean": 0.0,
            "delta_A_std": 1.0,
            "power_mean": 100.0,
            "power_std": 10.0,
            "power_min": 70.0,
            "power_max": 140.0,
        },
    )

    gmm_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "gmms" / "toy_H100_tp1_k3.json"
    _write_json(
        gmm_path,
        {
            "config_id": cfg,
            "k": 3,
            "covariance_type": "full",
            "means": [90.0, 110.0, 130.0],
            "variances": [4.0, 9.0, 16.0],
            "weights": [0.3, 0.4, 0.3],
            "order": [0, 1, 2],
            "label_map": [0, 1, 2],
            "aic": 0.0,
            "bic": 0.0,
        },
    )

    run_manifest_path = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2" / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {
            "schema_version": "continuous-v1-gmm-bigru-train-run-v1",
            "configs": {
                cfg: {
                    "status": "trained",
                    "checkpoint_path": str(checkpoint_path),
                    "norm_params_path": str(norm_path),
                    "gmm_params_path": str(gmm_path),
                    "k": 3,
                    "feature_set": "f2",
                    "input_dim": 2,
                    "hidden_dim": 8,
                    "num_layers": 1,
                }
            },
        },
    )

    power_train = np.array([96.0, 100.0, 102.0, 101.0, 103.0, 104.0], dtype=np.float64)
    power_test = np.array([98.0, 99.0, 101.0, 100.0, 103.0, 102.0], dtype=np.float64)
    dataset_path = root / "results" / "experimental_continuous_v1" / "datasets" / "toy_H100_tp1.npz"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dataset_path,
        config_id=np.asarray([cfg, cfg], dtype=object),
        dt=np.asarray([0.25], dtype=np.float64),
        pair_key=np.asarray([pair_key_train, pair_key_test], dtype=object),
        rate=np.asarray(["1", "1"], dtype=object),
        power_start_epoch_s=np.asarray([1000.0, 1000.0], dtype=np.float64),
        power=np.asarray([power_train, power_test], dtype=object),
    )

    split_path = root / "results" / "experimental_continuous_v1" / "splits" / "toy_H100_tp1.json"
    _write_json(split_path, {"train_indices": [0], "val_indices": [], "test_indices": [1]})

    experimental_manifest_path = root / "results" / "experimental_continuous_v1" / "manifest.json"
    _write_json(
        experimental_manifest_path,
        {
            "schema_version": "experimental-continuous-v1",
            "configs": {
                cfg: {
                    "dataset_npz": str(dataset_path),
                    "split_json": str(split_path),
                    "norm_params_json": str(norm_path),
                    "written": True,
                }
            },
        },
    )

    throughput_db_path = root / "model" / "config" / "throughput_database.json"
    _write_json(
        throughput_db_path,
        {
            "schema_version": "stage0-throughput-v1",
            "configs": {
                cfg: {
                    "prefill_rate_median_toks_per_s": 100.0,
                    "decode_rate_median_toks_per_s": 50.0,
                }
            },
        },
    )

    request_json_test = root / "data" / "requests_test.json"
    _write_json(
        request_json_test,
        {
            "input_lens": [32, 48, 16, 40],
            "output_lens": [20, 12, 10, 18],
            "request_timestamps": [1000.0, 1000.5, 1001.0, 1001.5],
            "duration": 2.0,
            "request_rate": 2.0,
        },
    )

    pair_manifest_path = root / "results" / "stage0" / "pair_manifest.csv"
    _write_pair_manifest(pair_manifest_path, pair_key=pair_key_test, json_path=str(request_json_test))

    perf_model_path = root / "data" / "perf_model.csv"
    _write_perf_model_csv(perf_model_path)

    ar1_params_dir = root / "results" / "continuous_v1_gmm_bigru" / "k3_f2_ar1_thresh" / "ar1_params"
    ar1_params_path = ar1_params_dir / f"{cfg}_ar1_params.json"
    _write_json(
        ar1_params_path,
        {
            "config_id": cfg,
            "phi": [0.5, 0.4, 0.3],
            "sigma_innov": [2.0, 2.0, 2.0],
            "sigma_marginal": [3.0, 3.0, 3.0],
            "phi_threshold": 0.3,
        },
    )

    return {
        "config_id": cfg,
        "run_manifest": run_manifest_path,
        "experimental_manifest": experimental_manifest_path,
        "throughput_db": throughput_db_path,
        "pair_manifest": pair_manifest_path,
        "perf_model_csv": perf_model_path,
        "ar1_params_dir": ar1_params_dir,
    }


class TestBaselineGenerators(unittest.TestCase):
    def test_moe_classification_policy(self):
        self.assertFalse(_is_moe_config("deepseek-r1-distill-70b_H100_tp4"))
        self.assertTrue(_is_moe_config("gpt-oss-20b_H100_tp2"))
        self.assertTrue(_is_moe_config("gpt-oss-120b_H100_tp8"))
        self.assertFalse(_is_moe_config("llama-3-70b_H100_tp4"))

    def test_generate_tdp(self):
        out_a100 = generate_tdp(4, {"hardware": "A100"})
        out_h100 = generate_tdp(3, {"hardware": "H100"})
        self.assertEqual(out_a100.shape, (4,))
        self.assertEqual(out_h100.shape, (3,))
        self.assertTrue(np.allclose(out_a100, 4300.0))
        self.assertTrue(np.allclose(out_h100, 5600.0))

    def test_generate_mean(self):
        train = np.array([100.0, 110.0, 90.0], dtype=np.float64)
        out = generate_mean(5, {}, train)
        self.assertEqual(out.shape, (5,))
        self.assertTrue(np.allclose(out, 100.0))

    def test_generate_marginal_gmm_seeded(self):
        gmm = {
            "means": np.array([100.0, 140.0], dtype=np.float64),
            "variances": np.array([4.0, 9.0], dtype=np.float64),
            "weights": np.array([0.6, 0.4], dtype=np.float64),
        }
        out_a = generate_marginal_gmm(20, {}, gmm, rng=np.random.default_rng(123))
        out_b = generate_marginal_gmm(20, {}, gmm, rng=np.random.default_rng(123))
        self.assertEqual(out_a.shape, (20,))
        self.assertTrue(np.all(np.isfinite(out_a)))
        self.assertTrue(np.allclose(out_a, out_b))

    def test_generate_ours_shape_finite(self):
        torch.manual_seed(0)
        model = GRUClassifier(Dx=2, K=2, H=4, num_layers=1).to("cpu")
        features = np.random.default_rng(7).normal(size=(16, 2)).astype(np.float32)
        gmm = {
            "means": np.array([100.0, 150.0], dtype=np.float64),
            "variances": np.array([9.0, 16.0], dtype=np.float64),
            "weights": np.array([0.5, 0.5], dtype=np.float64),
        }
        out = generate_ours(
            features,
            {
                "device": "cpu",
                "p0": 120.0,
                "decode_mode": "stochastic",
                "median_filter_window": 1,
                "ar1_params": {
                    "phi": np.array([0.4, 0.4], dtype=np.float64),
                    "sigma_innov": np.array([3.0, 3.0], dtype=np.float64),
                    "sigma_marginal": np.array([4.0, 4.0], dtype=np.float64),
                    "phi_threshold": 0.3,
                },
            },
            model,
            gmm,
            rng=np.random.default_rng(3),
        )
        self.assertEqual(out.shape, (16,))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_splitwise_lut_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf_model_path = root / "data" / "perf_model.csv"
            _write_perf_model_csv(perf_model_path)
            train = np.array([1100.0, 1200.0, 1400.0, 1600.0, 1800.0], dtype=np.float64)
            lut = build_splitwise_lut_params(
                config_id="toy-70b_H100_tp4",
                perf_model_csv=str(perf_model_path),
                train_power_flat=train,
            )
            self.assertTrue(lut["idle_w"] <= lut["decode_w"] <= lut["prefill_w"])
            a_raw = np.array([0.0, 2.0, 2.0, 0.0], dtype=np.float64)
            d_raw = np.array([0.0, 1.0, -0.1, 0.0], dtype=np.float64)
            out_a = generate_splitwise_lut(
                a_raw,
                d_raw,
                {"config_id": "toy-70b_H100_tp4", "tp": 4, "n_gpus_per_node": 8, "non_gpu_power_w": 1000.0},
                lut,
            )
            out_b = generate_splitwise_lut(
                a_raw,
                d_raw,
                {"config_id": "toy-70b_H100_tp4", "tp": 4, "n_gpus_per_node": 8, "non_gpu_power_w": 1000.0},
                lut,
            )
            self.assertEqual(out_a.shape, (4,))
            self.assertTrue(np.all(np.isfinite(out_a)))
            self.assertTrue(np.allclose(out_a, out_b))


class TestNodeBaselineSmoke(unittest.TestCase):
    def test_node_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = _build_toy_fixture(root)
            out_csv = root / "results" / "eval_paper" / "baselines_node_level.csv"

            run_baselines_node(
                run_manifest=str(fx["run_manifest"]),
                experimental_manifest=str(fx["experimental_manifest"]),
                throughput_db=str(fx["throughput_db"]),
                pair_manifest_csv=str(fx["pair_manifest"]),
                ar1_params_dir=str(fx["ar1_params_dir"]),
                out_csv=str(out_csv),
                config_ids=[fx["config_id"]],
                num_seeds=2,
                base_seed=11,
                device="cpu",
                decode_mode="stochastic",
                median_filter_window=1,
                splitwise_perf_model_csv=str(fx["perf_model_csv"]),
            )

            self.assertTrue(out_csv.exists())
            with open(out_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            methods = {row["method"] for row in rows}
            self.assertEqual(methods, {"tdp", "mean", "splitwise_lut", "ours"})

            for row in rows:
                self.assertEqual(row["status"], "evaluated")
                if row["method"] in {"tdp", "mean"}:
                    self.assertEqual(row["acf_note"], "N/A_constant_trace")
                    self.assertTrue(np.isnan(float(row["acf_r2"])))
                else:
                    self.assertEqual(row["acf_note"], "")


class TestFacilityBaselineSmoke(unittest.TestCase):
    def test_facility_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = _build_toy_fixture(root)
            out_csv = root / "results" / "eval_paper" / "baselines_facility_metrics.csv"
            traces_pdf = root / "figures" / "baselines_facility_traces.pdf"
            ldc_pdf = root / "figures" / "baselines_load_duration.pdf"

            run_baselines_facility(
                run_manifest=str(fx["run_manifest"]),
                experimental_manifest=str(fx["experimental_manifest"]),
                throughput_db=str(fx["throughput_db"]),
                pair_manifest_csv=str(fx["pair_manifest"]),
                ar1_params_dir=str(fx["ar1_params_dir"]),
                out_csv=str(out_csv),
                traces_pdf=str(traces_pdf),
                ldc_pdf=str(ldc_pdf),
                config_id=str(fx["config_id"]),
                n_nodes=3,
                duration_s=60.0,
                dt=0.25,
                lambda_req_per_s_per_node=0.5,
                pue=1.3,
                base_seed=9,
                device="cpu",
                decode_mode="stochastic",
                median_filter_window=1,
                splitwise_perf_model_csv=str(fx["perf_model_csv"]),
            )

            self.assertTrue(out_csv.exists())
            self.assertTrue(traces_pdf.exists())
            self.assertTrue(ldc_pdf.exists())
            self.assertGreater(traces_pdf.stat().st_size, 0)
            self.assertGreater(ldc_pdf.stat().st_size, 0)

            with open(out_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["method"] for row in rows}, {"tdp", "mean", "splitwise_lut", "ours"})

            for row in rows:
                self.assertEqual(row["status"], "evaluated")
                self.assertTrue(np.isfinite(float(row["par"])))
                self.assertTrue(np.isfinite(float(row["diversity_factor"])))
                self.assertTrue(np.isfinite(float(row["ldc_p99_kw"])))
                self.assertTrue(np.isfinite(float(row["ramp_p95_abs_kw_per_s"])))


class TestNodeGroundTruthReplaySmoke(unittest.TestCase):
    def test_node_groundtruth_runner_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = _build_toy_fixture(root)
            out_csv = root / "results" / "eval_paper" / "baselines_node_groundtruth_metrics.csv"
            out_pdf = root / "figures" / "baselines_node_groundtruth_trace.pdf"

            run_baselines_node_groundtruth(
                run_manifest=str(fx["run_manifest"]),
                experimental_manifest=str(fx["experimental_manifest"]),
                throughput_db=str(fx["throughput_db"]),
                pair_manifest_csv=str(fx["pair_manifest"]),
                config_id=str(fx["config_id"]),
                target_rate=1.0,
                out_csv=str(out_csv),
                out_plot_pdf=str(out_pdf),
                splitwise_perf_model_csv=str(fx["perf_model_csv"]),
                non_gpu_overhead_w=0.0,
            )

            self.assertTrue(out_csv.exists())
            self.assertTrue(out_pdf.exists())
            self.assertGreater(out_pdf.stat().st_size, 0)
            with open(out_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            methods = {row["method"] for row in rows}
            self.assertEqual(methods, {"tdp", "mean", "splitwise_lut", "ours"})
            for row in rows:
                self.assertEqual(row["config_id"], fx["config_id"])
                self.assertEqual(row["selection_mode"], "closest_rate_in_test_split")
                if row["method"] in {"tdp", "mean"}:
                    self.assertEqual(row["acf_note"], "N/A_constant_trace")
                    self.assertTrue(np.isnan(float(row["acf_r2"])))
                else:
                    self.assertEqual(row["acf_note"], "")


if __name__ == "__main__":
    unittest.main()
