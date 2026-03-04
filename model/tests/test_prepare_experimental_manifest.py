"""Unit tests for prepare_experimental_manifest.py."""
import csv
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np


class TestPrepareExperimentalManifest(unittest.TestCase):
    def test_parse_power_csv_raw_per_gpu_uses_fixed_8_row_blocks(self):
        from model.training_data.utils.prepare_experimental_manifest import _parse_power_csv

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "toy_tp4.csv"
            base = datetime(2024, 1, 1, 10, 0, 0)
            rows = []
            # Three sample windows, 8 GPUs each, with intra-window timestamp jitter.
            for block_idx, base_power in enumerate([10.0, 20.0, 30.0]):
                t0 = base.timestamp() + (0.25 * block_idx)
                for gpu_idx in range(8):
                    ts = datetime.fromtimestamp(t0 + (0.008 * gpu_idx)).strftime(
                        "%Y/%m/%d %H:%M:%S.%f"
                    )[:-3]
                    rows.append((ts, f"{base_power + gpu_idx:.2f} W"))

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "power.draw [W]"])
                for ts, pw in rows:
                    writer.writerow([ts, pw])

            parsed = _parse_power_csv(str(csv_path), tensor_parallelism=4, gpus_per_node=8)
            self.assertIsNotNone(parsed)
            power = np.asarray(parsed["power"], dtype=np.float64)
            self.assertEqual(power.size, 3)
            expected = np.asarray(
                [
                    10.0 + 11.0 + 12.0 + 13.0,
                    20.0 + 21.0 + 22.0 + 23.0,
                    30.0 + 31.0 + 32.0 + 33.0,
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(power, expected, rtol=0.0, atol=1e-9)

    def test_parse_power_csv_already_aggregated_keeps_single_series(self):
        from model.training_data.utils.prepare_experimental_manifest import _parse_power_csv

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "toy_tp1.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "power.draw [W]"])
                for i in range(6):
                    writer.writerow([f"2024/01/01 10:00:0{i}.000", f"{200 + i}.0 W"])

            parsed = _parse_power_csv(str(csv_path), tensor_parallelism=1, gpus_per_node=8)
            self.assertIsNotNone(parsed)
            power = np.asarray(parsed["power"], dtype=np.float64)
            self.assertEqual(power.size, 6)
            np.testing.assert_allclose(power, np.arange(200.0, 206.0), rtol=0.0, atol=1e-9)

    def test_full_pipeline_creates_manifest_and_datasets(self):
        """Test that the full pipeline creates expected outputs."""
        from model.training_data.utils.prepare_experimental_manifest import (
            run_prepare_experimental_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            stage0_dir = root / "stage0"
            stage0_dir.mkdir(parents=True, exist_ok=True)

            power_dir = root / "data" / "sharegpt-benchmark-llama-3-8b-H100"
            power_dir.mkdir(parents=True, exist_ok=True)

            n_pairs = 4
            pairs = []
            for i in range(n_pairs):
                power_csv = power_dir / f"llama-3-8b_tp1_p1.0_d2024010{i}.csv"
                json_path = power_dir / f"vllm-1.0qps-tp1-Llama-3.1-8B-Instruct-2024010{i}.json"

                n_samples = 20 + i * 5
                timestamps = [f"2024/01/0{i+1} 10:00:0{j}.000" for j in range(n_samples)]
                powers = [200.0 + j * 2 + np.random.randn() * 5 for j in range(n_samples)]

                with open(power_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "power.draw [W]", "utilization.gpu [%]", "memory.used [MiB]"])
                    for ts, pw in zip(timestamps, powers):
                        writer.writerow([ts, f"{pw:.1f} W", "50 %", "10000 MiB"])

                n_requests = 10 + i * 2
                json_data = {
                    "input_lens": [100 + j * 10 for j in range(n_requests)],
                    "output_lens": [50 + j * 5 for j in range(n_requests)],
                    "ttfts": [0.05 + j * 0.01 for j in range(n_requests)],
                    "itls": [[0.02] * (50 + j * 5) for j in range(n_requests)],
                    "request_timestamps": [
                        1704070800.0 + j * 5.0 for j in range(n_requests)
                    ],
                }
                with open(json_path, "w") as f:
                    json.dump(json_data, f)

                pairs.append({
                    "family": "sharegpt-benchmark",
                    "dataset_dir": str(power_dir),
                    "dataset_name": "sharegpt-benchmark-llama-3-8b-H100",
                    "status": "matched",
                    "model_name": "llama-3-8b",
                    "hardware": "H100",
                    "tensor_parallelism": "1",
                    "rate": "1.0",
                    "iteration": "",
                    "date_key": f"2024010{i}",
                    "pair_key": f"tp=1|rate=1.0|date=2024010{i}",
                    "power_csv_path": str(power_csv),
                    "json_path": str(json_path),
                })

            pair_manifest_csv = stage0_dir / "pair_manifest.csv"
            fieldnames = [
                "family", "dataset_dir", "dataset_name", "status",
                "model_name", "hardware", "tensor_parallelism", "rate",
                "iteration", "date_key", "pair_key", "power_csv_path", "json_path"
            ]
            with open(pair_manifest_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for pair in pairs:
                    writer.writerow(pair)

            out_dir = root / "experimental_continuous_v1"

            manifest = run_prepare_experimental_manifest(
                pair_manifest_csv=str(pair_manifest_csv),
                out_dir=str(out_dir),
                train_ratio=0.5,
                val_ratio=0.25,
                seed=42,
                min_traces_per_config=2,
                allowed_json_prefix="",
            )

            self.assertEqual(manifest["summary"]["num_configs_written"], 1)
            self.assertTrue((out_dir / "manifest.json").exists())

            config_id = "llama-3-8b_H100_tp1"
            self.assertIn(config_id, manifest["configs"])
            config = manifest["configs"][config_id]
            self.assertTrue(config["written"])
            self.assertEqual(config["num_traces"], n_pairs)

            dataset_path = Path(config["dataset_npz"])
            self.assertTrue(dataset_path.exists())
            with np.load(dataset_path, allow_pickle=True) as data:
                self.assertEqual(len(data["power"]), n_pairs)
                self.assertEqual(len(data["active_requests"]), n_pairs)
                self.assertEqual(len(data["pair_key"]), n_pairs)

            split_path = Path(config["split_json"])
            self.assertTrue(split_path.exists())
            with open(split_path, "r") as f:
                split = json.load(f)
            self.assertIn("train_indices", split)
            self.assertIn("val_indices", split)
            self.assertIn("test_indices", split)
            total_indices = (
                len(split["train_indices"])
                + len(split["val_indices"])
                + len(split["test_indices"])
            )
            self.assertGreaterEqual(total_indices, n_pairs)

            norm_path = Path(config["norm_params_json"])
            self.assertTrue(norm_path.exists())
            with open(norm_path, "r") as f:
                norm = json.load(f)
            self.assertIn("power_mean", norm)
            self.assertIn("power_std", norm)
            self.assertIn("active_mean", norm)
            self.assertIn("active_std", norm)
            self.assertTrue(np.isfinite(norm["power_mean"]))
            self.assertGreater(norm["power_std"], 0)

    def test_skips_configs_with_insufficient_traces(self):
        """Test that configs with too few traces are skipped."""
        from model.training_data.utils.prepare_experimental_manifest import (
            run_prepare_experimental_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage0_dir = root / "stage0"
            stage0_dir.mkdir(parents=True, exist_ok=True)

            power_dir = root / "data"
            power_dir.mkdir(parents=True, exist_ok=True)

            power_csv = power_dir / "llama-3-8b_tp1_p1.0_d20240101.csv"
            json_path = power_dir / "vllm-1.0qps-tp1-Llama-3.1-8B-Instruct-20240101.json"

            with open(power_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "power.draw [W]"])
                for j in range(10):
                    writer.writerow([f"2024/01/01 10:00:0{j}.000", f"{200 + j}.0 W"])

            json_data = {
                "input_lens": [100, 110],
                "output_lens": [50, 55],
                "ttfts": [0.05, 0.06],
                "itls": [[0.02] * 50, [0.02] * 55],
                "request_timestamps": [1704070800.0, 1704070805.0],
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            pair_manifest_csv = stage0_dir / "pair_manifest.csv"
            with open(pair_manifest_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "status", "model_name", "hardware", "tensor_parallelism",
                    "rate", "pair_key", "power_csv_path", "json_path"
                ])
                writer.writeheader()
                writer.writerow({
                    "status": "matched",
                    "model_name": "llama-3-8b",
                    "hardware": "H100",
                    "tensor_parallelism": "1",
                    "rate": "1.0",
                    "pair_key": "test",
                    "power_csv_path": str(power_csv),
                    "json_path": str(json_path),
                })

            out_dir = root / "experimental_continuous_v1"
            manifest = run_prepare_experimental_manifest(
                pair_manifest_csv=str(pair_manifest_csv),
                out_dir=str(out_dir),
                min_traces_per_config=5,
                allowed_json_prefix="",
            )

            self.assertEqual(manifest["summary"]["num_configs_written"], 0)
            self.assertEqual(manifest["summary"]["num_configs_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
