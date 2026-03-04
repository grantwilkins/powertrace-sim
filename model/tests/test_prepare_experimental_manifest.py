"""Unit tests for prepare_experimental_manifest.py."""
import csv
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class TestPrepareExperimentalManifest(unittest.TestCase):
    def test_power_timestamp_parse_treats_naive_values_as_utc(self):
        from model.training_data.utils.prepare_experimental_manifest import (
            _power_timestamp_to_epoch,
        )

        ts_text = "2025/05/13 01:45:16.864"
        expected = datetime(
            2025, 5, 13, 1, 45, 16, 864000, tzinfo=timezone.utc
        ).timestamp()
        parsed = _power_timestamp_to_epoch(ts_text)
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(float(parsed), float(expected), places=6)

    def test_align_trace_rebases_out_of_range_request_timestamps(self):
        """Recorded arrivals on a different clock should be rebased to power time."""
        from model.training_data.utils.prepare_experimental_manifest import (
            _align_trace_to_grid,
        )

        power_data = {
            "timestamps": np.asarray([1000.00, 1000.25, 1000.50, 1000.75, 1001.00], dtype=np.float64),
            "power": np.asarray([220.0, 240.0, 260.0, 240.0, 220.0], dtype=np.float64),
        }
        # Request timestamps are on a different epoch/clock origin.
        request_data = {
            "request_timestamps": [10.0, 10.3, 10.6],
            "ttfts": [0.05, 0.05, 0.05],
            "decode_times": [0.20, 0.20, 0.20],
            "has_timestamps": True,
            "input_lens": [100, 120, 140],
            "output_lens": [20, 30, 40],
        }

        aligned = _align_trace_to_grid(power_data, request_data)
        self.assertIsNotNone(aligned)
        active = np.asarray(aligned["active_requests"], dtype=np.float64)
        t_arrive = np.asarray(aligned["t_arrive_log"], dtype=np.float64)

        self.assertGreater(float(np.max(active)), 0.0)
        self.assertGreater(int(np.count_nonzero(t_arrive > 0.0)), 0)

    def test_parse_power_csv_aggregates_fixed_8_row_groups(self):
        """Raw nvidia-smi rows should aggregate by contiguous 8-row groups."""
        from model.training_data.utils.prepare_experimental_manifest import _parse_power_csv

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            power_csv = root / "llama-3-8b_tp4_p1.0_d20240101.csv"

            # Two samples, each represented by 8 GPU rows.
            # First sample has slight per-row timestamp jitter (no exact duplicates).
            rows = [
                ("2024/01/01 10:00:00.000", 10.0),
                ("2024/01/01 10:00:00.001", 11.0),
                ("2024/01/01 10:00:00.002", 12.0),
                ("2024/01/01 10:00:00.003", 13.0),
                ("2024/01/01 10:00:00.004", 14.0),
                ("2024/01/01 10:00:00.005", 15.0),
                ("2024/01/01 10:00:00.006", 16.0),
                ("2024/01/01 10:00:00.007", 17.0),
                ("2024/01/01 10:00:00.250", 20.0),
                ("2024/01/01 10:00:00.250", 21.0),
                ("2024/01/01 10:00:00.250", 22.0),
                ("2024/01/01 10:00:00.250", 23.0),
                ("2024/01/01 10:00:00.250", 24.0),
                ("2024/01/01 10:00:00.250", 25.0),
                ("2024/01/01 10:00:00.250", 26.0),
                ("2024/01/01 10:00:00.250", 27.0),
            ]
            with open(power_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "power.draw [W]", "utilization.gpu [%]", "memory.used [MiB]"])
                for ts, pw in rows:
                    writer.writerow([ts, f"{pw:.1f} W", "50 %", "10000 MiB"])

            parsed = _parse_power_csv(str(power_csv), tensor_parallelism=4)
            self.assertIsNotNone(parsed)
            self.assertEqual(len(parsed["power"]), 2)
            self.assertAlmostEqual(float(parsed["power"][0]), 10.0 + 11.0 + 12.0 + 13.0, places=6)
            self.assertAlmostEqual(float(parsed["power"][1]), 20.0 + 21.0 + 22.0 + 23.0, places=6)

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
            )

            self.assertEqual(manifest["summary"]["num_configs_written"], 0)
            self.assertEqual(manifest["summary"]["num_configs_skipped"], 1)

    def test_requires_request_timestamps_by_default(self):
        """Traces without request_timestamps should be rejected by default."""
        from model.training_data.utils.prepare_experimental_manifest import (
            run_prepare_experimental_manifest,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage0_dir = root / "stage0"
            stage0_dir.mkdir(parents=True, exist_ok=True)

            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            power_csv = data_dir / "llama-3-8b_tp1_p1.0_d20240101.csv"
            json_path = data_dir / "vllm-1.0qps-tp1-Llama-3.1-8B-Instruct-20240101.json"

            with open(power_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "power.draw [W]"])
                # 3 samples * 8 GPU rows/sample
                for i in range(3):
                    for _ in range(8):
                        writer.writerow([f"2024/01/01 10:00:0{i}.000", f"{200 + i}.0 W"])

            # Intentionally omit request_timestamps.
            json_data = {
                "input_lens": [100, 110, 120],
                "output_lens": [50, 55, 60],
                "ttfts": [0.05, 0.06, 0.07],
                "itls": [[0.02] * 50, [0.02] * 55, [0.02] * 60],
            }
            with open(json_path, "w") as f:
                json.dump(json_data, f)

            pair_manifest_csv = stage0_dir / "pair_manifest.csv"
            with open(pair_manifest_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "status",
                        "model_name",
                        "hardware",
                        "tensor_parallelism",
                        "rate",
                        "pair_key",
                        "power_csv_path",
                        "json_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "matched",
                        "model_name": "llama-3-8b",
                        "hardware": "H100",
                        "tensor_parallelism": "1",
                        "rate": "1.0",
                        "pair_key": "tp=1|rate=1.0|date=20240101",
                        "power_csv_path": str(power_csv),
                        "json_path": str(json_path),
                    }
                )

            out_dir = root / "experimental_continuous_v1"
            manifest = run_prepare_experimental_manifest(
                pair_manifest_csv=str(pair_manifest_csv),
                out_dir=str(out_dir),
                min_traces_per_config=1,
            )

            self.assertEqual(manifest["summary"]["num_configs_written"], 0)
            self.assertEqual(manifest["summary"]["num_configs_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
