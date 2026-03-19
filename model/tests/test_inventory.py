import json
import tempfile
import unittest
from pathlib import Path

from model.training_data.inventory import (
    discover_pairs_for_dataset,
    normalize_sharegpt_date,
    parse_benchmark_filename,
    parse_sharegpt_csv_filename,
    parse_sharegpt_json_filename,
)
from model.training_data.stage0_inventory_and_throughput import (
    run_stage0_inventory_and_throughput,
)
from model.training_data.throughput import (
    concurrency_binned_decode_medians,
    extract_request_metrics,
    select_decode_model_type,
)
from model.utils.decode_time import derive_decode_time
from model.utils.io import write_json as _write_json


def _write_power_csv(path: Path, timestamps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("timestamp, power.draw [W], utilization.gpu [%], memory.used [MiB]\n")
        for ts in timestamps:
            for gpu in range(8):
                f.write(f"{ts}, {100.0 + gpu:.2f} W, 0 %, 10 MiB\n")


class TestStage0InventoryAndThroughput(unittest.TestCase):
    def test_filename_parsers(self):
        bench = parse_benchmark_filename(
            "gpt-oss-120b_tp8_rate1_iter1_2025-10-21-20-35-21.json"
        )
        self.assertIsNotNone(bench)
        self.assertEqual(bench["model"], "gpt-oss-120b")
        self.assertEqual(bench["tp"], 8)
        self.assertEqual(bench["rate"], "1")
        self.assertEqual(bench["iteration"], 1)

        sg_json = parse_sharegpt_json_filename(
            "vllm-0.25qps-tp1-Llama-3.1-8B-Instruct-20250513-021637.json"
        )
        self.assertIsNotNone(sg_json)
        self.assertEqual(sg_json["tp"], 1)
        self.assertEqual(sg_json["rate"], "0.25")
        self.assertEqual(sg_json["date"], "20250513-021637")

        sg_csv = parse_sharegpt_csv_filename("llama-3-8b_tp2_p4.0_d2025-05-13-12-10-19.csv")
        self.assertIsNotNone(sg_csv)
        self.assertEqual(sg_csv["tp"], 2)
        self.assertEqual(sg_csv["rate"], "4")
        self.assertEqual(sg_csv["date"], "20250513-121019")
        self.assertEqual(normalize_sharegpt_date("2025-05-13-12-10-19"), "20250513-121019")

    def test_pair_discovery_with_mixed_date_and_unmatched(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            ds = data_root / "sharegpt-benchmark-foo-a100"
            ds.mkdir(parents=True, exist_ok=True)

            _write_json(
                ds / "vllm-1.0qps-tp1-Model-20250101-000001.json",
                {"input_lens": [], "output_lens": [], "ttfts": [], "itls": []},
            )
            _write_power_csv(
                ds / "foo_tp1_p1.0_d2025-01-01-00-00-01.csv",
                ["2025/01/01 00:00:00.000", "2025/01/01 00:00:00.250"],
            )
            # Unmatched extra csv.
            _write_power_csv(
                ds / "foo_tp1_p1.0_d2025-01-01-00-00-02.csv",
                ["2025/01/01 00:00:01.000", "2025/01/01 00:00:01.250"],
            )

            manifest, pairs, summary = discover_pairs_for_dataset("sharegpt-benchmark", str(ds))
            self.assertEqual(summary["num_matched_pairs"], 1)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(summary["num_power_only"], 1)
            statuses = sorted([row["status"] for row in manifest])
            self.assertEqual(statuses, ["matched", "power_only"])

    def test_decode_time_derivation(self):
        decode_time, kind = derive_decode_time([0.1, 0.2, 0.3], 4)
        self.assertEqual(kind, "list")
        self.assertAlmostEqual(decode_time, 0.6, places=8)

        decode_time, kind = derive_decode_time(0.01, 100)
        self.assertEqual(kind, "scalar")
        self.assertAlmostEqual(decode_time, 0.99, places=8)

        decode_time, _ = derive_decode_time([], 10)
        self.assertIsNone(decode_time)
        decode_time, _ = derive_decode_time(float("nan"), 10)
        self.assertIsNone(decode_time)

    def test_array_alignment_and_truncation(self):
        payload = {
            "input_lens": [10, 20, 30, 40],
            "output_lens": [5, 6, 7, 8],
            "ttfts": [1.0, 1.0, 1.0, 1.0],
            "itls": [[0.1], [0.1], [0.1], [0.1]],
            "request_timestamps": [1000.0, 1001.0],
        }
        extracted = extract_request_metrics(payload)
        stats = extracted["stats"]
        self.assertEqual(stats["num_requests_total"], 4)
        self.assertEqual(stats["num_requests_aligned"], 2)
        self.assertEqual(stats["num_requests_used"], 2)
        self.assertTrue(stats["mismatched_array_lengths"])

    def test_concurrency_model_selection(self):
        flat_windows = [
            (1.0, 2.0, 1.5, 100.0),
            (3.0, 4.0, 3.5, 102.0),
            (5.0, 6.0, 5.5, 98.0),
        ]
        flat_bins = concurrency_binned_decode_medians(flat_windows, min_bin_samples=1)
        model_type, ratio = select_decode_model_type(flat_bins, batch_variation_threshold=2.0)
        self.assertEqual(model_type, "constant")
        self.assertIsNone(ratio)

        varying_windows = [
            (10.1, 11.1, 10.6, 200.0),  # concurrency 2 at midpoint
            (10.3, 12.3, 11.3, 50.0),   # concurrency 1 at midpoint
            (12.6, 14.6, 13.6, 50.0),   # concurrency 1 at midpoint
        ]
        varying_bins = concurrency_binned_decode_medians(varying_windows, min_bin_samples=1)
        model_type, ratio = select_decode_model_type(varying_bins, batch_variation_threshold=2.0)
        self.assertEqual(model_type, "by_concurrency")
        self.assertGreater(ratio, 2.0)

    def test_smoke_cli_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"

            # Config without request timestamps -> constant fallback.
            bench_dir = data_root / "benchmark-bar-a100" / "tp1"
            _write_power_csv(
                bench_dir / "bar_tp1_rate1_iter1_2025-01-01-00-00-00.csv",
                ["2025/01/01 00:00:00.000", "2025/01/01 00:00:00.250"],
            )
            _write_json(
                bench_dir / "bar_tp1_rate1_iter1_2025-01-01-00-00-00.json",
                {
                    "input_lens": [10, 20],
                    "output_lens": [10, 20],
                    "ttfts": [1.0, 2.0],
                    "itls": [[0.1] * 9, [0.2] * 19],
                },
            )

            # Config with request timestamps + rate variation by concurrency.
            sg_dir = data_root / "sharegpt-benchmark-foo-a100"
            _write_power_csv(
                sg_dir / "foo_tp1_p1.0_d2025-01-01-00-00-01.csv",
                ["2025/01/01 00:00:10.000", "2025/01/01 00:00:10.250"],
            )
            _write_json(
                sg_dir / "vllm-1.0qps-tp1-Model-20250101-000001.json",
                {
                    "input_lens": [100, 100, 100],
                    "output_lens": [200, 100, 100],
                    "ttfts": [0.1, 0.1, 0.1],
                    "itls": [[1.0], [2.0], [2.0]],
                    "request_timestamps": [10.0, 10.2, 12.5],
                },
            )

            out_inventory = str(Path(tmp) / "results" / "stage0" / "data_inventory.json")
            out_manifest = str(Path(tmp) / "results" / "stage0" / "pair_manifest.csv")
            out_db = str(Path(tmp) / "model" / "config" / "throughput_database.json")

            run_stage0_inventory_and_throughput(
                data_root_dir=str(data_root),
                include_families=["benchmark", "sharegpt-benchmark"],
                out_inventory_json=out_inventory,
                out_pair_manifest_csv=out_manifest,
                out_throughput_db=out_db,
                min_bin_samples=1,
                batch_variation_threshold=1.1,
            )

            self.assertTrue(Path(out_inventory).exists())
            self.assertTrue(Path(out_manifest).exists())
            self.assertTrue(Path(out_db).exists())

            db = json.loads(Path(out_db).read_text())
            self.assertIn("configs", db)
            self.assertIn("foo_A100_tp1", db["configs"])
            self.assertIn("bar_A100_tp1", db["configs"])

            foo_cfg = db["configs"]["foo_A100_tp1"]
            bar_cfg = db["configs"]["bar_A100_tp1"]
            self.assertGreater(len(foo_cfg["decode_model"]["by_concurrency_bins"]), 0)
            self.assertEqual(foo_cfg["decode_model"]["type"], "by_concurrency")
            self.assertEqual(bar_cfg["decode_model"]["type"], "constant")


if __name__ == "__main__":
    unittest.main()
