"""
Claim:
Both raw layouts produce one RunRecord that preserves per-GPU measurements,
request timing, identity, architecture, and provenance; its feature views are
equivalent to the established legacy paths.

Plausible wrong implementations:
- Collapse GPU rows during ingestion or sum the wrong TP group.
- Reorder or independently filter request columns.
- Read bundle identity from paths instead of the emitted manifest.
- Accept incomplete bundles or omit source hashes.
- Shift timestamps while constructing the record rather than in a named view.
- Couple the GRU-only legacy projection to the physics architecture registry.
"""

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from model.training_data.alignment import align_trace_to_grid
from model.training_data.power_parsing import parse_power_csv, parse_request_json
from model.training_data.run_record import load_bundle_run, load_legacy_run
from profiling.client.run_manifest import build_manifest, write_manifest

EPOCH_TEXT = "2024/01/01 10:00:{s:02d}.{ms:03d}"


def _write_legacy_pair(root: Path, *, n_samples=40, gpus=8, n_req=12):
    power_csv = root / "llama-3-70b_tp8_p1.0_d20240101.csv"
    with open(power_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "power.draw [W]", "utilization.gpu [%]", "memory.used [MiB]"]
        )
        for s in range(n_samples):
            ms = (s * 250) % 1000
            sec = (s * 250) // 1000
            for g in range(gpus):
                writer.writerow(
                    [
                        EPOCH_TEXT.format(s=sec, ms=ms + g),
                        f"{100.0 + s + g:.1f} W",
                        f"{50 + g} %",
                        f"{10000 + g} MiB",
                    ]
                )

    # Arrivals inside the power window (epoch of 2024/01/01 10:00:00 UTC).
    t0 = 1704103200.0
    json_path = root / "vllm-1.0qps-tp8-llama-70b-20240101.json"
    payload = {
        "input_lens": [256] * n_req,
        "output_lens": [64] * n_req,
        "ttfts": [0.4] * n_req,
        "itls": [[0.02] * 64] * n_req,
        "request_timestamps": [t0 + 1.0 + 0.5 * i for i in range(n_req)],
    }
    json_path.write_text(json.dumps(payload))

    return {
        "status": "matched",
        "model_name": "llama-3-70b",
        "hardware": "H100",
        "tensor_parallelism": "8",
        "rate": "1.0",
        "pair_key": "tp=8|rate=1.0|date=20240101",
        "power_csv_path": str(power_csv),
        "json_path": str(json_path),
    }


class TestLoadLegacyRun(unittest.TestCase):
    def test_record_fields_and_per_gpu_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = _write_legacy_pair(Path(tmp))
            rec = load_legacy_run(row)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.config_id, "llama-3-70b_H100_tp8")
            self.assertEqual(rec.source_layout, "sharegpt")
            self.assertEqual(rec.timestamp_source, "recorded")
            self.assertEqual(rec.power_per_gpu.shape, (40, 8))
            self.assertIsNone(rec.node_power)
            # util/mem survive into the record (legacy parse drops them).
            self.assertAlmostEqual(float(rec.util_per_gpu[0, 3]), 53.0)
            self.assertAlmostEqual(float(rec.mem_per_gpu[0, 7]), 10007.0)
            self.assertEqual(rec.arch["family"], "dense-70b")
            self.assertEqual(rec.clock_basis, "naive_local_as_utc")

    def test_tp_sum_and_gru_view_match_legacy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = _write_legacy_pair(Path(tmp))
            rec = load_legacy_run(row)

            legacy_power = parse_power_csv(row["power_csv_path"], tensor_parallelism=8)
            legacy_req = parse_request_json(row["json_path"])
            np.testing.assert_array_equal(rec.tp_sum_power(), legacy_power["power"])
            np.testing.assert_array_equal(
                rec.power_timestamps, legacy_power["timestamps"]
            )
            np.testing.assert_array_equal(
                rec.request_timestamps, legacy_req["request_timestamps"]
            )

            legacy_aligned = align_trace_to_grid(legacy_power, legacy_req)
            record_aligned = align_trace_to_grid(
                {"timestamps": rec.power_timestamps, "power": rec.tp_sum_power()},
                {
                    "request_timestamps": rec.request_timestamps,
                    "ttfts": rec.ttfts,
                    "decode_times": rec.decode_times,
                    "input_lens": rec.input_lens,
                    "output_lens": rec.output_lens,
                    "has_timestamps": rec.has_timestamps,
                },
            )
            for key in ("power", "active_requests", "t_arrive_log"):
                np.testing.assert_array_equal(record_aligned[key], legacy_aligned[key])

    def test_missing_files_return_none(self):
        rec = load_legacy_run(
            {
                "model_name": "llama-3-70b",
                "hardware": "H100",
                "tensor_parallelism": "8",
                "power_csv_path": "/nonexistent.csv",
                "json_path": "/nonexistent.json",
            }
        )
        self.assertIsNone(rec)

    def test_gru_projection_can_explicitly_omit_unknown_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = _write_legacy_pair(Path(tmp))
            row["model_name"] = "unregistered-model"
            with self.assertRaises(KeyError):
                load_legacy_run(row)
            rec = load_legacy_run(row, require_arch=False)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.arch, {})
            self.assertEqual(
                rec.provenance["arch_source"], "omitted_for_gru_projection"
            )

    def test_legacy_missing_gpu_power_is_preserved_without_blocking_tp_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _write_legacy_pair(root)
            power_path = Path(row["power_csv_path"])
            with power_path.open(newline="") as f:
                rows = list(csv.reader(f))
            rows[1][1] = ""
            with power_path.open("w", newline="") as f:
                csv.writer(f).writerows(rows)

            record = load_legacy_run(row)

            self.assertIsNotNone(record)
            assert record is not None
            self.assertTrue(np.isnan(record.power_per_gpu[0, 0]))
            self.assertTrue(np.all(np.isfinite(record.tp_sum_power())))


class TestLoadBundleRun(unittest.TestCase):
    def test_emitter_consumer_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            power_csv = root / "power.csv"
            with open(power_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp", "index", "uuid", "power.draw",
                        "clocks.sm", "clocks.mem", "utilization.gpu",
                        "utilization.memory", "memory.used", "temperature.gpu",
                    ]
                )
                for sample in range(3):
                    for gpu in range(8):
                        writer.writerow(
                            [
                                f"2024/01/01 10:00:00.{sample * 250:03d}",
                                gpu, f"GPU-{gpu}", 100 + 10 * sample + gpu,
                                1900 + gpu, 2600 + gpu, 40 + gpu, 20 + gpu,
                                10000 + gpu, 60 + gpu,
                            ]
                        )
            (root / "engine.csv").write_text(
                "timestamp,num_requests_running\n1704103200.0,1\n"
            )
            request_payload = {
                "input_lens": [128, 256],
                "output_lens": [4, 3],
                "ttfts": [0.2, 0.3],
                "itls": [[0.1] * 4, [0.2] * 3],
                "request_timestamps": [1704103200.1, 1704103200.4],
            }
            (root / "requests.json").write_text(json.dumps(request_payload))
            arch = {
                "family": "unit",
                "n_active": 1.0,
                "w_bytes": 2.0,
                "d_model": 4,
                "n_layers": 2,
                "n_kv": 1,
                "head_dim": 2,
                "moe_frac": 0.0,
                "n_experts": 1,
                "top_k": 1,
                "swa_window": 0,
                "fp8": 0,
            }
            manifest = build_manifest(
                run_id="unit-run",
                probe={"type": "unit", "levels": []},
                model="unit/model",
                arch=arch,
                hardware="H100",
                tp=4,
                gpus_per_node=8,
                server={"active_gpu_uuids": [f"GPU-{gpu}" for gpu in range(4)]},
                versions={"git_sha": "unit"},
                clock={"local_utc_offset_s": 0.0},
            )
            write_manifest(str(root / "manifest.json"), manifest)

            record = load_bundle_run(root)

            self.assertEqual(record.config_id, "unit/model_H100_tp4")
            self.assertEqual(record.source_layout, "bundle")
            self.assertEqual(
                record.clock_basis, "power_local_wall_time_corrected_to_epoch"
            )
            self.assertEqual(record.arch, arch)
            np.testing.assert_array_equal(
                record.tp_sum_power(),
                np.asarray([406.0, 446.0, 486.0]),
            )
            np.testing.assert_array_equal(record.input_lens, [128.0, 256.0])
            np.testing.assert_allclose(record.decode_times, [0.4, 0.6])
            self.assertEqual(record.device_ids, ("GPU-0", "GPU-1", "GPU-2", "GPU-3", "GPU-4", "GPU-5", "GPU-6", "GPU-7"))
            self.assertAlmostEqual(record.device_table["clocks.sm"][1, 6], 1906.0)
            self.assertAlmostEqual(record.device_table["temperature.gpu"][2, 7], 67.0)
            self.assertEqual(record.provenance["run_id"], "unit-run")
            self.assertEqual(
                set(record.provenance["sha256"]),
                {"manifest.json", "power.csv", "engine.csv", "requests.json"},
            )

    def test_incomplete_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Incomplete bundle"):
                load_bundle_run(tmp)

    def test_manifest_topology_mismatch_is_rejected_without_folding_samples(self):
        """Four logged devices must never be consumed as eight-device blocks."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "power.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "index", "uuid", "power.draw"])
                for sample in range(3):
                    ts = f"2024/01/01 10:00:00.{sample * 250:03d}"
                    for gpu in range(4):
                        writer.writerow([ts, gpu, f"GPU-{gpu}", 100 + gpu])
            (root / "engine.csv").write_text("timestamp\n1704103200.0\n")
            (root / "requests.json").write_text(json.dumps({
                "input_lens": [8], "output_lens": [1], "ttfts": [0.1],
                "itls": [[]], "request_timestamps": [1704103200.1],
            }))
            (root / "manifest.json").write_text(json.dumps({
                "manifest_version": 1, "run_id": "bad-topology", "model": "unit/model",
                "hardware": "H100", "tp": 4, "gpus_per_node": 8,
                "arch": {"family": "unit"}, "clock": {"local_utc_offset_s": 0.0},
            }))
            with self.assertRaisesRegex(ValueError, "declares 8 GPUs.*contains 4"):
                load_bundle_run(root)

    def test_request_extensions_and_engine_columns_are_preserved(self):
        """Model projection stays narrow while raw per-row columns survive."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "power.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "index", "uuid", "power.draw"])
                for sample in range(2):
                    ts = f"2024/01/01 10:00:00.{sample * 250:03d}"
                    writer.writerow([ts, 0, "GPU-a", 100 + sample])
            (root / "engine.csv").write_text(
                "timestamp,num_requests_running,custom_counter\n"
                "1704103200.0,1,9\n1704103200.25,2,11\n"
            )
            (root / "requests.json").write_text(json.dumps({
                "input_lens": [8], "output_lens": [1], "ttfts": [0.1],
                "itls": [[]], "request_timestamps": [1704103200.1],
                "session_ids": ["s-1"], "tool_class": ["search"],
            }))
            (root / "manifest.json").write_text(json.dumps({
                "manifest_version": 1, "run_id": "preserve", "model": "unit/model",
                "hardware": "H100", "tp": 1, "gpus_per_node": 1,
                "arch": {"family": "unit"}, "clock": {"local_utc_offset_s": 0.0},
            }))
            record = load_bundle_run(root)
            self.assertEqual(record.request_table["session_ids"].tolist(), ["s-1"])
            self.assertEqual(record.request_table["tool_class"].tolist(), ["search"])
            np.testing.assert_array_equal(record.engine_table["custom_counter"], [9.0, 11.0])
            self.assertEqual(record.decode_times.tolist(), [0.0])
            self.assertEqual(record.provenance["request_projection_indices"], [0])
            self.assertEqual(
                record.provenance["request_rows"]["request_column_lengths"]["session_ids"],
                1,
            )
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["server"] = {"active_gpu_uuids": ["GPU-wrong"]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "do not match the TP-group"):
                load_bundle_run(root)

    def test_record_rejects_nonmonotonic_power_and_ragged_engine_table(self):
        """Time order and engine row identity are boundary invariants."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "power.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "index", "uuid", "power.draw"])
                writer.writerow(["2024/01/01 10:00:00.000", 0, "GPU-a", 100])
                writer.writerow(["2024/01/01 10:00:00.250", 0, "GPU-a", 101])
            (root / "engine.csv").write_text("timestamp,x\n1704103200,1\n")
            (root / "requests.json").write_text(json.dumps({
                "input_lens": [8], "output_lens": [1], "ttfts": [0.1],
                "itls": [[]], "request_timestamps": [1704103200.1],
            }))
            (root / "manifest.json").write_text(json.dumps({
                "manifest_version": 2, "run_id": "invariants", "model": "unit/model",
                "hardware": "H100", "tp": 1, "gpus_per_node": 1,
                "arch": {"family": "unit"}, "clock": {"local_utc_offset_s": 0.0},
            }))
            record = load_bundle_run(root)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                replace(record, power_timestamps=np.asarray([2.0, 1.0]))
            with self.assertRaisesRegex(ValueError, "engine columns"):
                replace(record, engine_table={"timestamp": np.asarray([1.0]), "x": np.asarray([1.0, 2.0])})


if __name__ == "__main__":
    unittest.main()
