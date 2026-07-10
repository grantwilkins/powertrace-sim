"""B1 tests: per-GPU power parsing (D4) and the shared request-row core (D6).

Claims under test:
- parse_power_csv_per_gpu preserves per-device power/util/memory that the
  legacy collapsed parse drops, and tp_sum_power over it reproduces the legacy
  TP-sum bit-for-bit (including NaN cells).
- extract_request_rows is a pure alignment core; both full readers reject
  negative token counts; an empty
  request_timestamps list is ignored by the first and aligns to zero rows in
  the second; decode-time drops take precedence over field drops in the
  throughput reader.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.training_data.power_parsing import (
    extract_request_rows,
    parse_power_csv,
    parse_power_csv_per_gpu,
    parse_request_json,
    tp_sum_power,
)
from model.training_data.throughput import extract_request_metrics


def _write_raw_power_csv(path: Path, samples, gpus: int = 8) -> None:
    """samples: list of per-sample lists of (power, util, mem) tuples."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "power.draw [W]", "utilization.gpu [%]", "memory.used [MiB]"]
        )
        for s, sample in enumerate(samples):
            base_ms = s * 250
            for g, (pw, util, mem) in enumerate(sample):
                ts = f"2024/01/01 10:00:{base_ms // 1000:02d}.{(base_ms % 1000) + g:03d}"
                pw_cell = "" if pw is None else f"{pw:.1f} W"
                writer.writerow([ts, pw_cell, f"{util:.0f} %", f"{mem:.0f} MiB"])


class TestPerGpuPowerParsing(unittest.TestCase):
    def _samples(self):
        # Two samples x 8 GPUs; one NaN power cell (empty string) in sample 2.
        s1 = [(10.0 + g, 50.0 + g, 10000.0 + g) for g in range(8)]
        s2 = [(20.0 + g, 60.0 + g, 20000.0 + g) for g in range(8)]
        s2[1] = (None, 61.0, 20001.0)  # unparseable power -> NaN
        return [s1, s2]

    def test_per_gpu_shapes_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            _write_raw_power_csv(path, self._samples())
            out = parse_power_csv_per_gpu(str(path))
            self.assertIsNotNone(out)
            self.assertEqual(out["power_per_gpu"].shape, (2, 8))
            self.assertEqual(out["util_per_gpu"].shape, (2, 8))
            self.assertEqual(out["mem_per_gpu"].shape, (2, 8))
            self.assertAlmostEqual(float(out["power_per_gpu"][0, 3]), 13.0)
            self.assertTrue(np.isnan(out["power_per_gpu"][1, 1]))
            self.assertAlmostEqual(float(out["util_per_gpu"][1, 1]), 61.0)
            self.assertAlmostEqual(float(out["mem_per_gpu"][0, 7]), 10007.0)

    def test_tp_sum_matches_legacy_parse_including_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.csv"
            _write_raw_power_csv(path, self._samples())
            per_gpu = parse_power_csv_per_gpu(str(path))
            for tp in (1, 4, 8):
                legacy = parse_power_csv(str(path), tensor_parallelism=tp)
                summed = tp_sum_power(per_gpu["power_per_gpu"], tp)
                np.testing.assert_array_equal(summed, legacy["power"])
                np.testing.assert_array_equal(
                    per_gpu["timestamps"], legacy["timestamps"]
                )
            # Hand-worked: sample 2, tp=4 -> 20 + NaN(=0) + 22 + 23 = 65.
            self.assertAlmostEqual(
                float(tp_sum_power(per_gpu["power_per_gpu"], 4)[1]), 65.0
            )

    def test_aggregated_trace_has_no_per_gpu_form(self):
        # One row per second: not the raw fixed-block stream.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agg.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "power.draw [W]"])
                for s in range(20):
                    writer.writerow([f"2024/01/01 10:00:{s:02d}.000", f"{100 + s}.0 W"])
            self.assertIsNone(parse_power_csv_per_gpu(str(path)))
            legacy = parse_power_csv(str(path), tensor_parallelism=1)
            self.assertIsNotNone(legacy)
            self.assertEqual(len(legacy["power"]), 20)

    def test_identity_schema_groups_by_timestamp_not_row_blocks(self):
        """Device row permutations cannot change the per-device time series."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "identified.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "index", "uuid", "power.draw", "clocks.sm",
                    "clocks.mem", "utilization.gpu", "utilization.memory",
                    "memory.used", "temperature.gpu",
                ])
                for sample, order in enumerate(([2, 0, 3, 1], [1, 3, 0, 2])):
                    ts = f"2024/01/01 10:00:00.{sample * 250:03d}"
                    for gpu in order:
                        writer.writerow([
                            ts, gpu, f"GPU-{gpu}", 100 * sample + gpu,
                            1900 + gpu, 2600 + gpu, 40 + gpu, 20 + gpu,
                            10000 + gpu, 60 + gpu,
                        ])
            out = parse_power_csv_per_gpu(str(path), gpus_per_node=4, strict_topology=True)
            self.assertEqual(out["device_ids"], ("GPU-0", "GPU-1", "GPU-2", "GPU-3"))
            np.testing.assert_array_equal(out["power_per_gpu"], [[0, 1, 2, 3], [100, 101, 102, 103]])
            np.testing.assert_array_equal(out["device_table"]["clocks.sm"][0], [1900, 1901, 1902, 1903])

    def test_identity_schema_rejects_declared_eight_when_four_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "four.csv"
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "index", "uuid", "power.draw"])
                for sample in range(2):
                    ts = f"2024/01/01 10:00:00.{sample * 250:03d}"
                    for gpu in range(4):
                        writer.writerow([ts, gpu, f"GPU-{gpu}", 10 + gpu])
            with self.assertRaisesRegex(ValueError, "declares 8 GPUs.*contains 4"):
                parse_power_csv_per_gpu(str(path), gpus_per_node=8, strict_topology=True)


class TestRequestRowCore(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "input_lens": [100, 200, -5],
            "output_lens": [10, 20, 30],
            "ttfts": [0.5, 0.6, 0.7],
            "itls": [[0.01] * 10, 0.02, [0.03] * 30],  # list and scalar forms
            "request_timestamps": [1000.0, 1001.0, 1002.0],
        }
        payload.update(overrides)
        return payload

    def test_core_aligns_and_derives_decode_times(self):
        rows = extract_request_rows(self._payload())
        self.assertEqual(rows["n_base"], 3)
        self.assertTrue(rows["has_timestamps_array"])
        # list itls: sum; scalar itls: itl * (n_out - 1).
        self.assertAlmostEqual(rows["decode_times"][0], 0.1, places=9)
        self.assertAlmostEqual(rows["decode_times"][1], 0.02 * 19, places=9)

    def test_core_rejects_non_list_arrays(self):
        self.assertIsNone(extract_request_rows(self._payload(itls="oops")))

    def test_gru_reader_rejects_negative_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(self._payload()))
            out = parse_request_json(str(path))
            self.assertIsNotNone(out)
            self.assertEqual(len(out["input_lens"]), 2)
            self.assertEqual(out["stats"]["num_requests_dropped_invalid_fields"], 1)

    def test_throughput_reader_drops_negative_tokens(self):
        out = extract_request_metrics(self._payload())
        stats = out["stats"]
        self.assertEqual(stats["num_requests_used"], 2)
        self.assertEqual(stats["num_requests_dropped_invalid_fields"], 1)

    def test_empty_timestamp_list_policies_differ(self):
        payload = self._payload(request_timestamps=[])
        # GRU reader: empty list == missing; optional mode proceeds NaN-filled.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(payload))
            self.assertIsNone(parse_request_json(str(path)))
            optional = parse_request_json(str(path), require_request_timestamps=False)
            self.assertEqual(len(optional["input_lens"]), 2)
            self.assertFalse(optional["has_timestamps"])
        # Throughput reader: empty list aligns everything away.
        stats = extract_request_metrics(payload)["stats"]
        self.assertEqual(stats["num_requests_aligned"], 0)
        self.assertEqual(stats["num_requests_used"], 0)

    def test_decode_drop_takes_precedence_over_field_drop(self):
        # Row 0 has BOTH an empty itls list (decode drop) and a negative token
        # (field drop): the throughput reader must count it as decode-dropped.
        payload = self._payload(
            input_lens=[-100, 200],
            output_lens=[10, 20],
            ttfts=[0.5, 0.6],
            itls=[[], [0.02] * 20],
            request_timestamps=[1000.0, 1001.0],
        )
        stats = extract_request_metrics(payload)["stats"]
        self.assertEqual(stats["num_requests_dropped_decode_time"], 1)
        self.assertEqual(stats["num_requests_dropped_invalid_fields"], 0)
        self.assertEqual(stats["num_requests_used"], 1)

    def test_one_token_completion_preserves_prefill_only_request(self):
        payload = self._payload(
            input_lens=[100], output_lens=[1], ttfts=[0.25], itls=[[]],
            request_timestamps=[1000.0], session_ids=["s1"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(payload))
            out = parse_request_json(str(path))
        self.assertEqual(out["decode_times"].tolist(), [0.0])
        self.assertEqual(out["request_table"]["session_ids"].tolist(), ["s1"])

    def test_unequal_source_lengths_are_accounted_before_projection(self):
        payload = self._payload(
            input_lens=[1, 2, 3, 4],
            output_lens=[1, 2, 3],
            ttfts=[0.1, 0.1],
            itls=[[], []],
            request_timestamps=[1000.0, 1001.0, 1002.0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(payload))
            out = parse_request_json(str(path))
        self.assertEqual(out["stats"]["source_lengths"]["input_lens"], 4)
        self.assertEqual(out["stats"]["num_requests_raw"], 4)
        self.assertEqual(out["stats"]["num_requests_aligned"], 2)
        self.assertEqual(out["stats"]["num_requests_dropped_unaligned"], 2)

    def test_all_list_valued_request_columns_have_recorded_lengths(self):
        payload = self._payload(
            session_ids=["s0", "s1", "s2"],
            tool_class=["search"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            path.write_text(json.dumps(payload))
            out = parse_request_json(str(path))
        assert out is not None
        self.assertEqual(out["stats"]["request_column_lengths"]["session_ids"], 3)
        self.assertEqual(out["stats"]["request_column_lengths"]["tool_class"], 1)
        self.assertEqual(out["projection_indices"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
