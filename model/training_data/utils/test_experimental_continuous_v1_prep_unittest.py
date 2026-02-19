import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from model.training_data.utils.experimental_continuous_v1_prep import (
    TraceRecord,
    _align_request_times,
    _build_xy_for_trace,
    _compute_feature_series,
    _extract_power_series,
    _extract_request_log,
    _fit_norm_params,
    _split_indices,
    run_experimental_continuous_v1_prep,
)


def _ts(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s).strftime("%Y/%m/%d %H:%M:%S.%f")


def _write_power_csv(path: Path, groups: list[tuple[float, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "power.draw [W]", "utilization.gpu [%]", "memory.used [MiB]"])
        for ts, powers in groups:
            for p in powers:
                writer.writerow([_ts(ts), f"{p:.3f} W", "0 %", "0 MiB"])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


class TestExperimentalContinuousV1Prep(unittest.TestCase):
    def test_power_grouping_tp_sum_resample_and_short_gap_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "power.csv"
            _write_power_csv(
                csv_path,
                groups=[
                    (1000.00, [10.0, 20.0, 30.0, 40.0]),
                    (1000.24, [20.0, 20.0, 20.0, 20.0]),
                    (1000.50, [40.0, 40.0, 40.0, 40.0]),
                ],
            )
            power, t0, err = _extract_power_series(
                csv_path=str(csv_path),
                tp=2,
                dt=0.25,
                power_group_window_s=0.02,
                max_interp_gap_steps=3,
            )
            self.assertIsNone(err)
            self.assertIsNotNone(power)
            self.assertIsNotNone(t0)
            self.assertEqual(len(power), 3)
            self.assertAlmostEqual(float(power[0]), 35.0, places=6)
            self.assertAlmostEqual(float(power[1]), 57.5, places=6)
            self.assertAlmostEqual(float(power[2]), 80.0, places=6)

    def test_power_drop_when_gap_longer_than_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "power.csv"
            _write_power_csv(
                csv_path,
                groups=[
                    (1000.00, [10.0, 10.0]),
                    (1001.25, [20.0, 20.0]),  # gap = 4 bins between samples
                ],
            )
            power, _t0, err = _extract_power_series(
                csv_path=str(csv_path),
                tp=1,
                dt=0.25,
                power_group_window_s=0.02,
                max_interp_gap_steps=3,
            )
            self.assertIsNone(power)
            self.assertEqual(err, "power_unresolved_gap")

    def test_request_parsing_itl_list_and_scalar(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "requests.json"
            _write_json(
                json_path,
                {
                    "input_lens": [10, 20],
                    "output_lens": [3, 4],
                    "ttfts": [0.5, 0.4],
                    "itls": [[0.2, 0.3], 0.1],
                    "request_timestamps": [100.0, 101.0],
                },
            )
            records, err, stats = _extract_request_log(str(json_path))
            self.assertIsNone(err)
            self.assertEqual(stats["num_requests_used"], 2)
            self.assertEqual(len(records), 2)
            self.assertAlmostEqual(records[0]["total_latency"], 1.0, places=6)
            self.assertAlmostEqual(records[1]["total_latency"], 0.7, places=6)

    def test_missing_request_timestamps_uses_duration_and_rate_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "requests.json"
            _write_json(
                json_path,
                {
                    "input_lens": [10, 20],
                    "output_lens": [3, 4],
                    "ttfts": [0.5, 0.4],
                    "itls": [[0.2], [0.1]],
                    "duration": 4.0,
                    "request_rate": 0.5,
                },
            )
            records, err, stats = _extract_request_log(str(json_path))
            self.assertIsNone(err)
            self.assertEqual(stats["timestamp_source"], "synthetic_duration_uniform")
            self.assertIsNotNone(records)
            self.assertEqual(len(records), 2)
            self.assertGreater(records[0]["arrival_epoch_s"], 0.0)
            self.assertGreater(records[1]["arrival_epoch_s"], records[0]["arrival_epoch_s"])

    def test_missing_request_timestamps_without_fallback_fields_drops_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "requests.json"
            _write_json(
                json_path,
                {
                    "input_lens": [10],
                    "output_lens": [3],
                    "ttfts": [0.5],
                    "itls": [[0.2]],
                },
            )
            records, err, _stats = _extract_request_log(str(json_path))
            self.assertIsNone(records)
            self.assertEqual(err, "missing_request_timestamps")

    def test_shift_only_alignment_preserves_interarrival(self):
        requests = [
            {"arrival_epoch_s": 900.0, "input_tokens": 10.0, "output_tokens": 2.0, "ttft": 0.1, "total_latency": 0.5},
            {"arrival_epoch_s": 901.5, "input_tokens": 10.0, "output_tokens": 2.0, "ttft": 0.1, "total_latency": 0.5},
        ]
        aligned, shifted = _align_request_times(
            requests=requests,
            power_start_epoch_s=1000.0,
            trace_duration_s=10.0,
            dt=0.25,
        )
        self.assertTrue(shifted)
        self.assertAlmostEqual(aligned[0]["arrival_time"], 0.0, places=6)
        self.assertAlmostEqual(aligned[1]["arrival_time"], 1.5, places=6)
        self.assertAlmostEqual(
            aligned[1]["arrival_time"] - aligned[0]["arrival_time"],
            1.5,
            places=6,
        )

    def test_feature_series_active_requests_and_arrivals(self):
        power = np.zeros((5,), dtype=np.float64)
        requests = [
            {"arrival_time": 0.0, "input_tokens": 5, "output_tokens": 1, "ttft": 0.01, "total_latency": 0.1},
            {"arrival_time": 0.1, "input_tokens": 10, "output_tokens": 1, "ttft": 0.01, "total_latency": 0.4},
        ]
        active, t_arrive, t_arrive_log = _compute_feature_series(power=power, requests=requests, dt=0.25)
        np.testing.assert_allclose(active, np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(t_arrive, np.array([15.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(t_arrive_log, np.log1p(t_arrive))

    def test_norm_train_only_and_xy_shape_target(self):
        tr0 = TraceRecord(
            pair_key="p0",
            family="f",
            rate="1",
            iteration="1",
            power_start_epoch_s=0.0,
            power=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            active_requests=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            t_arrive=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            t_arrive_log=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            requests=[],
        )
        tr1 = TraceRecord(
            pair_key="p1",
            family="f",
            rate="1",
            iteration="1",
            power_start_epoch_s=0.0,
            power=np.array([4.0, 5.0, 6.0], dtype=np.float64),
            active_requests=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            t_arrive=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            t_arrive_log=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            requests=[],
        )
        tr2 = TraceRecord(
            pair_key="p2",
            family="f",
            rate="1",
            iteration="1",
            power_start_epoch_s=0.0,
            power=np.array([100.0, 100.0, 100.0], dtype=np.float64),
            active_requests=np.array([10.0, 10.0, 10.0], dtype=np.float64),
            t_arrive=np.array([10.0, 10.0, 10.0], dtype=np.float64),
            t_arrive_log=np.array([10.0, 10.0, 10.0], dtype=np.float64),
            requests=[],
        )
        norm = _fit_norm_params([tr0, tr1])
        self.assertAlmostEqual(norm["power_mean"], 3.5, places=6)  # train-only
        x, y = _build_xy_for_trace(tr0, norm=norm)
        self.assertEqual(tuple(x.shape), (2, 3))
        self.assertEqual(tuple(y.shape), (2,))
        expected_y = (np.array([2.0, 3.0], dtype=np.float64) - norm["power_mean"]) / norm["power_std"]
        np.testing.assert_allclose(y, expected_y.astype(np.float32))
        self.assertGreater(abs(norm["power_mean"] - np.mean(tr2.power)), 10.0)

    def test_split_determinism_and_min_val_test(self):
        tr_a, va_a, te_a = _split_indices(n=3, train_fraction=0.70, val_fraction=0.15, seed=123)
        tr_b, va_b, te_b = _split_indices(n=3, train_fraction=0.70, val_fraction=0.15, seed=123)
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(va_a, va_b)
        np.testing.assert_array_equal(te_a, te_b)
        self.assertEqual(len(tr_a), 1)
        self.assertEqual(len(va_a), 1)
        self.assertEqual(len(te_a), 1)

    def test_end_to_end_writes_artifacts_and_excludes_missing_timestamp_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "pair_manifest.csv"
            out_dir = root / "out"

            rows = []

            # Valid config with 3 traces.
            for i in range(3):
                p_csv = root / f"a_power_{i}.csv"
                j_json = root / f"a_req_{i}.json"
                _write_power_csv(
                    p_csv,
                    groups=[
                        (1000.00, [100.0, 10.0]),
                        (1000.25, [120.0, 10.0]),
                        (1000.50, [140.0, 10.0]),
                    ],
                )
                _write_json(
                    j_json,
                    {
                        "input_lens": [10, 5],
                        "output_lens": [3, 2],
                        "ttfts": [0.1, 0.1],
                        "itls": [[0.1, 0.1], [0.2]],
                        "request_timestamps": [1000.05 + (i * 0.01), 1000.30 + (i * 0.01)],
                    },
                )
                rows.append(
                    {
                        "family": "benchmark",
                        "dataset_dir": str(root),
                        "dataset_name": "ds",
                        "status": "matched",
                        "model_name": "model-a",
                        "hardware": "H100",
                        "tensor_parallelism": "1",
                        "rate": "1",
                        "iteration": str(i + 1),
                        "date_key": "k",
                        "pair_key": f"a|{i}",
                        "power_csv_path": str(p_csv),
                        "json_path": str(j_json),
                    }
                )

            # Fallback config: missing request_timestamps but has duration + request_rate.
            for i in range(3):
                p_csv = root / f"c_power_{i}.csv"
                j_json = root / f"c_req_{i}.json"
                _write_power_csv(
                    p_csv,
                    groups=[
                        (3000.00, [100.0, 10.0]),
                        (3000.25, [115.0, 10.0]),
                        (3000.50, [130.0, 10.0]),
                    ],
                )
                _write_json(
                    j_json,
                    {
                        "input_lens": [10, 5],
                        "output_lens": [3, 2],
                        "ttfts": [0.1, 0.1],
                        "itls": [[0.1, 0.1], [0.2]],
                        "duration": 3.0,
                        "request_rate": 1.0,
                    },
                )
                rows.append(
                    {
                        "family": "benchmark",
                        "dataset_dir": str(root),
                        "dataset_name": "ds",
                        "status": "matched",
                        "model_name": "model-c",
                        "hardware": "H100",
                        "tensor_parallelism": "1",
                        "rate": "1",
                        "iteration": str(i + 1),
                        "date_key": "k",
                        "pair_key": f"c|{i}",
                        "power_csv_path": str(p_csv),
                        "json_path": str(j_json),
                    }
                )

            # Invalid config: missing request_timestamps in JSON.
            for i in range(3):
                p_csv = root / f"b_power_{i}.csv"
                j_json = root / f"b_req_{i}.json"
                _write_power_csv(
                    p_csv,
                    groups=[
                        (2000.00, [100.0, 10.0]),
                        (2000.25, [110.0, 10.0]),
                        (2000.50, [120.0, 10.0]),
                    ],
                )
                _write_json(
                    j_json,
                    {
                        "input_lens": [10],
                        "output_lens": [3],
                        "ttfts": [0.1],
                        "itls": [[0.1, 0.1]],
                    },
                )
                rows.append(
                    {
                        "family": "benchmark",
                        "dataset_dir": str(root),
                        "dataset_name": "ds",
                        "status": "matched",
                        "model_name": "model-b",
                        "hardware": "H100",
                        "tensor_parallelism": "1",
                        "rate": "1",
                        "iteration": str(i + 1),
                        "date_key": "k",
                        "pair_key": f"b|{i}",
                        "power_csv_path": str(p_csv),
                        "json_path": str(j_json),
                    }
                )

            with open(manifest_path, "w", newline="") as f:
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
                for r in rows:
                    writer.writerow(r)

            manifest = run_experimental_continuous_v1_prep(
                pair_manifest_csv=str(manifest_path),
                out_dir=str(out_dir),
                seed=7,
                dt=0.25,
            )

            self.assertEqual(manifest["summary"]["num_configs_written"], 2)
            self.assertIn("model-a_H100_tp1", manifest["configs"])
            self.assertIn("model-b_H100_tp1", manifest["configs"])
            self.assertIn("model-c_H100_tp1", manifest["configs"])
            self.assertTrue(manifest["configs"]["model-a_H100_tp1"]["written"])
            self.assertTrue(manifest["configs"]["model-c_H100_tp1"]["written"])
            self.assertFalse(manifest["configs"]["model-b_H100_tp1"]["written"])

            cfg_a = manifest["configs"]["model-a_H100_tp1"]
            self.assertTrue(Path(cfg_a["dataset_npz"]).exists())
            self.assertTrue(Path(cfg_a["split_json"]).exists())
            self.assertTrue(Path(cfg_a["norm_params_json"]).exists())
            self.assertTrue((out_dir / "manifest.json").exists())

            split_payload = json.loads(Path(cfg_a["split_json"]).read_text())
            self.assertGreaterEqual(len(split_payload["train_indices"]), 1)
            self.assertGreaterEqual(len(split_payload["test_indices"]), 1)
            self.assertGreaterEqual(len(split_payload["val_indices"]), 1)

            data = np.load(cfg_a["dataset_npz"], allow_pickle=True)
            x0 = np.asarray(data["x_norm"][0], dtype=np.float32)
            y0 = np.asarray(data["y_norm"][0], dtype=np.float32)
            self.assertEqual(x0.shape[1], 3)
            self.assertEqual(x0.shape[0], y0.shape[0])

            drops = manifest["summary"]["drop_reasons"]
            self.assertGreaterEqual(int(drops.get("missing_request_timestamps", 0)), 1)


if __name__ == "__main__":
    unittest.main()
