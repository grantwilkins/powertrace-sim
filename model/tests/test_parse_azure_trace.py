"""
Tests for scripts/eval/parse_azure_trace.py day parsing flow.
"""

import csv
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from parse_azure_trace import (  # noqa: E402
    compute_day_metadata,
    load_day_requests,
    parse_timestamp_utc,
    save_metadata_json,
    save_parsed_requests_csv,
)


def _write_day_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        writer.writeheader()
        writer.writerows(rows)


class TestParseTimestampUTC:
    def test_timezone_normalization(self):
        dt = parse_timestamp_utc("2024-05-16 01:30:00+02:00")
        assert dt.isoformat() == "2024-05-15T23:30:00+00:00"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty timestamp"):
            parse_timestamp_utc("")


class TestLoadDayRequests:
    def test_sorted_output_and_arrival_time(self):
        rows = [
            {
                "TIMESTAMP": "2024-05-16 00:00:05+00:00",
                "ContextTokens": "200",
                "GeneratedTokens": "20",
            },
            {
                "TIMESTAMP": "2024-05-16 00:00:00+00:00",
                "ContextTokens": "100",
                "GeneratedTokens": "10",
            },
            {
                "TIMESTAMP": "2024-05-16 00:00:03+00:00",
                "ContextTokens": "150",
                "GeneratedTokens": "15",
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "2024-05-16.csv")
            _write_day_csv(csv_path, rows)

            parsed = load_day_requests(csv_path, scale_factor=1.0)
            assert len(parsed) == 3
            assert [r["request_id"] for r in parsed] == [0, 1, 2]
            assert [r["arrival_time"] for r in parsed] == pytest.approx([0.0, 3.0, 5.0])
            assert [r["n_in"] for r in parsed] == [100, 150, 200]

    def test_scale_factor(self):
        rows = [
            {
                "TIMESTAMP": "2024-05-16 00:00:00+00:00",
                "ContextTokens": "100",
                "GeneratedTokens": "10",
            },
            {
                "TIMESTAMP": "2024-05-16 00:00:10+00:00",
                "ContextTokens": "200",
                "GeneratedTokens": "20",
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "2024-05-16.csv")
            _write_day_csv(csv_path, rows)
            parsed = load_day_requests(csv_path, scale_factor=2.0)
            assert [r["arrival_time"] for r in parsed] == pytest.approx([0.0, 5.0])

    def test_missing_columns_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            writer = csv.DictWriter(f, fieldnames=["TIMESTAMP", "ContextTokens"])
            writer.writeheader()
            writer.writerow({"TIMESTAMP": "2024-05-16 00:00:00+00:00", "ContextTokens": "100"})
            bad_path = f.name
        try:
            with pytest.raises(ValueError, match="CSV missing required columns"):
                load_day_requests(bad_path)
        finally:
            os.unlink(bad_path)


class TestMetadataAndSave:
    def test_full_day_metadata_detection(self):
        rows = []
        for hour in range(24):
            rows.append(
                {
                    "TIMESTAMP": f"2024-05-16 {hour:02d}:00:00+00:00",
                    "ContextTokens": "100",
                    "GeneratedTokens": "10",
                }
            )
        rows.append(
            {
                "TIMESTAMP": "2024-05-16 23:59:59+00:00",
                "ContextTokens": "200",
                "GeneratedTokens": "20",
            }
        )

        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "2024-05-16.csv")
            _write_day_csv(csv_path, rows)
            parsed = load_day_requests(csv_path)
            meta = compute_day_metadata(parsed)

            assert meta["num_requests"] == 25
            assert meta["hours_present"] == 24
            assert meta["is_full_day"] is True
            assert meta["start_timestamp_utc"].startswith("2024-05-16 00:00:00")
            assert meta["end_timestamp_utc"].startswith("2024-05-16 23:59:59")
            assert meta["span_seconds"] >= 86399.0

    def test_save_outputs(self):
        parsed_rows = [
            {
                "request_id": 0,
                "timestamp_utc": "2024-05-16 00:00:00+00:00",
                "arrival_time": 0.0,
                "n_in": 100,
                "n_out": 10,
            },
            {
                "request_id": 1,
                "timestamp_utc": "2024-05-16 00:00:01+00:00",
                "arrival_time": 1.0,
                "n_in": 120,
                "n_out": 12,
            },
        ]
        metadata = compute_day_metadata(parsed_rows)

        with tempfile.TemporaryDirectory() as td:
            requests_path = os.path.join(td, "parsed.csv")
            metadata_path = os.path.join(td, "metadata.json")
            save_parsed_requests_csv(parsed_rows, requests_path)
            save_metadata_json(metadata, metadata_path)

            assert os.path.exists(requests_path)
            assert os.path.exists(metadata_path)

            with open(requests_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
                assert len(rows) == 2
                assert rows[0]["request_id"] == "0"
                assert rows[1]["n_out"] == "12"

            with open(metadata_path, "r") as f:
                payload = json.load(f)
                assert payload["num_requests"] == 2
