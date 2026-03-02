"""
Tests for scripts/eval/split_azure_week_to_days.py.
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts/eval"))

from split_azure_week_to_days import split_week_csv_to_days  # noqa: E402


def _write_week_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        writer.writeheader()
        writer.writerows(rows)


def test_split_and_manifest():
    rows = [
        {
            "TIMESTAMP": "2024-05-10 00:00:00+00:00",
            "ContextTokens": "100",
            "GeneratedTokens": "10",
        },
        {
            "TIMESTAMP": "2024-05-10 12:00:00+00:00",
            "ContextTokens": "110",
            "GeneratedTokens": "11",
        },
        {
            "TIMESTAMP": "2024-05-11 00:00:00+00:00",
            "ContextTokens": "120",
            "GeneratedTokens": "12",
        },
    ]

    with tempfile.TemporaryDirectory() as td:
        week_csv = os.path.join(td, "week.csv")
        out_dir = os.path.join(td, "days")
        _write_week_csv(week_csv, rows)

        manifest_rows = split_week_csv_to_days(week_csv, out_dir)
        assert len(manifest_rows) == 2

        day1 = os.path.join(out_dir, "2024-05-10.csv")
        day2 = os.path.join(out_dir, "2024-05-11.csv")
        manifest_path = os.path.join(out_dir, "day_manifest.csv")
        assert os.path.exists(day1)
        assert os.path.exists(day2)
        assert os.path.exists(manifest_path)

        with open(day1, "r", newline="") as f:
            day1_rows = list(csv.DictReader(f))
            assert len(day1_rows) == 2
            assert all(r["TIMESTAMP"].startswith("2024-05-10") for r in day1_rows)

        with open(day2, "r", newline="") as f:
            day2_rows = list(csv.DictReader(f))
            assert len(day2_rows) == 1
            assert day2_rows[0]["TIMESTAMP"].startswith("2024-05-11")

        with open(manifest_path, "r", newline="") as f:
            manifest = {row["day_utc"]: row for row in csv.DictReader(f)}
            assert set(manifest.keys()) == {"2024-05-10", "2024-05-11"}
            assert int(manifest["2024-05-10"]["row_count"]) == 2
            assert int(manifest["2024-05-11"]["row_count"]) == 1


def test_manifest_full_day_flag():
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
        week_csv = os.path.join(td, "week.csv")
        out_dir = os.path.join(td, "days")
        _write_week_csv(week_csv, rows)

        split_week_csv_to_days(week_csv, out_dir)
        manifest_path = os.path.join(out_dir, "day_manifest.csv")

        with open(manifest_path, "r", newline="") as f:
            manifest = list(csv.DictReader(f))
            assert len(manifest) == 1
            row = manifest[0]
            assert row["day_utc"] == "2024-05-16"
            assert int(row["row_count"]) == 25
            assert int(row["hours_present"]) == 24
            assert row["is_full_day"] == "True"
