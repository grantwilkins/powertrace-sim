"""
Split Azure 1-week trace CSV into per-day CSV files.

Outputs:
  - data/azure_trace/days/YYYY-MM-DD.csv
  - data/azure_trace/days/day_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, IO, List, Set, Tuple


DEFAULT_INPUT_CSV = "/Users/grantwilkins/Downloads/AzureLLMInferenceTrace_code_1week.csv"


def parse_timestamp_utc(timestamp_raw: str) -> datetime:
    """
    Parse timestamp string and normalize to UTC.
    """
    ts = str(timestamp_raw).strip()
    if not ts:
        raise ValueError("empty timestamp")

    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # Fallback for minor format variants.
        if "." in ts:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f%z")
        else:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S%z")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _default_output_dir() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "data" / "azure_trace" / "days")


def split_week_csv_to_days(input_csv: str, output_dir: str) -> List[Dict[str, object]]:
    """
    Stream the week CSV and split rows into one output CSV per UTC day.

    Manifest columns:
      day_utc,row_count,min_timestamp,max_timestamp,span_seconds,is_full_day,hours_present
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Week CSV not found: {input_csv}")

    os.makedirs(output_dir, exist_ok=True)

    required_cols = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
    day_writers: Dict[str, Tuple[IO[str], csv.DictWriter]] = {}
    day_stats: Dict[str, Dict[str, object]] = {}

    def get_day_writer(day_key: str, fieldnames: List[str]) -> csv.DictWriter:
        if day_key in day_writers:
            return day_writers[day_key][1]
        day_path = os.path.join(output_dir, f"{day_key}.csv")
        f = open(day_path, "w", newline="")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        day_writers[day_key] = (f, writer)
        day_stats[day_key] = {
            "row_count": 0,
            "min_dt": None,
            "max_dt": None,
            "hours": set(),
        }
        return writer

    failed_ts = 0
    try:
        with open(input_csv, "r", newline="") as f_in:
            reader = csv.DictReader(f_in)
            if reader.fieldnames is None:
                raise ValueError("CSV is empty or missing headers")
            missing_cols = required_cols - set(reader.fieldnames)
            if missing_cols:
                raise ValueError(
                    f"CSV missing required columns: {missing_cols}. Found: {reader.fieldnames}"
                )

            fieldnames = list(reader.fieldnames)
            for row_idx, row in enumerate(reader, start=2):
                try:
                    dt_utc = parse_timestamp_utc(row["TIMESTAMP"])
                except Exception as exc:
                    failed_ts += 1
                    if failed_ts <= 5:
                        print(
                            f"Warning: skipping row {row_idx} due to timestamp parse error: {exc}"
                        )
                    continue

                day_key = dt_utc.strftime("%Y-%m-%d")
                writer = get_day_writer(day_key, fieldnames)
                writer.writerow({k: row.get(k, "") for k in fieldnames})

                stats = day_stats[day_key]
                stats["row_count"] = int(stats["row_count"]) + 1
                stats["hours"].add(dt_utc.hour)  # type: ignore[union-attr]
                current_min = stats["min_dt"]
                current_max = stats["max_dt"]
                if (current_min is None) or (dt_utc < current_min):
                    stats["min_dt"] = dt_utc
                if (current_max is None) or (dt_utc > current_max):
                    stats["max_dt"] = dt_utc
    finally:
        for f_handle, _ in day_writers.values():
            f_handle.close()

    manifest_rows: List[Dict[str, object]] = []
    for day_key in sorted(day_stats):
        stats = day_stats[day_key]
        min_dt = stats["min_dt"]
        max_dt = stats["max_dt"]
        hours: Set[int] = stats["hours"]  # type: ignore[assignment]

        if min_dt is None or max_dt is None:
            span_seconds = 0.0
            min_ts = ""
            max_ts = ""
        else:
            span_seconds = (max_dt - min_dt).total_seconds()
            min_ts = min_dt.isoformat(sep=" ")
            max_ts = max_dt.isoformat(sep=" ")

        is_full_day = (
            len(hours) == 24 and 0 in hours and 23 in hours and span_seconds >= 86399.0
        )
        manifest_rows.append(
            {
                "day_utc": day_key,
                "row_count": int(stats["row_count"]),
                "min_timestamp": min_ts,
                "max_timestamp": max_ts,
                "span_seconds": float(span_seconds),
                "is_full_day": bool(is_full_day),
                "hours_present": int(len(hours)),
            }
        )

    manifest_path = os.path.join(output_dir, "day_manifest.csv")
    with open(manifest_path, "w", newline="") as f_manifest:
        writer = csv.DictWriter(
            f_manifest,
            fieldnames=[
                "day_utc",
                "row_count",
                "min_timestamp",
                "max_timestamp",
                "span_seconds",
                "is_full_day",
                "hours_present",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("=" * 70)
    print("Azure Week -> Day Split")
    print("=" * 70)
    print(f"Input CSV     : {input_csv}")
    print(f"Output dir    : {output_dir}")
    print(f"Days emitted  : {len(manifest_rows)}")
    if failed_ts > 0:
        print(f"Skipped rows  : {failed_ts} (timestamp parse failures)")
    print(f"Manifest path : {manifest_path}")
    print("=" * 70)

    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Azure week CSV into per-day CSV files.")
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help=f"Path to 1-week Azure CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help="Output directory for per-day files (default: data/azure_trace/days/)",
    )
    args = parser.parse_args()

    split_week_csv_to_days(input_csv=args.input_csv, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
