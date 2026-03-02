"""
Parse a single full-day Azure trace CSV into normalized request tuples.

This stage consumes one day file from data/azure_trace/days/ and emits:
  - parsed request CSV with arrival_time, n_in, n_out
  - metadata JSON summary for validation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def parse_timestamp_utc(timestamp_raw: str) -> datetime:
    """
    Parse Azure timestamp string and normalize to UTC.
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


def load_day_requests(csv_path: str, scale_factor: float = 1.0) -> List[Dict[str, object]]:
    """
    Load one day CSV and return sorted parsed request rows.

    Returns rows with:
        request_id, timestamp_utc, arrival_time, n_in, n_out
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Day CSV not found: {csv_path}")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")

    rows: List[Dict[str, object]] = []
    required_cols = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or missing header")
        missing_cols = required_cols - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. Found: {reader.fieldnames}"
            )

        ts_fail_count = 0
        for row_idx, row in enumerate(reader, start=2):
            try:
                dt_utc = parse_timestamp_utc(row["TIMESTAMP"])
            except Exception as exc:
                ts_fail_count += 1
                if ts_fail_count <= 5:
                    print(
                        f"Warning: skipping row {row_idx} due to timestamp parse error: {exc}"
                    )
                continue

            try:
                n_in = int(row["ContextTokens"])
                n_out = int(row["GeneratedTokens"])
            except Exception as exc:
                raise ValueError(
                    f"Error parsing token columns at row {row_idx}: {exc}. Row: {row}"
                ) from exc

            rows.append(
                {
                    "timestamp_dt": dt_utc,
                    "n_in": n_in,
                    "n_out": n_out,
                }
            )

    if not rows:
        raise ValueError(f"No valid rows found in day CSV: {csv_path}")

    rows.sort(key=lambda r: r["timestamp_dt"])
    t0 = rows[0]["timestamp_dt"]

    parsed: List[Dict[str, object]] = []
    for request_id, row in enumerate(rows):
        arrival_time = (row["timestamp_dt"] - t0).total_seconds() / scale_factor
        parsed.append(
            {
                "request_id": request_id,
                "timestamp_utc": row["timestamp_dt"].isoformat(sep=" "),
                "arrival_time": float(arrival_time),
                "n_in": int(row["n_in"]),
                "n_out": int(row["n_out"]),
            }
        )

    return parsed


def compute_day_metadata(parsed_rows: List[Dict[str, object]]) -> Dict[str, object]:
    """
    Compute validation metadata for parsed day rows.
    """
    if not parsed_rows:
        return {
            "num_requests": 0,
            "start_timestamp_utc": "",
            "end_timestamp_utc": "",
            "span_seconds": 0.0,
            "avg_rate_req_per_s": 0.0,
            "avg_input_tokens": 0.0,
            "avg_output_tokens": 0.0,
            "hours_present": 0,
            "is_full_day": False,
        }

    dt_values = [
        datetime.fromisoformat(str(row["timestamp_utc"])).astimezone(timezone.utc)
        for row in parsed_rows
    ]
    start_dt = dt_values[0]
    end_dt = dt_values[-1]
    span_seconds = (end_dt - start_dt).total_seconds()
    num_requests = len(parsed_rows)
    avg_rate = (num_requests / span_seconds) if span_seconds > 0 else 0.0
    hours = {dt.hour for dt in dt_values}

    is_full_day = (len(hours) == 24) and (0 in hours) and (23 in hours) and (span_seconds >= 86399.0)

    return {
        "num_requests": num_requests,
        "start_timestamp_utc": start_dt.isoformat(sep=" "),
        "end_timestamp_utc": end_dt.isoformat(sep=" "),
        "span_seconds": float(span_seconds),
        "avg_rate_req_per_s": float(avg_rate),
        "avg_input_tokens": float(sum(int(r["n_in"]) for r in parsed_rows) / num_requests),
        "avg_output_tokens": float(sum(int(r["n_out"]) for r in parsed_rows) / num_requests),
        "hours_present": int(len(hours)),
        "is_full_day": bool(is_full_day),
    }


def save_parsed_requests_csv(parsed_rows: List[Dict[str, object]], output_path: str) -> None:
    """
    Save parsed request rows to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["request_id", "timestamp_utc", "arrival_time", "n_in", "n_out"]
        )
        writer.writeheader()
        writer.writerows(parsed_rows)


def save_metadata_json(metadata: Dict[str, object], output_path: str) -> None:
    """
    Save metadata JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def build_default_paths() -> Dict[str, str]:
    """
    Build default input/output paths from repository root.
    """
    repo_root = Path(__file__).resolve().parents[2]
    days_dir = repo_root / "data" / "azure_trace" / "days"
    parsed_dir = repo_root / "data" / "azure_trace" / "parsed"
    default_day = "2024-05-16"
    return {
        "input_csv": str(days_dir / f"{default_day}.csv"),
        "output_dir": str(parsed_dir),
        "default_day": default_day,
    }


def main() -> None:
    """
    Main execution for Experiment 0b:
      one day CSV -> parsed request CSV + metadata JSON.
    """
    defaults = build_default_paths()

    parser = argparse.ArgumentParser(description="Parse a single Azure day CSV into request tuples.")
    parser.add_argument(
        "--input-csv",
        default=defaults["input_csv"],
        help="Path to one day CSV (default: data/azure_trace/days/2024-05-16.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default=defaults["output_dir"],
        help="Directory for parsed outputs (default: data/azure_trace/parsed/)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Optional arrival-rate scale (default: 1.0)",
    )
    args = parser.parse_args()

    day_stem = Path(args.input_csv).stem
    requests_csv_path = os.path.join(args.output_dir, f"day_{day_stem}_requests.csv")
    metadata_json_path = os.path.join(args.output_dir, f"day_{day_stem}_metadata.json")

    print("=" * 70)
    print("Azure Day Parser (Experiment 0b)")
    print("=" * 70)
    print(f"Input day CSV: {args.input_csv}")
    print(f"Output CSV   : {requests_csv_path}")
    print(f"Output JSON  : {metadata_json_path}")
    print(f"Scale factor : {args.scale_factor}")

    parsed_rows = load_day_requests(csv_path=args.input_csv, scale_factor=args.scale_factor)
    metadata = compute_day_metadata(parsed_rows)

    save_parsed_requests_csv(parsed_rows, requests_csv_path)
    save_metadata_json(metadata, metadata_json_path)

    print("\nParsed summary:")
    print(f"  Requests        : {metadata['num_requests']:,}")
    print(f"  Start (UTC)     : {metadata['start_timestamp_utc']}")
    print(f"  End (UTC)       : {metadata['end_timestamp_utc']}")
    print(f"  Span (s)        : {metadata['span_seconds']:.6f}")
    print(f"  Avg rate (req/s): {metadata['avg_rate_req_per_s']:.2f}")
    print(f"  Hours present   : {metadata['hours_present']}")
    print(f"  Full day        : {metadata['is_full_day']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
