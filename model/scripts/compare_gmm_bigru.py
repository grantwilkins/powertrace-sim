#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from model.utils.io import ensure_dir


def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def _nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _parse_inputs(inputs: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for token in inputs:
        text = str(token).strip()
        if text == "":
            continue
        if "=" not in text:
            raise ValueError(
                f"Invalid --input '{text}'. Expected format '<label>=<path_to_config_summary.csv>'."
            )
        label, path = text.split("=", 1)
        label = label.strip()
        path = path.strip()
        if label == "" or path == "":
            raise ValueError(f"Invalid --input '{text}'. Label and path must be non-empty.")
        resolved = str(Path(path))
        if not Path(resolved).exists():
            raise ValueError(f"Input CSV not found: {resolved}")
        parsed.append((label, resolved))
    if len(parsed) == 0:
        raise ValueError("At least one --input must be provided.")
    return parsed


def build_comparison_tables(
    *,
    inputs: Sequence[Tuple[str, str]],
    out_csv: str,
    aggregate_csv: str,
) -> Dict[str, object]:
    detail_rows: List[Dict[str, object]] = []
    for label, path in inputs:
        rows = _load_csv_rows(path)
        for row in rows:
            detail_rows.append(
                {
                    "run_label": label,
                    "source_csv": path,
                    "config_id": str(row.get("config_id", "")),
                    "status": str(row.get("status", "")),
                    "reason": str(row.get("reason", "")),
                    "k": str(row.get("k", row.get("n_mix", ""))),
                    "feature_set": str(row.get("feature_set", row.get("model_kind", ""))),
                    "decode_mode": str(row.get("decode_mode", "")),
                    "median_filter_window": str(row.get("median_filter_window", "")),
                    "gmm_covariance_type": str(row.get("gmm_covariance_type", "")),
                    "ks_stat_median": _to_float(row.get("ks_stat_median", "")),
                    "acf_r2_median": _to_float(row.get("acf_r2_median", "")),
                    "nrmse_median": _to_float(row.get("nrmse_median", "")),
                    "p95_error_pct_median": _to_float(row.get("p95_error_pct_median", "")),
                    "p99_error_pct_median": _to_float(row.get("p99_error_pct_median", "")),
                    "delta_energy_pct_median": _to_float(row.get("delta_energy_pct_median", "")),
                }
            )

    detail_fields = [
        "run_label",
        "source_csv",
        "config_id",
        "status",
        "reason",
        "k",
        "feature_set",
        "decode_mode",
        "median_filter_window",
        "gmm_covariance_type",
        "ks_stat_median",
        "acf_r2_median",
        "nrmse_median",
        "p95_error_pct_median",
        "p99_error_pct_median",
        "delta_energy_pct_median",
    ]
    _write_csv(out_csv, detail_rows, detail_fields)

    aggregate_rows: List[Dict[str, object]] = []
    by_label: Dict[str, List[Dict[str, object]]] = {}
    for row in detail_rows:
        by_label.setdefault(str(row["run_label"]), []).append(row)

    for label, rows in sorted(by_label.items()):
        evaluated = [r for r in rows if str(r.get("status", "")) == "evaluated"]
        ref = evaluated if len(evaluated) > 0 else rows
        aggregate_rows.append(
            {
                "run_label": label,
                "num_rows": int(len(rows)),
                "num_evaluated_rows": int(len(evaluated)),
                "ks_stat_median": _nanmedian(r["ks_stat_median"] for r in ref),
                "acf_r2_median": _nanmedian(r["acf_r2_median"] for r in ref),
                "nrmse_median": _nanmedian(r["nrmse_median"] for r in ref),
                "p95_error_pct_median": _nanmedian(r["p95_error_pct_median"] for r in ref),
                "p99_error_pct_median": _nanmedian(r["p99_error_pct_median"] for r in ref),
                "delta_energy_pct_median": _nanmedian(r["delta_energy_pct_median"] for r in ref),
            }
        )

    aggregate_fields = [
        "run_label",
        "num_rows",
        "num_evaluated_rows",
        "ks_stat_median",
        "acf_r2_median",
        "nrmse_median",
        "p95_error_pct_median",
        "p99_error_pct_median",
        "delta_energy_pct_median",
    ]
    _write_csv(aggregate_csv, aggregate_rows, aggregate_fields)

    return {
        "num_inputs": int(len(inputs)),
        "num_detail_rows": int(len(detail_rows)),
        "num_aggregate_rows": int(len(aggregate_rows)),
        "out_csv": out_csv,
        "aggregate_csv": aggregate_csv,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare config summary CSVs across model families/runs.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Input in format '<label>=<path_to_config_summary.csv>'. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--out-csv",
        default="results/continuous_v1_gmm_bigru/comparison/combined_config_summary.csv",
    )
    parser.add_argument(
        "--aggregate-csv",
        default="results/continuous_v1_gmm_bigru/comparison/aggregate_summary.csv",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    inputs = _parse_inputs(args.input)
    run = build_comparison_tables(
        inputs=inputs,
        out_csv=args.out_csv,
        aggregate_csv=args.aggregate_csv,
    )
    print("[compare_gmm_bigru] Done")
    print(f"  num_inputs      : {run['num_inputs']}")
    print(f"  num_detail_rows : {run['num_detail_rows']}")
    print(f"  num_aggregate   : {run['num_aggregate_rows']}")
    print(f"  out_csv         : {run['out_csv']}")
    print(f"  aggregate_csv   : {run['aggregate_csv']}")


if __name__ == "__main__":
    main()
