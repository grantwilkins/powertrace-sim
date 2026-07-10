"""Build the legacy profiling-run ledger through the shared RunRecord view.

Run from the repository root:
    uv run python feature-test/build_ledger_cache.py --max-per-cell 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.pipeline.artifact_resolution import resolve_throughput  # noqa: E402
from model.training_data.arch import ARCH  # noqa: E402
from model.training_data.ledger_view import reconstruct_bins_from_record  # noqa: E402
from model.training_data.run_record import load_legacy_run  # noqa: E402
from model.utils.io import (  # noqa: E402
    load_json,
    resolve_existing_path,
    write_json,
)

HARDWARE_INDEX = {"A100": 0, "H100": 1}


def resolve_prefill_rate(throughput_db, config_id):
    """Resolve a measured prefill rate or fail with the source config named."""
    throughput = resolve_throughput(throughput_db, config_id)
    value = throughput.get("lambda_prefill")
    if value is None:
        value = throughput.get("prefill_rate_median_toks_per_s")
    if value is None or not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"No positive prefill throughput for {config_id}")
    return float(value)


def run_source_entry(run_id, record):
    provenance = record.provenance
    pair_key = str(provenance["pair_key"])
    return {
        "run_id": int(run_id),
        "source_id": f"{record.config_id}|{pair_key}",
        "config_id": record.config_id,
        "pair_key": pair_key,
        "source_layout": record.source_layout,
        "paths": {
            "power_csv": str(provenance["power_csv_path"]),
            "requests_json": str(provenance["json_path"]),
        },
        "sha256": dict(provenance["sha256"]),
    }


def hardware_index(hardware):
    try:
        return HARDWARE_INDEX[str(hardware)]
    except KeyError as exc:
        raise ValueError(f"Unsupported ledger hardware: {hardware!r}") from exc


def select_successful_runs(runs, max_per_cell, build):
    """Apply per-cell quota to successful parses, not attempted source rows."""
    accepted = []
    failed = 0
    counts = defaultdict(int)
    for run in runs:
        cell = (run["model"], run["hardware"], run["tp"], run["rate"])
        if counts[cell] >= max_per_cell:
            continue
        payload = build(run)
        if payload is None:
            failed += 1
            continue
        counts[cell] += 1
        accepted.append((run, payload))
    return accepted, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    parser.add_argument("--max-per-cell", type=int, default=3)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--out", default="feature-test/ledger_cache.npz")
    parser.add_argument("--throughput-db", default="model/throughput_database.json")
    parser.add_argument("--run-index-out")
    args = parser.parse_args()

    throughput_db = load_json(args.throughput_db)
    base = Path(args.pair_manifest_csv).resolve().parent
    runs = []
    with open(args.pair_manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "").strip() != "matched":
                continue
            model = row["model_name"].strip()
            if model not in ARCH:
                continue
            json_path = resolve_existing_path(row["json_path"].strip(), str(base))
            csv_path = resolve_existing_path(row["power_csv_path"].strip(), str(base))
            if json_path is None or csv_path is None:
                continue
            runs.append(
                {
                    "model": model,
                    "hardware": row["hardware"].strip(),
                    "tp": int(row["tensor_parallelism"]),
                    "rate": float(row["rate"]),
                    "json_path": json_path,
                    "csv_path": csv_path,
                    "pair_key": row.get("pair_key", "").strip(),
                }
            )
    print(f"Selected {len(runs)} runs")

    cols = defaultdict(list)
    rate_by_config = {}
    run_sources = []
    family_names = sorted({value["family"] for value in ARCH.values()})

    def build(run):
        config_id = f"{run['model']}_{run['hardware']}_tp{run['tp']}"
        if config_id not in rate_by_config:
            rate_by_config[config_id] = resolve_prefill_rate(throughput_db, config_id)
        record = load_legacy_run(
            {
                "model_name": run["model"],
                "hardware": run["hardware"],
                "tensor_parallelism": str(run["tp"]),
                "rate": str(run["rate"]),
                "pair_key": run["pair_key"],
                "json_path": run["json_path"],
                "power_csv_path": run["csv_path"],
            }
        )
        bins = None if record is None else reconstruct_bins_from_record(
            record, lambda_prefill=rate_by_config[config_id], dt=args.dt
        )
        return None if bins is None else (record, bins)

    accepted, n_fail = select_successful_runs(runs, args.max_per_cell, build)
    for run_id, (run, (record, bins)) in enumerate(accepted):
        run_sources.append(run_source_entry(run_id, record))
        n = bins["n"]
        for key in (
            "power", "pre_tok", "dec_tok", "batch", "pre_active", "iters",
            "w_read", "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm",
        ):
            cols[key].append(bins[key])
        arch = bins["arch"]
        cols["n_active"].append(np.full(n, arch["n_active"]))
        cols["w_bytes"].append(np.full(n, arch["w_bytes"]))
        cols["fp8"].append(np.full(n, arch["fp8"]))
        cols["tp"].append(np.full(n, float(run["tp"])))
        cols["rate"].append(np.full(n, run["rate"]))
        cols["run_id"].append(np.full(n, run_id, dtype=np.int32))
        cols["model_idx"].append(np.full(n, list(ARCH).index(run["model"]), dtype=np.int32))
        cols["hw_idx"].append(np.full(n, hardware_index(run["hardware"]), dtype=np.int32))
        family = ARCH[run["model"]]["family"]
        cols["family_idx"].append(np.full(n, family_names.index(family), dtype=np.int32))

    if not cols:
        raise ValueError("No profiling runs produced ledger bins")
    out = {key: np.concatenate(values) for key, values in cols.items()}
    out["model_names"] = np.array(list(ARCH))
    out["model_arch_json"] = np.asarray(
        [json.dumps(ARCH[name], sort_keys=True) for name in ARCH]
    )
    out["family_names"] = np.array(family_names)
    out["hw_names"] = np.array(["A100", "H100"])
    out["dt_s"] = np.asarray(float(args.dt))
    np.savez_compressed(args.out, **out)
    run_index_out = args.run_index_out or str(Path(args.out).with_suffix(".runs.json"))
    write_json(
        run_index_out,
        {
            "schema_version": "ledger-run-index-v1",
            "ledger_path": str(args.out),
            "runs": run_sources,
        },
    )
    print(
        f"Parsed {len(accepted)} runs OK ({n_fail} parse failures) -> {out['power'].size} bins "
        f"-> {args.out}; sources -> {run_index_out}"
    )


if __name__ == "__main__":
    main()
