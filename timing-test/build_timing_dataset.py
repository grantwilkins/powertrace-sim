"""Build the per-request timing dataset and frozen split manifest (timing-test).

Run from the repository root:
    uv run python timing-test/build_timing_dataset.py

Inputs: the legacy 450-run serving set, selected exactly like
feature-test/build_ledger_cache.py (pair_manifest status=matched, model in
ARCH, max 3 successfully parsed runs per (model, hardware, tp, rate) cell),
ingested through model.training_data.run_record.load_legacy_run. Only
requests with exact inter-token latencies are kept (len(itls) ==
output_tokens - 1, the exact_itl_mask convention); exclusions are counted
per run.

Outputs:
- timing-test/timing_dataset.npz — flat arrays:
    request-level (length n_requests):
      req_run_id        int32   index into the run-level arrays
      arrival_time_s    float64 arrival relative to run start (first
                                validated request timestamp of the run)
      input_tokens      int64   prompt tokens
      output_tokens     int64   completion tokens
      ttft_s            float64 time to first token
      decode_duration_s float64 recorded decode time (sum of ITLs domain)
      itl_offsets       int64   (n_requests + 1,) — request i's inter-token
                                latencies are itl_values[itl_offsets[i]:itl_offsets[i+1]]
      itl_values        float64 all ITLs concatenated in request order
    run-level (length n_runs, indexed by run_id):
      run_model, run_hardware, run_source_id, run_pair_key   (str)
      run_tp, run_repeat                                      int64
      run_rate                                                float64
      run_requests_source, run_requests_exact,
      run_requests_excluded                                   int64
- timing-test/timing_dataset.runs.json — per-run provenance with sha256 of
  the source power CSV and request JSON.
- timing-test/split_manifest.json — one role per run_id implementing
  DESIGN.md section 3 exactly, plus the frozen section 5 targets.

Repeat assignment: within each (config_id, rate) group, runs are sorted by
source_id and enumerated 0, 1, 2 — the same rule as
feature-test/evaluate_candidates.py:assign_repeats (re-stated here because
feature-test is not an importable package).
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.training_data.arch import ARCH  # noqa: E402
from model.training_data.ledger_view import exact_itl_mask  # noqa: E402
from model.training_data.run_record import load_legacy_run  # noqa: E402
from model.utils.io import resolve_existing_path, write_json  # noqa: E402

# DESIGN.md section 3 (frozen 2026-07-16).
HOLDOUT_MODEL = {"A100": "gpt-oss-120b", "H100": "llama-3-405b"}
HOLDOUT_TWIN = {"A100": "deepseek-r1-distill-70b", "H100": "deepseek-r1-distill-8b"}
HOLDOUT_RATE = 4.0
TRAIN_RATES = (0.125, 0.25, 0.5, 1.0, 2.0)
TRAIN_REPEATS = (0, 1)
TEST_REPEAT = 2
ROLES = ("train", "test_indomain", "holdout_model", "holdout_twin", "holdout_rate")

# DESIGN.md section 5, recorded verbatim so scoring cannot renegotiate them.
PROVISIONAL_TARGETS = {
    "source": "timing-test/DESIGN.md section 5 (frozen 2026-07-16, before fitting)",
    "end_to_end": {
        "median_abs_error_pct_max": 10.0,
        "median_abs_error_s_max": 1.0,
        "applies_to": "rates <= 2",
    },
    "ttft": {
        "median_abs_error_s_max_below_saturation": 0.3,
        "median_abs_error_pct_max_at_holdout_rate_4": 20.0,
    },
    "decode_duration": {"median_abs_error_pct_max": 10.0},
    "sign_bias": {"max_systematic_p50_bias_pct": 15.0},
}


def assign_repeats(run_rows):
    """Repeat within (config_id, rate) by sorted stable source identity.

    Same rule as feature-test/evaluate_candidates.py:assign_repeats.
    """
    groups = defaultdict(list)
    for row in run_rows:
        groups[(row["config_id"], float(row["rate"]))].append(row)
    return {int(row["run_id"]): repeat for rows in groups.values()
            for repeat, row in enumerate(sorted(rows, key=lambda x: x["source_id"]))}


def assign_role(model, hardware, rate, repeat):
    """DESIGN.md section 3: exactly one role per run; raises on anything else."""
    if hardware not in HOLDOUT_MODEL:
        raise ValueError(f"Hardware {hardware!r} has no holdout contract")
    if model == HOLDOUT_MODEL[hardware]:
        return "holdout_model"
    if model == HOLDOUT_TWIN[hardware]:
        return "holdout_twin"
    if float(rate) == HOLDOUT_RATE:
        return "holdout_rate"
    if float(rate) not in TRAIN_RATES:
        raise ValueError(f"Rate {rate!r} is outside the frozen contract")
    if int(repeat) in TRAIN_REPEATS:
        return "train"
    if int(repeat) == TEST_REPEAT:
        return "test_indomain"
    raise ValueError(f"Repeat {repeat!r} is outside the frozen contract")


def check_split(roles_by_run, run_ids):
    """Assert coverage (every run has a role) and disjointness (exactly one)."""
    missing = sorted(set(run_ids) - set(roles_by_run))
    extra = sorted(set(roles_by_run) - set(run_ids))
    if missing or extra:
        raise ValueError(f"Split does not cover the run table: missing={missing}, extra={extra}")
    bad = {run: role for run, role in roles_by_run.items() if role not in ROLES}
    if bad:
        raise ValueError(f"Unknown roles: {bad}")


def select_manifest_rows(pair_manifest_csv):
    """Mirror feature-test/build_ledger_cache.py row selection."""
    base = Path(pair_manifest_csv).resolve().parent
    rows = []
    with open(pair_manifest_csv, newline="") as f:
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
            rows.append({
                "model": model,
                "hardware": row["hardware"].strip(),
                "tp": int(row["tensor_parallelism"]),
                "rate": float(row["rate"]),
                "json_path": json_path,
                "csv_path": csv_path,
                "pair_key": row.get("pair_key", "").strip(),
            })
    return rows


def pack_itls(itls):
    """Ragged ITL lists -> (concatenated values, per-request offsets).

    Request i round-trips as values[offsets[i]:offsets[i + 1]].
    """
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1) for v in itls]
    offsets = np.concatenate(
        [[0], np.cumsum(np.asarray([a.size for a in arrays], dtype=np.int64))])
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)
    return values, offsets


def extract_requests(record):
    """Per-request timing rows for the exact-ITL subset of one run.

    Arrival times are relative to the run start (first validated request
    timestamp), so exclusions never shift the time origin.
    """
    exact = exact_itl_mask(record.output_lens, record.itls).astype(bool)
    t0 = float(np.min(record.request_timestamps))
    itls = [np.asarray(v, dtype=np.float64).reshape(-1)
            for v, keep in zip(record.itls, exact) if keep]
    out_tokens = record.output_lens[exact].astype(np.int64)
    for tokens, intervals in zip(out_tokens, itls):
        if intervals.size != max(int(tokens) - 1, 0):
            raise AssertionError("exact_itl_mask violated its own contract")
    return {
        "arrival_time_s": record.request_timestamps[exact] - t0,
        "input_tokens": record.input_lens[exact].astype(np.int64),
        "output_tokens": out_tokens,
        "ttft_s": record.ttfts[exact],
        "decode_duration_s": record.decode_times[exact],
        "itls": itls,
        "n_source": int(exact.size),
        "n_exact": int(exact.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    parser.add_argument("--max-per-cell", type=int, default=3)
    parser.add_argument("--out", default="timing-test/timing_dataset.npz")
    parser.add_argument("--split-out", default="timing-test/split_manifest.json")
    args = parser.parse_args()

    source_rows = select_manifest_rows(args.pair_manifest_csv)
    print(f"Selected {len(source_rows)} manifest rows")

    accepted, n_fail = [], 0
    counts = defaultdict(int)
    for row in source_rows:
        cell = (row["model"], row["hardware"], row["tp"], row["rate"])
        if counts[cell] >= args.max_per_cell:
            continue
        record = load_legacy_run({
            "model_name": row["model"], "hardware": row["hardware"],
            "tensor_parallelism": str(row["tp"]), "rate": str(row["rate"]),
            "pair_key": row["pair_key"], "json_path": row["json_path"],
            "power_csv_path": row["csv_path"],
        })
        if record is None:
            n_fail += 1
            continue
        counts[cell] += 1
        accepted.append((row, record))

    run_rows, run_index, req_cols, itl_chunks = [], [], defaultdict(list), []
    for run_id, (row, record) in enumerate(accepted):
        requests = extract_requests(record)
        source_id = f"{record.config_id}|{row['pair_key']}"
        run_rows.append({
            "run_id": run_id, "config_id": record.config_id,
            "source_id": source_id, "model": row["model"],
            "hardware": row["hardware"], "tp": row["tp"], "rate": row["rate"],
            "pair_key": row["pair_key"], "n_source": requests["n_source"],
            "n_exact": requests["n_exact"],
        })
        run_index.append({
            "run_id": run_id, "source_id": source_id,
            "config_id": record.config_id, "pair_key": row["pair_key"],
            "paths": {"power_csv": row["csv_path"], "requests_json": row["json_path"]},
            "sha256": dict(record.provenance["sha256"]),
            "requests_source": requests["n_source"],
            "requests_exact_itl": requests["n_exact"],
            "requests_excluded": requests["n_source"] - requests["n_exact"],
        })
        n = requests["n_exact"]
        req_cols["req_run_id"].append(np.full(n, run_id, dtype=np.int32))
        for key in ("arrival_time_s", "input_tokens", "output_tokens",
                    "ttft_s", "decode_duration_s"):
            req_cols[key].append(requests[key])
        itl_chunks.extend(requests["itls"])

    if not run_rows:
        raise ValueError("No runs parsed")
    repeats = assign_repeats(run_rows)
    roles = {}
    for run in run_rows:
        run["repeat"] = repeats[run["run_id"]]
        roles[run["run_id"]] = assign_role(
            run["model"], run["hardware"], run["rate"], run["repeat"])
    check_split(roles, [run["run_id"] for run in run_rows])

    out = {key: np.concatenate(values) for key, values in req_cols.items()}
    out["itl_values"], out["itl_offsets"] = pack_itls(itl_chunks)
    for key, field, dtype in (
        ("run_model", "model", None), ("run_hardware", "hardware", None),
        ("run_source_id", "source_id", None), ("run_pair_key", "pair_key", None),
        ("run_tp", "tp", np.int64), ("run_repeat", "repeat", np.int64),
        ("run_rate", "rate", np.float64),
        ("run_requests_source", "n_source", np.int64),
        ("run_requests_exact", "n_exact", np.int64),
    ):
        values = [run[field] for run in run_rows]
        out[key] = np.asarray(values) if dtype is None else np.asarray(values, dtype=dtype)
    out["run_requests_excluded"] = out["run_requests_source"] - out["run_requests_exact"]
    if int(out["itl_offsets"][-1]) != int(out["itl_values"].size):
        raise AssertionError("ITL offsets do not close over the value array")
    np.savez_compressed(args.out, **out)
    write_json(str(Path(args.out).with_suffix("")) + ".runs.json", {
        "schema_version": "timing-run-index-v1",
        "dataset_path": args.out,
        "pair_manifest_csv": args.pair_manifest_csv,
        "runs": run_index,
    })

    role_runs = {role: sorted(r for r, v in roles.items() if v == role) for role in ROLES}
    write_json(args.split_out, {
        "schema_version": "timing-split-manifest-v1",
        "design": "timing-test/DESIGN.md section 3 (frozen 2026-07-16)",
        "holdout_model": HOLDOUT_MODEL,
        "holdout_twin": HOLDOUT_TWIN,
        "holdout_rate": HOLDOUT_RATE,
        "train_rates": list(TRAIN_RATES),
        "train_repeats": list(TRAIN_REPEATS),
        "test_repeat": TEST_REPEAT,
        "provisional_targets": PROVISIONAL_TARGETS,
        "roles": {str(run_id): roles[run_id] for run_id in sorted(roles)},
        "role_runs": role_runs,
    })

    n_req = int(out["ttft_s"].size)
    n_src = int(out["run_requests_source"].sum())
    print(f"Parsed {len(run_rows)} runs OK ({n_fail} parse failures)")
    print(f"Requests: {n_req} exact-ITL of {n_src} source "
          f"({100.0 * n_req / n_src:.2f}% retained) -> {args.out}")
    run_role = np.asarray([roles[run["run_id"]] for run in run_rows])
    run_hw = out["run_hardware"]
    for role in ROLES:
        for hw in sorted(HOLDOUT_MODEL):
            selected = np.flatnonzero((run_role == role) & (run_hw == hw))
            reqs = int(np.isin(out["req_run_id"], selected).sum())
            print(f"  {role:15s} {hw}: {selected.size:3d} runs, {reqs:6d} requests")
    zero = [run["source_id"] for run in run_rows if run["n_exact"] == 0]
    if zero:
        print(f"Runs with zero exact requests: {zero}")
    short = {cell: n for cell, n in sorted(counts.items()) if n < args.max_per_cell}
    if short:
        print(f"Cells with fewer than {args.max_per_cell} runs: {short}")


if __name__ == "__main__":
    main()
