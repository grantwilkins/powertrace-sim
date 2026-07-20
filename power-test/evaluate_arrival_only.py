"""Freeze the power artifact and score every timing holdout without refitting.

The joined cache was produced entirely from request marks through the timing
simulator; this script loads the already-fitted dense hardware surfaces and
scores test_indomain, holdout_rate, holdout_twin, holdout_model, and
dtype_calibration runs.  It also attaches the closest pre-existing v2 M4A and
same-configuration B2 rows by exact source identity as contextual references.

Usage: uv run python power-test/evaluate_arrival_only.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO_ROOT / "feature-test"))

from evaluation_core import trace_metrics  # noqa: E402
from fit_power_surface import (  # noqa: E402
    PRIMARY_METRICS,
    coefficients_in_design_order,
    interpolate_nan,
    role_table,
    run_slices,
)
from power_surface import predict, surface_design  # noqa: E402
from response_chain import apply_chain  # noqa: E402

CACHE = BASE / "sim_ledger_power_250ms.npz"
ARTIFACT = BASE / "fitted_surface.json"
RUN_INDEX = REPO_ROOT / "timing-test" / "timing_dataset.runs.json"
V2_METRICS = REPO_ROOT / "results" / "feature_test_v2" / "per_run_metrics.csv"
OUT_JSON = BASE / "arrival_only_report.json"
OUT_CSV = BASE / "arrival_only_per_run.csv"
SCORED_ROLES = (
    "test_indomain",
    "holdout_rate",
    "holdout_twin",
    "holdout_model",
    "dtype_calibration",
)
ROW_METRICS = PRIMARY_METRICS + ("mean_bias_pct", "nrmse_mean")
ROUTING_LAWS = REPO_ROOT / "results" / "moe_routing" / "routing-laws.json"


def load_inputs(
    cache_path: Path = CACHE, artifact_path: Path = ARTIFACT
) -> tuple[dict, dict, dict[int, dict]]:
    with np.load(cache_path, allow_pickle=True) as data:
        cache = {key: data[key] for key in data.files}
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("schema_version") not in (
        "power-test-surface-v1", "power-test-surface-v2"
    ):
        raise ValueError("Unknown fitted power-surface schema")
    run_index = json.loads(RUN_INDEX.read_text())
    sources = {int(row["run_id"]): row for row in run_index["runs"]}
    if set(sources) != set(map(int, np.unique(cache["run_id"]))):
        raise ValueError("Timing run index and joined cache disagree on run IDs")
    return cache, artifact, sources


def evaluate(cache: dict, artifact: dict, sources: dict[int, dict]) -> list[dict]:
    rows = []
    role_names = list(map(str, cache["role_names"]))
    scored = {role_names.index(role) for role in SCORED_ROLES}
    dt = float(cache["dt_s"])
    for hardware in map(str, cache["hw_names"]):
        hw = cache["hw_idx"] == list(cache["hw_names"]).index(hardware)
        sub = {key: value[hw] for key, value in cache.items()
               if isinstance(value, np.ndarray) and value.shape == hw.shape}
        design, names = surface_design(sub, hardware)
        fit = artifact["per_hardware"][hardware]
        coefficients = coefficients_in_design_order(fit, names)
        for run_id, lo, hi in run_slices(sub["run_id"]):
            role_idx = int(sub["role_idx"][lo])
            if role_idx not in scored:
                continue
            measured, n_bad = interpolate_nan(sub["power"][lo:hi].astype(float))
            raw = predict(design[lo:hi], coefficients, sub["tp"][lo:hi], hardware)
            predicted = apply_chain(raw, dt, hardware, float(fit["delay_s"]))
            metrics = trace_metrics(measured, predicted, native_dt=dt)
            metrics = {key: metrics.get(key, float("nan")) for key in ROW_METRICS}
            model = str(cache["model_names"][sub["model_idx"][lo]])
            family = str(cache["family_names"][sub["family_idx"][lo]])
            rows.append({
                "hardware": hardware,
                "role": role_names[role_idx],
                "run_id": run_id,
                "source_id": sources[run_id]["source_id"],
                "model": model,
                "tp": int(sub["tp"][lo]),
                "rate": float(sub["rate"][lo]),
                "dense": not family.startswith("moe"),
                "interpolated_power_bins": n_bad,
                "n_bins": hi - lo,
                **metrics,
            })
    return rows


def model_tables(rows: list[dict]) -> list[dict]:
    output = []
    for key in sorted({(r["hardware"], r["role"], r["model"]) for r in rows}):
        selected = [r for r in rows
                    if (r["hardware"], r["role"], r["model"]) == key]
        table = role_table([{**r, "role": f"{r['role']}:{r['model']}"}
                            for r in selected])[0]
        table["role"], table["model"] = key[1], key[2]
        output.append(table)
    return output


def reference_split(row: dict, candidate: str) -> str:
    if candidate == "B2":
        return f"S0_{row['hardware']}"
    if row["model"] == "gpt-oss-120b":
        return "S1_A100_gpt_oss"
    if row["model"] == "llama-3-405b":
        return "S2b_H100_llama405"
    return f"S0_{row['hardware']}"


def legacy_references(rows: list[dict]) -> tuple[list[dict], dict]:
    with V2_METRICS.open(newline="") as f:
        legacy = list(csv.DictReader(f))
    lookup = {(r["source_id"], r["candidate"], r["split"]): r for r in legacy}
    output = []
    counts = {"arrival_only_runs": len(rows), "M4A_matches": 0, "B2_matches": 0}
    for row in rows:
        for candidate in ("M4A", "B2"):
            key = (row["source_id"], candidate, reference_split(row, candidate))
            if key not in lookup:
                continue
            old = lookup[key]
            output.append({
                "hardware": row["hardware"],
                "role": row["role"],
                "model": row["model"],
                "run_id": row["run_id"],
                "rate": row["rate"],
                "source_id": row["source_id"],
                "candidate": candidate,
                "split": old["split"],
                **{metric: float(old[metric]) for metric in ROW_METRICS},
            })
            counts[f"{candidate}_matches"] += 1
    return output, counts


def write_csv(rows: list[dict], path: Path = OUT_CSV) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--surface", type=Path, default=ARTIFACT)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    cache, artifact, sources = load_inputs(args.cache, args.surface)
    routing_mode = str(cache.get("moe_routing_mode", "uniform"))
    routing_provenance = None
    if routing_mode == "measured":
        laws = json.loads(ROUTING_LAWS.read_text())
        routing_provenance = {
            "schema_version": laws["schema_version"],
            "source": "sharegpt",
            "source_binding": laws["source_binding"],
            "capture_sha256": {
                model: record["capture_sha256"]
                for model, record in laws["models"].items()
            },
        }
    rows = evaluate(cache, artifact, sources)
    references, counts = legacy_references(rows)
    reference_tables = {
        candidate: role_table(
            [r for r in references if r["candidate"] == candidate])
        for candidate in ("M4A", "B2")
    }
    report = {
        "schema_version": "arrival-only-evaluation-v1",
        "timing_contract": "arrival_only",
        "power_fit_population": "dense training bins only",
        "scored_roles": list(SCORED_ROLES),
        "holdout_power_used_for_fit": False,
        "moe_status": (
            "measured ShareGPT routing; dense-only power fit"
            if routing_mode == "measured"
            else "uniform-independent routing; reported, not used for fit"
        ),
        "moe_routing_provenance": routing_provenance,
        "known_timing_inheritance": {
            "llama-3-405b_rate_4": "scheduler has no preemption mechanism",
            "gpt-oss": routing_mode,
        },
        "comparison_note": "M4A and B2 rows are exact-source contextual references. "
                           "Their conditional-timing/S0 contracts and trace horizons "
                           "are not identical to this arrival-only evaluation.",
        "reference_coverage": counts,
        "tables": role_table(rows),
        "tables_dense_only": role_table([row for row in rows if row["dense"]]),
        "tables_by_model": model_tables(rows),
        "legacy_reference_tables": reference_tables,
        "per_run": rows,
    }
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(rows, args.out_csv)
    print(f"Scored {len(rows)} frozen arrival-only runs -> {args.out_json}")
    for table in report["tables_by_model"]:
        print(f"{table['hardware']} {table['role']} {table['model']}: "
              f"runs={table['runs']} energy={table['energy_error_pct_median']:.2f}% "
              f"acf_mae={table['acf_mae_median']:.4f} "
              f"nrmse={table['nrmse_range_median']:.3f}")
    print(f"Legacy reference matches: M4A={counts['M4A_matches']} "
          f"B2={counts['B2_matches']}")


if __name__ == "__main__":
    main()
