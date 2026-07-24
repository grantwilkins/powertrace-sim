"""Fit and score the clean power model on the 117/333 coverage split."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [str(BASE), str(ROOT)]

from fit_clean_power_pipelines import (  # noqa: E402
    fit_surfaces,
    load_cache,
    run_metadata,
    score_runs,
)
from model.training_data.coverage_split import assign_role  # noqa: E402

METRICS = (
    ("energy_error_pct", "Energy error (percent)"),
    ("rmse_w_per_gpu", "Power RMSE (W/GPU)"),
    ("acf_r2", "Temporal similarity (R2)"),
)
ROLE_LABELS = {
    "train_source": "Training examples",
    "heldout_rate": "New rates",
    "heldout_tp": "New TP setups",
    "heldout_model": "New model setups",
}


def coverage_roles(metadata: dict[int, dict]) -> dict[int, str]:
    roles = {}
    for run, row in metadata.items():
        role = assign_role(row["model"], row["hardware"], row["tp"], row["rate"])
        roles[run] = "train_source" if role == "train" else role
    return roles


def bootstrap_median(rows: list[dict], metric: str, rng, draws=10_000):
    cells = defaultdict(list)
    for row in rows:
        cell = (row["model"], row["hardware"], row["tp"], row["rate"])
        cells[cell].append(float(row[metric]))
    keys = list(cells)
    samples = np.empty(draws)
    for draw in range(draws):
        chosen = rng.integers(0, len(keys), len(keys))
        values = [value for index in chosen for value in cells[keys[index]]]
        samples[draw] = np.median(values)
    observed = np.median([float(row[metric]) for row in rows])
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(observed), float(low), float(high)


def bootstrap_values(values, rng, draws=10_000):
    values = np.asarray(values, float)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    medians = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, (0.025, 0.975))
    return float(np.median(values)), float(low), float(high)


def summarize(rows: list[dict], seed=20260721) -> list[dict]:
    rng = np.random.default_rng(seed)
    output = []
    for role in ROLE_LABELS:
        selected = [row for row in rows if row["role"] == role]
        record = {
            "role": role,
            "label": ROLE_LABELS[role],
            "traces": len(selected),
            "cells": len({
                (row["model"], row["hardware"], row["tp"], row["rate"])
                for row in selected
            }),
        }
        for metric, _ in METRICS:
            median, low, high = bootstrap_median(selected, metric, rng)
            record[metric] = {"median": median, "ci95": [low, high]}
        output.append(record)
    heldout = [row for row in rows if row["role"] != "train_source"]
    record = {
        "role": "all_heldout",
        "label": "All held-out traces",
        "traces": len(heldout),
        "cells": len({
            (row["model"], row["hardware"], row["tp"], row["rate"])
            for row in heldout
        }),
    }
    for metric, _ in METRICS:
        median, low, high = bootstrap_median(heldout, metric, rng)
        record[metric] = {"median": median, "ci95": [low, high]}
    output.append(record)
    return output


def add_timing(summary: list[dict], path: Path, seed=20260721) -> None:
    rows = list(csv.DictReader(path.open()))
    rng = np.random.default_rng(seed)
    for record in summary:
        if record["role"] == "train_source":
            continue
        selected = rows if record["role"] == "all_heldout" else [
            row for row in rows if row["role"] == record["role"]
        ]
        values = [float(row["model_e2e_medabs_pct"]) for row in selected]
        median, low, high = bootstrap_values(values, rng)
        record["e2e_error_pct"] = {"median": median, "ci95": [low, high]}


def interval(record: dict, key: str, digits: int) -> str:
    value = record[key]
    return (
        f"{value['median']:.{digits}f} "
        f"[{value['ci95'][0]:.{digits}f}, {value['ci95'][1]:.{digits}f}]"
    )


def write_latex(path: Path, summary: list[dict]) -> None:
    rows = [row for row in summary if row["role"] != "train_source"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Accuracy on held-out traces. Values are medians with 95\% confidence intervals. Lower is better for errors; temporal similarity is best at 1.}",
        r"\label{tab:coverage-trace-fidelity}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Held-out data} & \textbf{Traces} & \textbf{Timing error (\%)} & \textbf{Energy error (\%)} & \textbf{Power RMSE (W/GPU)} & \textbf{Temporal similarity} \\",
        r"\midrule",
    ]
    for row in rows:
        label = row["label"]
        values = [
            label,
            str(row["traces"]),
            interval(row, "e2e_error_pct", 2),
            interval(row, "energy_error_pct", 2),
            interval(row, "rmse_w_per_gpu", 1),
            interval(row, "acf_r2", 3),
        ]
        if row["role"] == "all_heldout":
            lines.append(r"\midrule")
            values = [rf"\textbf{{{value}}}" for value in values]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="power-test/sim_ledger_power_coverage_250ms.npz")
    parser.add_argument("--surface-out", default="power-test/coverage_power_surfaces.json")
    parser.add_argument("--per-run-out", default="power-test/coverage_power_per_run.csv")
    parser.add_argument("--report-out", default="power-test/coverage_power_report.json")
    parser.add_argument(
        "--timing-csv", default="results/timing_test_coverage/cell_metrics.csv"
    )
    parser.add_argument(
        "--latex-out", default="power-test/coverage_trace_fidelity_table.tex"
    )
    args = parser.parse_args()

    cache = load_cache(Path(args.cache))
    metadata = run_metadata(cache)
    roles = coverage_roles(metadata)
    artifact, prediction = fit_surfaces(cache, metadata, roles)
    artifact["schema_version"] = "coverage-basis-power-surfaces-v1"
    artifact["training_policy"] = (
        "117 traces from 39 whole configuration/rate cells; every measured "
        "hardware, TP degree, rate, model class, and precision regime appears "
        "in training; 333 traces remain held out by rate, TP setup, or model setup"
    )
    Path(args.surface_out).write_text(json.dumps(artifact, indent=2) + "\n")

    rows = score_runs(cache, metadata, roles, prediction)
    write_csv(Path(args.per_run_out), rows)
    summary = summarize(rows)
    add_timing(summary, Path(args.timing_csv))
    report = {
        "schema_version": "coverage-basis-power-report-v1",
        "surface": args.surface_out,
        "cache": args.cache,
        "groups": summary,
    }
    Path(args.report_out).write_text(json.dumps(report, indent=2) + "\n")
    write_latex(Path(args.latex_out), summary)
    for group in summary:
        print(group)


if __name__ == "__main__":
    main()
