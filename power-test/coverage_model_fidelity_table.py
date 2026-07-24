"""Write a model-wise held-out fidelity table for the coverage split."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RUN_CSV_PATH = BASE / "coverage_power_per_run.csv"
LATEX_PATH = BASE / "coverage_model_fidelity_table.tex"
CSV_PATH = BASE / "coverage_model_fidelity_table.csv"
REPORT_PATH = BASE / "coverage_model_fidelity_table.json"
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20260721

METRICS = (
    ("nrmse_range", "Power error", "pct", False),
    ("ks_agreement", "Distribution agreement", "unit", True),
    ("energy_error_pct", "Energy error", "already_pct", False),
    ("temporal_error_pct", "Temporal error", "pct_from_soft_dtw", False),
)
SOURCE_METRICS = (
    "nrmse_range", "ks_agreement", "energy_error_pct", "soft_dtw_divergence"
)

MODEL_LABELS = {
    "deepseek-r1-distill-8b": "DeepSeek-R1-Distill 8B",
    "deepseek-r1-distill-70b": "DeepSeek-R1-Distill 70B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
    "llama-3-8b": "Llama 3 8B",
    "llama-3-70b": "Llama 3 70B",
    "llama-3-405b": "Llama 3 405B",
}


def load_heldout_rows(path: Path = RUN_CSV_PATH) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        if row["role"] == "train_source":
            continue
        if row["surface_supported"] != "True":
            continue
        output.append({
            "model": row["model"],
            "hardware": row["hardware"],
            "tp": int(row["tp"]),
            "rate": float(row["rate"]),
            **{metric: float(row[metric]) for metric in SOURCE_METRICS},
        })
    return output


def point_medians(rows: list[dict], metric: str) -> list[float]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["hardware"], row["tp"], row["rate"])].append(row[metric])
    return [float(np.median(values)) for _, values in sorted(groups.items())]


def bootstrap_median(
    values: list[float], rng: np.random.Generator, draws: int = BOOTSTRAP_DRAWS
) -> dict:
    array = np.asarray(values, float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty model group")
    indices = rng.integers(0, array.size, size=(draws, array.size))
    samples = np.median(array[indices], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "median": float(np.median(array)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def metric_values(rows: list[dict], metric: str, scale: str) -> list[float]:
    if scale == "pct":
        return [100.0 * value for value in point_medians(rows, metric)]
    if scale == "pct_from_soft_dtw":
        return [
            100.0 * np.sqrt(value)
            for value in point_medians(rows, "soft_dtw_divergence")
        ]
    if scale == "already_pct":
        return point_medians(rows, metric)
    return point_medians(rows, metric)


def summarize_by_model(
    rows: list[dict], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    output = []
    for model, selected in sorted(by_model.items()):
        points = {
            (row["hardware"], row["tp"], row["rate"]) for row in selected
        }
        record = {
            "model": model,
            "label": MODEL_LABELS.get(model, model),
            "traces": len(selected),
            "points": len(points),
        }
        for metric, _, scale, _ in METRICS:
            values = metric_values(selected, metric, scale)
            record[metric] = bootstrap_median(values, rng, draws)
        output.append(record)
    return output


def _interval(values: dict, digits: int) -> str:
    return (
        f"{values['median']:.{digits}f} "
        f"[{values['ci95_low']:.{digits}f}, {values['ci95_high']:.{digits}f}]"
    )


def latex_rows(summary: list[dict]) -> list[str]:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Held-out simulator fidelity by model. Values are medians with 95\% bootstrap confidence intervals over hardware/TP/rate configuration points (1000 resamples). Repeated runs at the same point are collapsed by their median before resampling. Temporal error is $100\sqrt{\mathrm{SoftDTW}}$, reported as a percentage of the measured trace power range. Lower is better for errors; higher is better for distribution agreement.}",
        r"\label{tab:coverage-model-fidelity}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"\textbf{Model} & \textbf{Traces} & \textbf{Points} & "
            r"\textbf{Power error (\%)} & \textbf{Distribution} & "
            r"\textbf{Energy error (\%)} & \textbf{Temporal error (\%)} \\"
        ),
        r"\midrule",
    ]
    for row in summary:
        lines.append(" & ".join([
            row["label"],
            str(row["traces"]),
            str(row["points"]),
            _interval(row["nrmse_range"], 2),
            _interval(row["ks_agreement"], 3),
            _interval(row["energy_error_pct"], 2),
            _interval(row["temporal_error_pct"], 2),
        ]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return lines


def write_outputs(summary: list[dict]) -> None:
    with CSV_PATH.open("w", newline="") as stream:
        fields = ["model", "label", "traces", "points"]
        for metric, _, _, _ in METRICS:
            fields.extend([
                f"{metric}_median", f"{metric}_ci95_low",
                f"{metric}_ci95_high",
            ])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            flat = {key: row[key] for key in ("model", "label", "traces", "points")}
            for metric, _, _, _ in METRICS:
                flat[f"{metric}_median"] = row[metric]["median"]
                flat[f"{metric}_ci95_low"] = row[metric]["ci95_low"]
                flat[f"{metric}_ci95_high"] = row[metric]["ci95_high"]
            writer.writerow(flat)
    LATEX_PATH.write_text("\n".join(latex_rows(summary)) + "\n")
    REPORT_PATH.write_text(json.dumps({
        "schema_version": "coverage-model-fidelity-table-v1",
        "input": str(RUN_CSV_PATH),
        "rows": "held-out supported rows only",
        "point": "one model-local hardware, TP, and rate configuration; repeated runs are collapsed by median before bootstrapping",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "metrics": [
            {"name": name, "label": label, "scale": scale, "higher_is_better": hib}
            for name, label, scale, hib in METRICS
        ],
        "models": summary,
    }, indent=2) + "\n")


def main() -> None:
    summary = summarize_by_model(load_heldout_rows())
    write_outputs(summary)
    print(LATEX_PATH)
    print(CSV_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
