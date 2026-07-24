"""Plot rate sweeps and rate-4 traces for coverage-split held-out models."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [str(BASE), str(ROOT)]

from fit_clean_power_pipelines import predict_dense_node_power  # noqa: E402
from fit_power_surface import interpolate_nan  # noqa: E402
from plot_power_metric_audit import one_second_pair  # noqa: E402

CACHE_PATH = BASE / "sim_ledger_power_coverage_250ms.npz"
SURFACE_PATH = BASE / "coverage_power_surfaces.json"
RUN_CSV_PATH = BASE / "coverage_power_per_run.csv"
RATE_SWEEP_PATH = BASE / "coverage_heldout_models_rate_sweep.pdf"
RATE_SWEEP_PNG = BASE / "coverage_heldout_models_rate_sweep.png"
TRACE_PATH = BASE / "coverage_heldout_models_rate4_traces.pdf"
TRACE_PNG = BASE / "coverage_heldout_models_rate4_traces.png"
REPORT_PATH = BASE / "coverage_heldout_models_plots.json"

RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
MODEL_ROWS = (
    ("A100", "deepseek-r1-distill-70b", "A100 · DeepSeek-R1-Distill 70B"),
    ("H100", "deepseek-r1-distill-8b", "H100 · DeepSeek-R1-Distill 8B"),
)
TP_COLORS = {1: "#0072B2", 2: "#009E73", 4: "#D55E00", 8: "#CC79A7"}
STANFORD_RED = "#8C1515"


def load_rows(path: Path = RUN_CSV_PATH) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        if row["role"] != "heldout_model":
            continue
        output.append({
            **row,
            "run_id": int(row["run_id"]),
            "tp": int(row["tp"]),
            "rate": float(row["rate"]),
            "energy_error_pct": float(row["energy_error_pct"]),
            "rmse_w_per_gpu": float(row["rmse_w_per_gpu"]),
            "acf_r2": float(row["acf_r2"]),
            "nrmse_range": float(row["nrmse_range"]),
        })
    return output


def aggregate_cells(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["hardware"], row["model"], row["tp"], row["rate"])].append(row)
    output = []
    for (hardware, model, tp, rate), values in sorted(groups.items()):
        output.append({
            "hardware": hardware,
            "model": model,
            "tp": tp,
            "rate": rate,
            "runs": len(values),
            **{
                metric: float(np.median([row[metric] for row in values]))
                for metric in ("energy_error_pct", "rmse_w_per_gpu", "acf_r2")
            },
        })
    return output


def metric_medoid(rows: list[dict]) -> dict:
    values = np.asarray([
        [row["energy_error_pct"], row["rmse_w_per_gpu"], -row["acf_r2"]]
        for row in rows
    ])
    center = np.median(values, axis=0)
    span = np.ptp(values, axis=0)
    span[span == 0.0] = 1.0
    distance = np.sum(np.abs(values - center) / span, axis=1)
    index = min(range(len(rows)), key=lambda i: (distance[i], rows[i]["run_id"]))
    return rows[index]


def selected_rate4_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["rate"] == 4.0:
            groups[(row["hardware"], row["model"], row["tp"])].append(row)
    return [metric_medoid(values) for _, values in sorted(groups.items())]


def load_cache(path: Path = CACHE_PATH) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def one_second_run(cache: dict, prediction: np.ndarray, run_id: int):
    selected = cache["run_id"] == run_id
    tp_values = np.unique(cache["tp"][selected])
    if tp_values.size != 1:
        raise ValueError(f"Run {run_id} has non-constant TP")
    tp = int(tp_values[0])
    measured, _ = interpolate_nan(cache["power"][selected].astype(float))
    measured, predicted = one_second_pair(
        measured / tp, prediction[selected] / tp, float(cache["dt_s"])
    )
    return measured, predicted


def plot_rate_sweep(cells: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    metrics = (
        ("energy_error_pct", "Energy error (%)", (0.0, None)),
        ("rmse_w_per_gpu", "Power RMSE (W/GPU)", (0.0, None)),
        ("acf_r2", r"Temporal similarity ($R^2$)", None),
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.0), sharex=True)
    positions = np.arange(len(RATES))
    for row_index, (hardware, model, label) in enumerate(MODEL_ROWS):
        selected = [
            row for row in cells
            if row["hardware"] == hardware and row["model"] == model
        ]
        for column, (metric, ylabel, ylim) in enumerate(metrics):
            axis = axes[row_index, column]
            axis.axvspan(4.5, 5.5, color=STANFORD_RED, alpha=0.07, linewidth=0)
            for tp in sorted({row["tp"] for row in selected}):
                values = [
                    next(row[metric] for row in selected
                         if row["tp"] == tp and row["rate"] == rate)
                    for rate in RATES
                ]
                axis.plot(
                    positions, values, marker="o", markersize=3.8,
                    linewidth=1.1, color=TP_COLORS[tp], label=f"TP{tp}",
                )
            axis.set_ylabel(ylabel)
            if ylim is not None:
                axis.set_ylim(bottom=ylim[0])
            if metric == "acf_r2":
                axis.axhline(0.9, color="0.45", linestyle="--", linewidth=0.7)
            if row_index == 0:
                axis.set_title(label, fontsize=9.5)
            else:
                axis.set_title(label, fontsize=9.5)
            axis.grid(True, alpha=0.25)
            axis.set_xticks(positions, [f"{rate:g}" for rate in RATES])
            if row_index == 1:
                axis.set_xlabel("Request rate (requests/s)")
    handles = [
        Line2D([], [], color=color, marker="o", linewidth=1.1, label=f"TP{tp}")
        for tp, color in sorted(TP_COLORS.items())
    ]
    labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(RATE_SWEEP_PATH, bbox_inches="tight")
    fig.savefig(RATE_SWEEP_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rate4_traces(rows: list[dict], cache: dict, prediction: np.ndarray) -> list[dict]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    fig = plt.figure(figsize=(10.0, 5.0))
    grid = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.28)
    selected = selected_rate4_rows(rows)
    placements = {}
    for row in selected:
        key = (row["hardware"], row["model"], row["tp"])
        if row["hardware"] == "A100":
            placements[key] = grid[0, 0:2] if row["tp"] == 4 else grid[0, 2:4]
        else:
            placements[key] = grid[1, {1: 0, 2: 1, 4: 2, 8: 3}[row["tp"]]]

    plotted = []
    for row in selected:
        axis = fig.add_subplot(placements[(row["hardware"], row["model"], row["tp"])])
        measured, predicted = one_second_run(cache, prediction, row["run_id"])
        seconds = min(600, measured.size, predicted.size)
        time_min = np.arange(seconds) / 60.0
        axis.plot(time_min, measured[:seconds], color="black", linewidth=0.65)
        axis.plot(time_min, predicted[:seconds], color=STANFORD_RED, linewidth=0.85)
        axis.set_title(f"{row['hardware']} · TP{row['tp']}", fontsize=9)
        axis.set_xlabel("Time (min)")
        axis.set_ylabel("Power/GPU (W)")
        axis.grid(True, alpha=0.23)
        axis.text(
            0.02, 0.96,
            f"Energy {row['energy_error_pct']:.1f}%\n$R^2$ {row['acf_r2']:.3f}",
            transform=axis.transAxes, va="top", fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
        plotted.append({
            "run_id": row["run_id"], "hardware": row["hardware"],
            "model": row["model"], "tp": row["tp"], "rate": row["rate"],
            "energy_error_pct": row["energy_error_pct"],
            "rmse_w_per_gpu": row["rmse_w_per_gpu"], "acf_r2": row["acf_r2"],
        })
    handles = [
        Line2D([], [], color="black", linewidth=0.8, label="Measured"),
        Line2D([], [], color=STANFORD_RED, linewidth=0.9, label="PowerTrace-Sim"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Held-out model traces at 4 requests/s", y=0.96, fontsize=11)
    fig.savefig(TRACE_PATH, bbox_inches="tight")
    fig.savefig(TRACE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plotted


def main() -> None:
    rows = load_rows()
    cells = aggregate_cells(rows)
    plot_rate_sweep(cells)
    cache = load_cache()
    artifact = json.loads(SURFACE_PATH.read_text())
    prediction = predict_dense_node_power(cache, artifact)
    plotted = plot_rate4_traces(rows, cache, prediction)
    REPORT_PATH.write_text(json.dumps({
        "input_rows": str(RUN_CSV_PATH),
        "heldout_traces": len(rows),
        "heldout_cells": len(cells),
        "rates": list(RATES),
        "rate_sweep": str(RATE_SWEEP_PATH),
        "rate4_traces": str(TRACE_PATH),
        "rate4_selections": plotted,
    }, indent=2) + "\n")
    for path in (RATE_SWEEP_PATH, RATE_SWEEP_PNG, TRACE_PATH, TRACE_PNG, REPORT_PATH):
        print(path)


if __name__ == "__main__":
    main()
