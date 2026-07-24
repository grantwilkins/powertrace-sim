"""Plot representative held-out traces at four rates for the best fixed setup."""
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

from fit_clean_power_pipelines import (  # noqa: E402
    predict_dense_node_power,
    sha256_file,
)
from plot_best_rate_traces import (  # noqa: E402
    MEASURED_ALPHA_RANGE,
    PREDICTED_ALPHA_RANGE,
    STANFORD_RED,
    add_alpha_gradient_line,
    crop_ten_minutes,
)
from plot_coverage_heldout_models import (  # noqa: E402
    load_cache,
    load_rows,
    metric_medoid,
    one_second_run,
)

CACHE_PATH = BASE / "sim_ledger_power_coverage_250ms.npz"
SURFACE_PATH = BASE / "coverage_power_surfaces.json"
RUN_CSV_PATH = BASE / "coverage_power_per_run.csv"
REPORT_PATH = BASE / "best_coverage_holdout_traces_1s.json"
CSV_PATH = BASE / "best_coverage_holdout_traces_1s.csv"
PLOT_RATES = (0.125, 1.0, 2.0, 4.0)
TRACE_SECONDS = 600
FIGSIZE = (11.0, 4.0)
PREDICTED_LABEL = "Our Simulator"
LEGEND_ANCHOR = (0.5, -0.34)
SEABORN_CONTEXT = "talk"
FONT_SCALE = 1.2
MEASURED_LINEWIDTH = 1.35
PREDICTED_LINEWIDTH = 1.8
Y_LABEL = "Power (W/GPU)"


def _ordinal_ranks(values: list[float], *, maximize: bool = False) -> np.ndarray:
    array = np.asarray(values, float)
    order = np.argsort(-array if maximize else array, kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def select_configuration(rows: list[dict], rates=PLOT_RATES) -> dict:
    """Select one complete held-out setup using three standard trace metrics."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["hardware"], row["model"], int(row["tp"]))].append(row)
    summaries = []
    for (hardware, model, tp), values in sorted(groups.items()):
        selected = [row for row in values if float(row["rate"]) in rates]
        if {float(row["rate"]) for row in selected} != set(rates):
            continue
        summaries.append({
            "hardware": hardware,
            "model": model,
            "tp": tp,
            "runs": len(selected),
            "energy_error_pct": float(np.median(
                [row["energy_error_pct"] for row in selected]
            )),
            "rmse_w_per_gpu": float(np.median(
                [row["rmse_w_per_gpu"] for row in selected]
            )),
            "acf_r2": float(np.median([row["acf_r2"] for row in selected])),
        })
    if not summaries:
        raise ValueError("No held-out configuration covers all requested rates")
    metrics = (
        ("energy_error_pct", False),
        ("rmse_w_per_gpu", False),
        ("acf_r2", True),
    )
    for metric, maximize in metrics:
        ranks = _ordinal_ranks(
            [row[metric] for row in summaries], maximize=maximize
        )
        for row, rank in zip(summaries, ranks):
            row.setdefault("ranks", {})[metric] = int(rank)
    for row in summaries:
        row["rank_sum"] = sum(row["ranks"].values())
    return min(summaries, key=lambda row: (
        row["rank_sum"], row["energy_error_pct"], row["rmse_w_per_gpu"],
        -row["acf_r2"], row["hardware"], row["model"], row["tp"],
    ))


def select_representative_rows(
    rows: list[dict], selection: dict, rates=PLOT_RATES
) -> list[dict]:
    output = []
    for rate in rates:
        candidates = [row for row in rows if (
            row["hardware"] == selection["hardware"]
            and row["model"] == selection["model"]
            and int(row["tp"]) == selection["tp"]
            and float(row["rate"]) == rate
        )]
        if not candidates:
            raise ValueError(f"No candidate traces at {rate:g} requests/s")
        output.append(metric_medoid(candidates))
    return output


def trace_series(cache: dict, prediction: np.ndarray, run_id: int):
    measured, predicted = one_second_run(cache, prediction, run_id)
    measured = crop_ten_minutes(measured, 1.0)
    predicted = crop_ten_minutes(predicted, 1.0)
    time_min = np.arange(TRACE_SECONDS) / 60.0
    return time_min, measured, predicted


def plot_paths(selection: dict, rate: float) -> tuple[Path, Path]:
    rate_slug = f"{rate:g}".replace(".", "p")
    stem = (
        f"power_trace_{selection['model'].replace('-', '_')}_"
        f"{selection['hardware'].lower()}_tp{selection['tp']}_"
        f"rate_{rate_slug}_coverage_1s"
    )
    return BASE / f"{stem}.pdf", BASE / f"{stem}.png"


def save_trace(row: dict, series, paths: tuple[Path, Path], ymax_w: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns

    sns.set_theme(style="whitegrid", context=SEABORN_CONTEXT,
                  font_scale=FONT_SCALE)
    time_min, measured, predicted = series
    fig, axis = plt.subplots(figsize=FIGSIZE)
    add_alpha_gradient_line(
        axis, time_min, measured, color="black", linewidth=MEASURED_LINEWIDTH,
        alpha_range=MEASURED_ALPHA_RANGE, label="Measured",
    )
    add_alpha_gradient_line(
        axis, time_min, predicted, color=STANFORD_RED,
        linewidth=PREDICTED_LINEWIDTH,
        alpha_range=PREDICTED_ALPHA_RANGE, label=PREDICTED_LABEL,
    )
    axis.set(
        xlabel="Time (min)", ylabel=Y_LABEL,
        xlim=(0.0, TRACE_SECONDS / 60.0), ylim=(0.0, ymax_w),
    )
    axis.grid(True, alpha=0.25)
    handles = [
        Line2D([], [], color="black", linewidth=MEASURED_LINEWIDTH,
               label="Measured"),
        Line2D([], [], color=STANFORD_RED, linewidth=PREDICTED_LINEWIDTH,
               label=PREDICTED_LABEL),
    ]
    axis.legend(
        handles=handles, loc="upper center", bbox_to_anchor=LEGEND_ANCHOR,
        frameon=False, ncol=2, handlelength=1.5,
        columnspacing=0.8, borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.11, right=0.995, top=0.96, bottom=0.34)
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths[1], dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    rows = load_rows(RUN_CSV_PATH)
    selection = select_configuration(rows)
    selected = select_representative_rows(rows, selection)
    cache = load_cache(CACHE_PATH)
    artifact = json.loads(SURFACE_PATH.read_text())
    prediction = predict_dense_node_power(cache, artifact)
    series = {
        row["run_id"]: trace_series(cache, prediction, row["run_id"])
        for row in selected
    }
    ymax_w = 1.05 * max(
        np.max(values) for traces in series.values() for values in traces[1:]
    )
    files = []
    for row in selected:
        paths = plot_paths(selection, row["rate"])
        save_trace(row, series[row["run_id"]], paths, ymax_w)
        files.extend(paths)
        print(*paths, sep="\n")
    with CSV_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "rate_requests_s", "run_id", "time_s", "measured_w_per_gpu",
            "predicted_w_per_gpu",
        ))
        writer.writeheader()
        for row in selected:
            _, measured, predicted = series[row["run_id"]]
            writer.writerows({
                "rate_requests_s": row["rate"],
                "run_id": row["run_id"],
                "time_s": second,
                "measured_w_per_gpu": float(measured[second]),
                "predicted_w_per_gpu": float(predicted[second]),
            } for second in range(TRACE_SECONDS))
    REPORT_PATH.write_text(json.dumps({
        "selection_rule": "lowest rank sum across median energy error, power "
                          "RMSE, and temporal similarity at all four rates; "
                          "one fixed held-out configuration",
        "repetition_rule": "three-metric medoid within each rate",
        "rates": list(PLOT_RATES),
        "selection": selection,
        "trace_seconds": TRACE_SECONDS,
        "aggregation": "matched nonoverlapping one-second per-GPU means",
        "inputs": {
            "cache": str(CACHE_PATH),
            "cache_sha256": sha256_file(CACHE_PATH),
            "surface": str(SURFACE_PATH),
            "surface_sha256": sha256_file(SURFACE_PATH),
            "rows": str(RUN_CSV_PATH),
            "rows_sha256": sha256_file(RUN_CSV_PATH),
        },
        "panels": selected,
        "data_file": str(CSV_PATH),
        "files": list(map(str, files)),
    }, indent=2) + "\n")
    print(CSV_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
