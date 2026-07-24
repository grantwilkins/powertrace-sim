"""Plot four 10-minute traces for Llama-3 8B on H100 TP1."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "feature-test"))

from evaluation_core import trace_metrics  # noqa: E402
from fit_clean_power_pipelines import (  # noqa: E402
    predict_dense_node_power,
    sha256_file,
)
from fit_power_surface import interpolate_nan  # noqa: E402
from plot_power_metric_audit import one_second_pair  # noqa: E402

CACHE_PATH = BASE / "sim_ledger_power_uniform_current_250ms.npz"
ARTIFACT_PATH = BASE / "clean_power_surfaces.json"
REPORT_PATH = BASE / "best_rate_traces_1s.json"
CSV_PATH = BASE / "best_rate_traces_1s.csv"
SELECTION_RATES = (0.125, 0.25, 0.5, 1.0, 2.0)
PLOT_RATES = (0.125, 0.5, 1.0, 2.0)
TRACE_SECONDS = 600.0
STANFORD_RED = "#8C1515"
MEASURED_ALPHA_RANGE = (0.35, 0.78)
PREDICTED_ALPHA_RANGE = (0.50, 0.98)
TARGET_MODEL = "llama-3-8b"
TARGET_HARDWARE = "H100"
TARGET_TP = 1


def load_cache(path: Path = CACHE_PATH) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def run_view(cache: dict, run_id: int) -> dict:
    selected = cache["run_id"] == run_id
    if not selected.any():
        raise ValueError(f"Run {run_id} is absent from the cache")
    return {
        key: value[selected]
        for key, value in cache.items()
        if isinstance(value, np.ndarray) and value.shape == selected.shape
    }


def run_metadata(cache: dict) -> dict[int, dict]:
    output = {}
    for run_id in map(int, np.unique(cache["run_id"])):
        index = int(np.flatnonzero(cache["run_id"] == run_id)[0])
        output[run_id] = {
            "hardware": str(cache["hw_names"][cache["hw_idx"][index]]),
            "model": str(cache["model_names"][cache["model_idx"][index]]),
            "tp": int(cache["tp"][index]),
            "rate": float(cache["rate"][index]),
            "role": str(cache["role_names"][cache["role_idx"][index]]),
        }
    return output


def _ordinal_ranks(values: list[float], *, maximize: bool = False) -> np.ndarray:
    values_array = np.asarray(values, float)
    order = np.argsort(-values_array if maximize else values_array, kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def select_configuration(rows: list[dict], rates=SELECTION_RATES) -> dict:
    """Select the lowest rank sum over energy, ACF R², and soft-DTW."""
    keys = sorted({(row["hardware"], row["model"], int(row["tp"]))
                   for row in rows if row["dense"]})
    summaries = []
    for hardware, model, tp in keys:
        selected = [
            row for row in rows
            if row["dense"] and row["hardware"] == hardware
            and row["model"] == model and int(row["tp"]) == tp
            and float(row["rate"]) in rates
        ]
        if set(float(row["rate"]) for row in selected) != set(rates):
            continue
        summaries.append({
            "hardware": hardware,
            "model": model,
            "tp": tp,
            "runs": len(selected),
            "energy_error_pct": float(np.median(
                [row["energy_error_pct"] for row in selected])),
            "acf_r2": float(np.median([row["acf_r2"] for row in selected])),
            "soft_dtw_divergence": float(np.median(
                [row["soft_dtw_divergence"] for row in selected])),
        })
    if not summaries:
        raise ValueError("No dense configuration covers every selection rate")
    energy_rank = _ordinal_ranks([row["energy_error_pct"] for row in summaries])
    acf_rank = _ordinal_ranks([row["acf_r2"] for row in summaries], maximize=True)
    dtw_rank = _ordinal_ranks(
        [row["soft_dtw_divergence"] for row in summaries])
    for index, row in enumerate(summaries):
        row["ranks"] = {
            "energy_error_pct": int(energy_rank[index]),
            "acf_r2": int(acf_rank[index]),
            "soft_dtw_divergence": int(dtw_rank[index]),
        }
        row["rank_sum"] = int(energy_rank[index] + acf_rank[index] + dtw_rank[index])
    return min(summaries, key=lambda row: (
        row["rank_sum"], row["soft_dtw_divergence"], row["model"], row["tp"]
    ))


def target_configuration_rows(rows: list[dict]) -> list[dict]:
    return [
        row for row in rows
        if row["model"] == TARGET_MODEL
        and row["hardware"] == TARGET_HARDWARE
        and int(row["tp"]) == TARGET_TP
    ]


def metric_medoid(rows: list[dict]) -> dict:
    """Choose the central repetition in the three requested metrics."""
    values = np.asarray([[row["energy_error_pct"], -row["acf_r2"],
                          row["soft_dtw_divergence"]] for row in rows], float)
    center = np.median(values, axis=0)
    span = np.ptp(values, axis=0)
    span[span == 0] = 1.0
    distance = np.sum(np.abs(values - center) / span, axis=1)
    index = min(range(len(rows)), key=lambda i: (distance[i], rows[i]["run_id"]))
    return rows[index]


def candidate_run_ids(cache: dict, selection: dict, rate: float) -> list[int]:
    metadata = run_metadata(cache)
    return sorted(run_id for run_id, row in metadata.items() if (
        row["hardware"] == selection["hardware"]
        and row["model"] == selection["model"]
        and row["tp"] == selection["tp"]
        and row["rate"] == rate
    ))


def comparison_row(cache: dict, prediction: np.ndarray, selection: dict,
                   rate: float) -> tuple[dict, list[dict]]:
    candidates = []
    traces = {}
    for run_id in candidate_run_ids(cache, selection, rate):
        run = run_view(cache, run_id)
        measured, _ = interpolate_nan(run["power"].astype(float))
        predicted = prediction[cache["run_id"] == run_id]
        metrics = trace_metrics(measured, predicted, native_dt=float(cache["dt_s"]))
        candidates.append({
            "run_id": run_id,
            **{key: float(metrics[key]) for key in (
                "energy_error_pct", "acf_r2", "soft_dtw_divergence")},
        })
        traces[run_id] = (measured, predicted)
    if not candidates:
        raise ValueError(f"No runs at {rate:g} requests/s")
    chosen = metric_medoid(candidates)
    measured, predicted = traces[chosen["run_id"]]
    return {
        **selection,
        "rate": rate,
        "run_id": chosen["run_id"],
        "dt_s": float(cache["dt_s"]),
        "measured": measured,
        "predicted": predicted,
        "metrics": {key: chosen[key] for key in (
            "energy_error_pct", "acf_r2", "soft_dtw_divergence")},
    }, candidates


def crop_ten_minutes(values: np.ndarray, dt_s: float) -> np.ndarray:
    bins = int(round(TRACE_SECONDS / dt_s))
    if not np.isclose(bins * dt_s, TRACE_SECONDS):
        raise ValueError("Trace timestep does not divide 600 seconds exactly")
    if len(values) < bins:
        raise ValueError(f"Trace has {len(values) * dt_s:g}s, needs 600s")
    return np.asarray(values, float)[:bins]


def plot_path(selection: dict, rate: float) -> Path:
    rate_slug = f"{rate:g}".replace(".", "p")
    return BASE / (
        f"power_trace_{selection['model'].replace('-', '_')}_"
        f"{selection['hardware'].lower()}_tp{selection['tp']}_rate_{rate_slug}_1s.pdf"
    )


def plot_series(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = int(row["tp"])
    measured = crop_ten_minutes(row["measured"], row["dt_s"]) / tp
    predicted = crop_ten_minutes(row["predicted"], row["dt_s"]) / tp
    measured_1s, predicted_1s = one_second_pair(
        measured, predicted, row["dt_s"]
    )
    time_min = np.arange(len(measured_1s)) / 60.0
    return time_min, measured_1s, predicted_1s


def add_alpha_gradient_line(axis, x, y, *, color, linewidth,
                            alpha_range, label):
    """Add a line whose segment opacity increases across elapsed time."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    points = np.column_stack((np.asarray(x, float), np.asarray(y, float)))
    if points.shape[0] < 2:
        raise ValueError("Alpha-gradient lines require at least two points")
    segments = np.stack((points[:-1], points[1:]), axis=1)
    colors = np.tile(to_rgba(color), (segments.shape[0], 1))
    colors[:, 3] = np.linspace(*alpha_range, segments.shape[0])
    collection = LineCollection(
        segments, colors=colors, linewidths=linewidth, label=label
    )
    axis.add_collection(collection)
    return collection


def save_trace(row: dict, path: Path, ymax_w: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.18)
    time_min, measured, predicted = plot_series(row)
    fig, axis = plt.subplots(figsize=(4.4, 2.5))
    add_alpha_gradient_line(
        axis, time_min, measured, color="black", linewidth=0.65,
        alpha_range=MEASURED_ALPHA_RANGE, label="Measured",
    )
    add_alpha_gradient_line(
        axis, time_min, predicted, color=STANFORD_RED, linewidth=0.9,
        alpha_range=PREDICTED_ALPHA_RANGE, label="PowerTrace-Sim",
    )
    axis.set(xlabel="Time (min)", ylabel="Power per GPU (W)",
             xlim=(0.0, TRACE_SECONDS / 60.0), ylim=(0.0, ymax_w))
    axis.grid(True, alpha=0.25)
    legend_handles = [
        Line2D([], [], color="black", linewidth=0.65, label="Measured"),
        Line2D([], [], color=STANFORD_RED, linewidth=0.9,
               label="PowerTrace-Sim"),
    ]
    axis.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        frameon=False, ncol=2, fontsize=8, handlelength=1.5,
        columnspacing=0.8, borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.35, rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = load_cache()
    artifact = json.loads(ARTIFACT_PATH.read_text())
    prediction = predict_dense_node_power(cache, artifact)
    selection = {
        "hardware": TARGET_HARDWARE,
        "model": TARGET_MODEL,
        "tp": TARGET_TP,
    }
    rows, candidates = [], {}
    for rate in PLOT_RATES:
        row, rate_candidates = comparison_row(
            cache, prediction, selection, rate
        )
        rows.append(row)
        candidates[f"{rate:g}"] = rate_candidates
    ymax_w = 1.05 * max(
        np.max(series)
        for row in rows for series in plot_series(row)[1:]
    )
    paths = []
    for row in rows:
        path = plot_path(selection, row["rate"])
        save_trace(row, path, ymax_w)
        paths.append(path)
        print(path)
    with CSV_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "rate_requests_s", "run_id", "time_s",
            "measured_w_per_gpu", "predicted_w_per_gpu",
        ))
        writer.writeheader()
        for row in rows:
            time_min, measured, predicted = plot_series(row)
            writer.writerows({
                "rate_requests_s": row["rate"],
                "run_id": row["run_id"],
                "time_s": float(time_value * 60.0),
                "measured_w_per_gpu": float(measured_value),
                "predicted_w_per_gpu": float(predicted_value),
            } for time_value, measured_value, predicted_value in zip(
                time_min, measured, predicted
            ))
    print(CSV_PATH)
    REPORT_PATH.write_text(json.dumps({
        "selection_rule": "Llama-3 8B H100 TP1 fixed by user; repetitions "
                          "chosen by the three-metric medoid at each rate; "
                          "clean dense surface candidate",
        "inputs": {
            "cache": str(CACHE_PATH),
            "cache_sha256": sha256_file(CACHE_PATH),
            "artifact": str(ARTIFACT_PATH),
            "artifact_sha256": sha256_file(ARTIFACT_PATH),
        },
        "selection_rates": list(SELECTION_RATES),
        "selection": selection,
        "trace_seconds": TRACE_SECONDS,
        "plot_aggregation": "matched nonoverlapping one-second means",
        "data_file": str(CSV_PATH),
        "power_scale": "per_gpu",
        "prediction_color": STANFORD_RED,
        "line_alpha_gradient": {
            "measured": list(MEASURED_ALPHA_RANGE),
            "predicted": list(PREDICTED_ALPHA_RANGE),
            "direction": "left_to_right",
        },
        "legend_position": "above_axes",
        "panels": [{
            key: value for key, value in row.items()
            if key not in ("measured", "predicted")
        } for row in rows],
        "candidates": candidates,
        "files": list(map(str, paths)),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
