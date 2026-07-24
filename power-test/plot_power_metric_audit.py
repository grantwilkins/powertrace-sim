"""Audit the frozen dense power surface over every legacy campaign run."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, rankdata, spearmanr

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [str(BASE), str(ROOT / "feature-test")]

from evaluation_core import trace_metrics  # noqa: E402
from fit_power_surface import (  # noqa: E402
    coefficients_in_design_order,
    interpolate_nan,
    run_slices,
)
from power_surface import HARDWARE, predict, surface_design  # noqa: E402
from response_chain import apply_chain  # noqa: E402

CACHE_PATH = BASE / "sim_ledger_power_250ms.npz"
ARTIFACT_PATH = BASE / "fitted_surface.json"
REPORT_PATH = BASE / "power_metric_audit.json"
RUN_CSV_PATH = BASE / "power_metric_audit_per_run.csv"
CELL_CSV_PATH = BASE / "power_metric_audit_cells.csv"

METRICS = {
    "energy_error_pct": ("Energy error (%)", "log"),
    "soft_dtw_divergence": ("Soft-DTW divergence", "log"),
    "nrmse_range": ("Range NRMSE", "log"),
    "rmse_w_per_gpu": ("RMSE (W/GPU)", "log"),
    "acf_r2": (r"ACF $R^2$", "acf"),
    "ks_agreement": ("KS agreement", "agreement"),
}
FAILURE_NAMES = {
    "energy_error_pct": "Energy error",
    "soft_dtw_divergence": "Soft-DTW",
    "nrmse_range": "NRMSE",
    "rmse_w_per_gpu": "RMSE",
    "acf_error": r"1 - ACF $R^2$",
    "ks_distance": "KS distance",
}
FEATURE_NAMES = {
    "rate": "Arrival rate",
    "tp": "Tensor parallelism",
    "measured_mean_w_per_gpu": "Measured mean power",
    "measured_range_w_per_gpu": "Measured power range",
    "busy_mean": "Busy fraction",
    "compute_util_mean": "Compute utilization",
    "memory_util_mean": "Memory utilization",
    "engine_iterations_rate_mean": "Iteration rate",
    "decode_batch_mean": "Decode batch",
    "waiting_requests_mean": "Waiting requests",
    "resident_weight_fraction": "Resident footprint",
    "residual_step_abs_w_per_gpu": "Residual step magnitude",
    "residual_step_r2": "Residual step $R^2$",
}
MODEL_LABELS = {
    "llama-3-8b": "Llama-3 8B",
    "deepseek-r1-distill-8b": "DeepSeek-R1 8B",
    "llama-3-70b": "Llama-3 70B",
    "deepseek-r1-distill-70b": "DeepSeek-R1 70B",
    "llama-3-405b": "Llama-3 405B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "gpt-oss-120b": "GPT-OSS 120B",
}
TP_MARKERS = {1: "o", 2: "s", 4: "D", 8: "^"}


def load_cache(path: Path = CACHE_PATH) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def one_second_pair(measured, predicted, native_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return matched nonoverlapping one-second means."""
    factor = int(round(1.0 / native_dt))
    if native_dt <= 0.0 or not np.isclose(factor * native_dt, 1.0):
        raise ValueError("native_dt must divide one second exactly")
    measured = np.asarray(measured, float).reshape(-1)
    predicted = np.asarray(predicted, float).reshape(-1)
    bins = min(measured.size, predicted.size) // factor
    return (
        measured[:bins * factor].reshape(bins, factor).mean(axis=1),
        predicted[:bins * factor].reshape(bins, factor).mean(axis=1),
    )


def diagnostic_metrics(measured, predicted, *, tp: int, native_dt: float) -> dict:
    """Compute all audit metrics on the same one-second per-GPU window."""
    if tp <= 0:
        raise ValueError("tp must be positive")
    base = trace_metrics(measured, predicted, native_dt=native_dt)
    y, p = one_second_pair(measured, predicted, native_dt)
    if y.size < 62:
        return {**base, "rmse_w_per_gpu": float("nan"),
                "ks_agreement": float("nan")}
    rmse = np.sqrt(np.mean(((p - y) / tp) ** 2))
    ks_distance = ks_2samp(y / tp, p / tp).statistic
    return {
        **base,
        "rmse_w_per_gpu": float(rmse),
        "ks_agreement": float(1.0 - ks_distance),
    }


def dense_predictions(cache: dict, artifact: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    output = np.full(cache["run_id"].shape, np.nan)
    feature_columns = {
        name: np.full(cache["run_id"].shape, np.nan)
        for name in ("resident_weights", "compute_linear", "memory_linear")
    }
    hardware_names = list(map(str, cache["hw_names"]))
    dt_s = float(cache["dt_s"])
    for hardware_index, hardware in enumerate(hardware_names):
        selected = cache["hw_idx"] == hardware_index
        sub = {
            key: value[selected]
            for key, value in cache.items()
            if isinstance(value, np.ndarray) and value.shape == selected.shape
        }
        design, names = surface_design(sub, hardware)
        fit = artifact["per_hardware"][hardware]
        coefficients = coefficients_in_design_order(fit, names)
        raw = predict(design, coefficients, sub["tp"], hardware)
        for _, lo, hi in run_slices(sub["run_id"]):
            raw[lo:hi] = apply_chain(
                raw[lo:hi], dt_s, hardware, float(fit["delay_s"])
            )
        output[selected] = raw
        for name in feature_columns:
            feature_columns[name][selected] = design[:, names.index(name)]
    return output, feature_columns


def _residual_step(residual: np.ndarray, dt_s: float, tp: int) -> tuple[float, float]:
    minimum = int(np.ceil(120.0 / dt_s))
    if residual.size < 2 * minimum:
        return float("nan"), float("nan")
    split = np.arange(minimum, residual.size - minimum + 1)
    sums = np.r_[0.0, np.cumsum(residual)]
    squares = np.r_[0.0, np.cumsum(residual * residual)]
    left_n, right_n = split, residual.size - split
    left_sum, right_sum = sums[split], sums[-1] - sums[split]
    sse = (
        squares[split] - left_sum * left_sum / left_n
        + squares[-1] - squares[split] - right_sum * right_sum / right_n
    )
    best = int(np.argmin(sse))
    before = left_sum[best] / left_n[best]
    after = right_sum[best] / right_n[best]
    total = np.sum((residual - residual.mean()) ** 2)
    explained = 0.0 if total == 0.0 else 1.0 - sse[best] / total
    return float(abs(after - before) / tp), float(explained)


def audit_rows(cache: dict, artifact: dict) -> list[dict]:
    predictions, design = dense_predictions(cache, artifact)
    models = list(map(str, cache["model_names"]))
    hardware = list(map(str, cache["hw_names"]))
    roles = list(map(str, cache["role_names"]))
    families = list(map(str, cache["family_names"]))
    dt_s = float(cache["dt_s"])
    rows = []
    for run_id, lo, hi in run_slices(cache["run_id"]):
        measured, interpolated = interpolate_nan(cache["power"][lo:hi].astype(float))
        predicted = predictions[lo:hi]
        tp = int(cache["tp"][lo])
        family = families[int(cache["family_idx"][lo])]
        metrics = diagnostic_metrics(measured, predicted, tp=tp, native_dt=dt_s)
        y, p = one_second_pair(measured, predicted, dt_s)
        step, step_r2 = _residual_step(measured - predicted, dt_s, tp)
        row = {
            "run_id": run_id,
            "hardware": hardware[int(cache["hw_idx"][lo])],
            "model": models[int(cache["model_idx"][lo])],
            "family": family,
            "tp": tp,
            "rate": float(cache["rate"][lo]),
            "role": roles[int(cache["role_idx"][lo])],
            "surface_supported": not family.startswith("moe"),
            "interpolated_power_bins": interpolated,
            "duration_s": (hi - lo) * dt_s,
            "measured_mean_w_per_gpu": float(y.mean() / tp),
            "predicted_mean_w_per_gpu": float(p.mean() / tp),
            "measured_range_w_per_gpu": float(np.ptp(y) / tp),
            "signed_energy_bias_pct": float(100.0 * (p.mean() - y.mean()) / y.mean()),
            "busy_mean": float(np.mean(cache["busy"][lo:hi])),
            "compute_util_mean": float(np.mean(design["compute_linear"][lo:hi] / tp)),
            "memory_util_mean": float(np.mean(design["memory_linear"][lo:hi] / tp)),
            "engine_iterations_rate_mean": float(np.mean(cache["engine_iterations_rate"][lo:hi])),
            "decode_batch_mean": float(np.mean(cache["batch"][lo:hi])),
            "waiting_requests_mean": float(np.mean(cache["waiting_requests"][lo:hi])),
            "resident_weight_fraction": float(np.mean(design["resident_weights"][lo:hi])),
            "residual_step_abs_w_per_gpu": step,
            "residual_step_r2": step_r2,
            **{key: float(metrics[key]) for key in METRICS},
        }
        rows.append(row)
    return rows


def aggregate_cells(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["hardware"], row["model"], row["family"], row["tp"], row["rate"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        record = dict(zip(("hardware", "model", "family", "tp", "rate"), key))
        record["runs"] = len(values)
        record["roles"] = ",".join(sorted({row["role"] for row in values}))
        record["surface_supported"] = values[0]["surface_supported"]
        feature_names = [name for name in FEATURE_NAMES if name not in ("rate", "tp")]
        for name in (*METRICS, "signed_energy_bias_pct", *feature_names):
            record[name] = float(np.nanmedian([row[name] for row in values]))
        output.append(record)
    return output


def failure_values(row: dict) -> dict:
    output = {
        name: row[name]
        for name in (
            "energy_error_pct", "soft_dtw_divergence", "nrmse_range",
            "rmse_w_per_gpu",
        )
        if name in row
    }
    if "acf_r2" in row:
        output["acf_error"] = 1.0 - row["acf_r2"]
    if "ks_agreement" in row:
        output["ks_distance"] = 1.0 - row["ks_agreement"]
    return output


def correlation_matrix(rows: list[dict], x_names: list[str], y_names: list[str]) -> list[list[float]]:
    matrix = []
    for x_name in x_names:
        line = []
        for y_name in y_names:
            pairs = []
            for row in rows:
                x = failure_values(row).get(x_name, row.get(x_name))
                y = failure_values(row).get(y_name, row.get(y_name))
                if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
                    pairs.append((x, y))
            values = np.asarray(pairs, float)
            correlation = float("nan")
            if values.shape[0] >= 3 and np.ptp(values[:, 0]) > 0 and np.ptp(values[:, 1]) > 0:
                correlation = float(spearmanr(values[:, 0], values[:, 1]).statistic)
            line.append(correlation)
        matrix.append(line)
    return matrix


def add_failure_scores(cells: list[dict]) -> None:
    oriented = np.asarray([
        list(failure_values(row).values()) for row in cells
    ], float)
    for column in range(oriented.shape[1]):
        finite = np.isfinite(oriented[:, column])
        oriented[finite, column] = rankdata(oriented[finite, column]) / finite.sum()
    scores = np.nanmean(oriented, axis=1)
    for row, score in zip(cells, scores):
        row["failure_percentile_mean"] = float(score)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(cells: list[dict], metric: str, path: Path,
                 *, include_comparator_legend: bool = True) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.35)
    palette = dict(zip(MODEL_LABELS, sns.color_palette("colorblind", len(MODEL_LABELS))))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.25), sharex=True, sharey=True)
    for axis, hardware in zip(axes, ("A100", "H100")):
        selected = [row for row in cells if row["hardware"] == hardware]
        for model, tp in sorted({(row["model"], row["tp"]) for row in selected}):
            group = sorted(
                [row for row in selected if row["model"] == model and row["tp"] == tp],
                key=lambda row: row["rate"],
            )
            supported = group[0]["surface_supported"]
            axis.plot(
                [row["rate"] for row in group], [row[metric] for row in group],
                color=palette[model], marker=TP_MARKERS[tp], markersize=4.5,
                linewidth=1.1, linestyle="-" if supported else ":", alpha=0.9,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks((0.125, 0.25, 0.5, 1, 2, 4), ("0.125", "0.25", "0.5", "1", "2", "4"))
        axis.set_xlabel("Arrival rate (requests/s)")
        axis.set_title(hardware, fontsize=12)
        axis.grid(True, alpha=0.25)
    label, scale = METRICS[metric]
    axes[0].set_ylabel(label)
    if scale == "log":
        axes[0].set_yscale("log")
    elif scale == "acf":
        axes[0].set_yscale("symlog", linthresh=1.0, linscale=0.7)
        axes[0].set_ylim(bottom=min(-1.0, min(row[metric] for row in cells) * 1.05), top=1.05)
    elif scale == "agreement":
        axes[0].set_ylim(0.0, 1.02)
    model_handles = [
        Line2D([0], [0], color=palette[name], label=label, linewidth=2)
        for name, label in MODEL_LABELS.items()
    ]
    tp_handles = [
        Line2D([0], [0], color="0.25", marker=marker, linestyle="none", label=f"TP{tp}")
        for tp, marker in TP_MARKERS.items()
    ]
    handles = model_handles + tp_handles
    if include_comparator_legend:
        handles.append(Line2D(
            [0], [0], color="0.25", linestyle=":", label="MoE comparator"
        ))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.19), ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(pad=0.45)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(matrix, row_names, column_names, path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper", font_scale=1.25)
    height = max(3.0, 0.38 * len(row_names) + 1.0)
    fig, axis = plt.subplots(figsize=(7.4, height))
    sns.heatmap(
        np.asarray(matrix, float), ax=axis, cmap="vlag", center=0.0,
        vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", linewidths=0.4,
        xticklabels=column_names, yticklabels=row_names,
        cbar_kws={"label": "Spearman correlation", "shrink": 0.82},
    )
    axis.tick_params(axis="x", rotation=35)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_signed_bias(cells: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", context="paper", font_scale=1.12)
    rates = sorted({row["rate"] for row in cells})
    configurations = sorted({
        (row["hardware"], row["model"], row["tp"], row["surface_supported"])
        for row in cells
    })
    lookup = {
        (row["hardware"], row["model"], row["tp"], row["rate"]):
        row["signed_energy_bias_pct"]
        for row in cells
    }
    values = np.asarray([
        [lookup[(hardware, model, tp, rate)] for rate in rates]
        for hardware, model, tp, _ in configurations
    ])
    labels = [
        f"{hardware}  {MODEL_LABELS[model]}  TP{tp}"
        + ("  [comparator]" if not supported else "")
        for hardware, model, tp, supported in configurations
    ]
    limit = max(5.0, float(np.nanpercentile(np.abs(values), 98)))
    fig, axis = plt.subplots(figsize=(7.8, 9.0))
    sns.heatmap(
        values, ax=axis, cmap="vlag", center=0.0, vmin=-limit, vmax=limit,
        annot=True, fmt=".1f", linewidths=0.35,
        xticklabels=[f"{rate:g}" for rate in rates], yticklabels=labels,
        cbar_kws={"label": "Signed energy bias (%)", "shrink": 0.72},
    )
    axis.set_xlabel("Arrival rate (requests/s)")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = load_cache()
    artifact = json.loads(ARTIFACT_PATH.read_text())
    rows = audit_rows(cache, artifact)
    cells = aggregate_cells(rows)
    add_failure_scores(cells)
    write_csv(RUN_CSV_PATH, rows)
    write_csv(CELL_CSV_PATH, cells)

    figure_paths = {}
    for metric in METRICS:
        path = BASE / f"power_metric_{metric}.pdf"
        _plot_metric(cells, metric, path)
        figure_paths[metric] = str(path)

    failure_keys = list(FAILURE_NAMES)
    supported = [row for row in rows if row["surface_supported"]]
    metric_matrix = correlation_matrix(supported, failure_keys, failure_keys)
    metric_path = BASE / "power_metric_error_correlations.pdf"
    _plot_heatmap(metric_matrix, list(FAILURE_NAMES.values()),
                  list(FAILURE_NAMES.values()), metric_path)
    feature_keys = list(FEATURE_NAMES)
    feature_matrix = correlation_matrix(supported, feature_keys, failure_keys)
    feature_path = BASE / "power_failure_feature_correlations.pdf"
    _plot_heatmap(feature_matrix, list(FEATURE_NAMES.values()),
                  list(FAILURE_NAMES.values()), feature_path)
    bias_path = BASE / "power_signed_bias_by_cell.pdf"
    _plot_signed_bias(cells, bias_path)

    worst = sorted(cells, key=lambda row: row["failure_percentile_mean"], reverse=True)
    report = {
        "schema_version": "power-metric-audit-v1",
        "scope": "all 450 legacy runs scored by the frozen dense surface",
        "support_note": (
            "Dense cells are in surface scope. GPT-OSS cells are retained only as "
            "explicitly unsupported dense-surface comparators; the separate MoE-v3 "
            "surface uses a different corrected ledger and is not mixed into this audit."
        ),
        "metric_contract": (
            "All metrics use the same matched nonoverlapping one-second window. RMSE is "
            "per GPU; normalized metrics and KS agreement are invariant to TP scaling. "
            "KS agreement is one minus the two-sample KS statistic."
        ),
        "runs": len(rows),
        "cells": len(cells),
        "supported_dense_runs": len(supported),
        "roles": {role: sum(row["role"] == role for row in rows)
                  for role in sorted({row["role"] for row in rows})},
        "figures": figure_paths | {
            "metric_error_correlations": str(metric_path),
            "failure_feature_correlations": str(feature_path),
            "signed_bias_by_cell": str(bias_path),
        },
        "metric_error_correlations_supported_dense": {
            "rows": list(FAILURE_NAMES.values()),
            "columns": list(FAILURE_NAMES.values()),
            "values": metric_matrix,
        },
        "failure_feature_correlations_supported_dense": {
            "rows": list(FEATURE_NAMES.values()),
            "columns": list(FAILURE_NAMES.values()),
            "values": feature_matrix,
        },
        "worst_cells": worst[:20],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for path in (*figure_paths.values(), metric_path, feature_path, bias_path,
                 RUN_CSV_PATH, CELL_CSV_PATH, REPORT_PATH):
        print(path)


if __name__ == "__main__":
    main()
