"""Plot the best validated GPT-OSS power paths against measured power."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "feature-test"))

from evaluation_core import trace_metrics  # noqa: E402
from fit_power_surface import (  # noqa: E402
    coefficients_in_design_order,
    interpolate_nan,
)
from moe_surface_core import (  # noqa: E402
    predict as predict_moe,
    surface_design as moe_design,
    validate_artifact_contract,
)
from power_surface import predict as predict_dense  # noqa: E402
from power_surface import surface_design as dense_design  # noqa: E402
from response_chain import apply_chain  # noqa: E402

CACHE_PATH = BASE / "sim_ledger_power_uniform_current_250ms.npz"
MOE_ARTIFACT_PATH = BASE / "fitted_moe_surface_v3.json"
DENSE_ARTIFACT_PATH = BASE / "fitted_surface.json"
PLOT_PATH = BASE / "gpt_oss_best_power_curves.png"
REPORT_PATH = BASE / "gpt_oss_best_power_curves.json"
DISPLAY_SMOOTH_S = 5.0
MODEL_COLORS = {
    "MoE v3": "#0072B2",
    "Frozen dense comparator": "#D55E00",
}
CELLS = (
    ("gpt-oss-20b", 1, "holdout_rate"),
    ("gpt-oss-20b", 2, "holdout_rate"),
    ("gpt-oss-120b", 4, "holdout_model"),
    ("gpt-oss-120b", 8, "holdout_model"),
)


def load_cache(path: Path = CACHE_PATH) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def names_for_rows(cache: dict, names_key: str, index_key: str) -> np.ndarray:
    return np.asarray(cache[names_key])[cache[index_key]].astype(str)


def candidate_run_ids(
    cache: dict, model: str, tp: int, rate: float, role: str,
) -> list[int]:
    models = names_for_rows(cache, "model_names", "model_idx")
    roles = names_for_rows(cache, "role_names", "role_idx")
    selected = (
        (models == model)
        & (cache["tp"] == tp)
        & (cache["rate"] == rate)
        & (roles == role)
    )
    return sorted(map(int, np.unique(cache["run_id"][selected])))


def run_view(cache: dict, run_id: int) -> dict:
    selected = cache["run_id"] == run_id
    if not selected.any():
        raise ValueError(f"Run {run_id} is absent from the cache")
    return {
        key: value[selected]
        for key, value in cache.items()
        if isinstance(value, np.ndarray) and value.shape == selected.shape
    }


def model_kind(model: str) -> str:
    if model == "gpt-oss-20b":
        return "MoE v3"
    if model == "gpt-oss-120b":
        return "Frozen dense comparator"
    raise ValueError(f"No validated GPT-OSS power path for {model}")


def prediction(
    cache: dict, run: dict, model: str, moe_artifact: dict,
    dense_artifact: dict,
) -> np.ndarray:
    hardware = str(cache["hw_names"][int(run["hw_idx"][0])])
    if model_kind(model) == "MoE v3":
        coefficients = validate_artifact_contract(moe_artifact)
        raw = predict_moe(moe_design(run), coefficients, run["tp"])
        delay_s = float(moe_artifact["response_delay_s"])
    else:
        design, names = dense_design(run, hardware)
        fit = dense_artifact["per_hardware"][hardware]
        coefficients = coefficients_in_design_order(fit, names)
        raw = predict_dense(design, coefficients, run["tp"], hardware)
        delay_s = float(fit["delay_s"])
    return apply_chain(raw, float(cache["dt_s"]), hardware, delay_s)


def median_repetition(rows: list[dict]) -> dict:
    """Choose the metric medoid, never the lowest-error repetition."""
    keys = ("energy_error_pct", "acf_mae", "nrmse_range")
    values = np.asarray([[row[key] for key in keys] for row in rows], float)
    center = np.nanmedian(values, axis=0)
    span = np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
    span[span == 0] = 1.0
    distance = np.nansum(np.abs(values - center) / span, axis=1)
    best = min(range(len(rows)), key=lambda i: (distance[i], rows[i]["run_id"]))
    return rows[best]


def comparison_row(
    cache: dict, model: str, tp: int, role: str, rate: float,
    moe_artifact: dict, dense_artifact: dict,
) -> tuple[dict, list[dict]]:
    candidates = []
    for run_id in candidate_run_ids(cache, model, tp, rate, role):
        run = run_view(cache, run_id)
        measured, _ = interpolate_nan(run["power"].astype(float))
        predicted = prediction(
            cache, run, model, moe_artifact, dense_artifact)
        metrics = trace_metrics(
            measured, predicted, native_dt=float(cache["dt_s"]))
        candidates.append({
            "run_id": run_id,
            **{key: float(metrics[key]) for key in (
                "energy_error_pct", "acf_mae", "acf_r2",
                "soft_dtw_divergence", "nrmse_range")},
        })
    if not candidates:
        raise ValueError(f"No {role} runs for {model} TP{tp} at {rate:g} req/s")
    chosen = median_repetition(candidates)
    run = run_view(cache, chosen["run_id"])
    measured, _ = interpolate_nan(run["power"].astype(float))
    predicted = prediction(cache, run, model, moe_artifact, dense_artifact)
    return {
        "model": model,
        "model_kind": model_kind(model),
        "tp": tp,
        "rate": rate,
        "role": role,
        "run_id": chosen["run_id"],
        "dt_s": float(cache["dt_s"]),
        "measured": measured,
        "predicted": predicted,
        "metrics": {key: chosen[key] for key in (
            "energy_error_pct", "acf_mae", "acf_r2",
            "soft_dtw_divergence", "nrmse_range")},
    }, candidates


def moving_average(values: np.ndarray, bins: int) -> np.ndarray:
    left = (bins - 1) // 2
    right = bins // 2
    padded = np.pad(np.asarray(values, float), (left, right), mode="edge")
    return np.convolve(padded, np.full(bins, 1.0 / bins), mode="valid")


def per_gpu(values: np.ndarray, tp: int) -> np.ndarray:
    return np.asarray(values, float) / tp


def save_plot(rows: list[dict], path: Path = PLOT_PATH) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, row in zip(axes.flat, rows):
        bins = int(round(DISPLAY_SMOOTH_S / row["dt_s"]))
        time_s = np.arange(len(row["measured"])) * row["dt_s"]
        measured = per_gpu(row["measured"], row["tp"])
        predicted = per_gpu(row["predicted"], row["tp"])
        axis.plot(
            time_s, measured, color="0.82", linewidth=0.45,
            label="Ground truth (250 ms)",
        )
        axis.plot(
            time_s, moving_average(measured, bins), color="black",
            linewidth=1.7, label="Ground truth (5 s mean)",
        )
        axis.plot(
            time_s, moving_average(predicted, bins),
            color=MODEL_COLORS[row["model_kind"]],
            linewidth=1.7, label=row["model_kind"],
        )
        metrics = row["metrics"]
        axis.text(
            0.02, 0.04,
            f"Energy error {metrics['energy_error_pct']:.2f}%\n"
            f"ACF R² {metrics['acf_r2']:.3f} · "
            f"NRMSE {metrics['nrmse_range']:.3f}",
            transform=axis.transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.8"},
        )
        axis.set_title(
            f"{row['model']} · A100 · TP{row['tp']} · "
            f"{row['rate']:g} req/s · median repetition {row['run_id']}")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Power per GPU (W)")
        axis.grid(alpha=0.2)
    by_label = {}
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        by_label.update(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(), loc="upper center",
        bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False,
    )
    fig.suptitle(
        "Best validated GPT-OSS power curves against ground truth",
        fontsize=14, y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = load_cache()
    moe_artifact = json.loads(MOE_ARTIFACT_PATH.read_text())
    dense_artifact = json.loads(DENSE_ARTIFACT_PATH.read_text())
    rows, candidates = [], {}
    for model, tp, role in CELLS:
        row, cell_candidates = comparison_row(
            cache, model, tp, role, 4.0, moe_artifact, dense_artifact)
        rows.append(row)
        candidates[f"{model}_tp{tp}"] = cell_candidates
    save_plot(rows)
    REPORT_PATH.write_text(json.dumps({
        "selection": "metric medoid among the three rate-4 repetitions",
        "display_smoothing_s": DISPLAY_SMOOTH_S,
        "panels": [{
            key: value for key, value in row.items()
            if key not in ("measured", "predicted")
        } for row in rows],
        "candidates": candidates,
    }, indent=2) + "\n")
    print(PLOT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
