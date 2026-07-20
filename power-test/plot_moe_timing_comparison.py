"""Plot GPT-OSS power with uniform and measured-routing timing."""

from __future__ import annotations

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
    coefficients_in_design_order,
    interpolate_nan,
)
from power_surface import predict, surface_design  # noqa: E402
from response_chain import apply_chain  # noqa: E402

UNIFORM_CACHE = BASE / "sim_ledger_power_uniform_current_250ms.npz"
ROUTING_CACHE = BASE / "sim_ledger_power_routing_250ms.npz"
ARTIFACT = BASE / "fitted_surface.json"
PLOT_PATH = BASE / "gpt_oss_timing_comparison.png"
CELLS = (
    ("gpt-oss-20b", 1, 4.0),
    ("gpt-oss-20b", 2, 4.0),
    ("gpt-oss-120b", 4, 4.0),
    ("gpt-oss-120b", 8, 4.0),
)
DISPLAY_SMOOTH_S = 5.0


def load_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def representative_run_id(
    uniform: dict, routing: dict, model: str, tp: int, rate: float,
) -> int:
    """Lowest run ID present in both caches for one exact cell."""
    shared = None
    for cache in (uniform, routing):
        model_index = list(map(str, cache["model_names"])).index(model)
        selected = (
            (cache["model_idx"] == model_index)
            & (cache["tp"] == tp)
            & (cache["rate"] == rate)
        )
        run_ids = set(map(int, np.unique(cache["run_id"][selected])))
        shared = run_ids if shared is None else shared & run_ids
    if not shared:
        raise ValueError(f"No matched run for {model} TP{tp} rate {rate:g}")
    return min(shared)


def common_ground_truth(
    uniform_power: np.ndarray, routing_power: np.ndarray,
) -> np.ndarray:
    """Return the shared measured trace, rejecting clock/run mismatches."""
    n = min(len(uniform_power), len(routing_power))
    uniform = np.asarray(uniform_power[:n], float)
    routed = np.asarray(routing_power[:n], float)
    if not np.array_equal(np.isfinite(uniform), np.isfinite(routed)):
        raise ValueError("Compared caches have different measured-power gaps")
    finite = np.isfinite(uniform)
    if not np.allclose(uniform[finite], routed[finite], rtol=0.0, atol=1e-9):
        raise ValueError("Compared caches do not contain the same ground truth")
    return uniform


def _run(cache: dict, run_id: int) -> dict:
    selected = cache["run_id"] == run_id
    if not np.any(selected):
        raise ValueError(f"Run {run_id} is absent from the cache")
    return {
        key: value[selected]
        for key, value in cache.items()
        if isinstance(value, np.ndarray) and value.shape == selected.shape
    }


def _prediction(cache: dict, run_id: int, artifact: dict) -> tuple[np.ndarray, dict]:
    run = _run(cache, run_id)
    hardware = str(cache["hw_names"][int(run["hw_idx"][0])])
    design, names = surface_design(run, hardware)
    fit = artifact["per_hardware"][hardware]
    coefficients = coefficients_in_design_order(fit, names)
    raw = predict(design, coefficients, run["tp"], hardware)
    predicted = apply_chain(
        raw, float(cache["dt_s"]), hardware, float(fit["delay_s"]))
    return predicted, run


def comparison_row(
    uniform: dict, routing: dict, artifact: dict,
    model: str, tp: int, rate: float,
) -> dict:
    run_id = representative_run_id(uniform, routing, model, tp, rate)
    uniform_prediction, uniform_run = _prediction(uniform, run_id, artifact)
    routing_prediction, routing_run = _prediction(routing, run_id, artifact)
    measured = common_ground_truth(
        uniform_run["power"], routing_run["power"])
    n = len(measured)
    measured, _ = interpolate_nan(measured)
    dt_s = float(uniform["dt_s"])
    if float(routing["dt_s"]) != dt_s:
        raise ValueError("Compared caches use different time steps")
    uniform_prediction = uniform_prediction[:n]
    routing_prediction = routing_prediction[:n]
    return {
        "model": model,
        "tp": tp,
        "rate": rate,
        "run_id": run_id,
        "dt_s": dt_s,
        "measured": measured,
        "uniform": uniform_prediction,
        "routing": routing_prediction,
        "uniform_metrics": trace_metrics(
            measured, uniform_prediction, native_dt=dt_s),
        "routing_metrics": trace_metrics(
            measured, routing_prediction, native_dt=dt_s),
    }


def _moving_average(values: np.ndarray, bins: int) -> np.ndarray:
    left = (bins - 1) // 2
    right = bins // 2
    padded = np.pad(np.asarray(values, float), (left, right), mode="edge")
    return np.convolve(padded, np.full(bins, 1.0 / bins), mode="valid")


def save_plot(rows: list[dict], path: Path = PLOT_PATH) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for axis, row in zip(axes.flat, rows):
        bins = int(round(DISPLAY_SMOOTH_S / row["dt_s"]))
        time_s = np.arange(len(row["measured"])) * row["dt_s"]
        tp = row["tp"]
        axis.plot(
            time_s, row["measured"] / tp, color="0.82", linewidth=0.45,
            label="Ground truth (250 ms)",
        )
        axis.plot(
            time_s, _moving_average(row["measured"] / tp, bins),
            color="black", linewidth=1.6, label="Ground truth (5 s mean)",
        )
        axis.plot(
            time_s, _moving_average(row["uniform"] / tp, bins),
            color="#0072B2", linewidth=1.6,
            label="Without routing timing",
        )
        axis.plot(
            time_s, _moving_average(row["routing"] / tp, bins),
            color="#D55E00", linewidth=1.6,
            label="With routing timing",
        )
        uniform = row["uniform_metrics"]
        routed = row["routing_metrics"]
        axis.text(
            0.02, 0.04,
            "without: "
            f"E {uniform['energy_error_pct']:.1f}% · "
            f"ACF R² {uniform['acf_r2']:.3f} · "
            f"NRMSE {uniform['nrmse_range']:.3f}\n"
            "with:     "
            f"E {routed['energy_error_pct']:.1f}% · "
            f"ACF R² {routed['acf_r2']:.3f} · "
            f"NRMSE {routed['nrmse_range']:.3f}",
            transform=axis.transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
        )
        axis.set_title(
            f"{row['model']} · A100 · TP{tp} · 4 req/s · run {row['run_id']}")
        axis.set_ylabel("Power per GPU (W)")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False,
    )
    fig.suptitle(
        "GPT-OSS power: measured-routing timing versus uniform timing",
        fontsize=14, y=0.992,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    uniform = load_cache(UNIFORM_CACHE)
    routing = load_cache(ROUTING_CACHE)
    artifact = json.loads(ARTIFACT.read_text())
    rows = [
        comparison_row(uniform, routing, artifact, model, tp, rate)
        for model, tp, rate in CELLS
    ]
    save_plot(rows)
    print(PLOT_PATH)


if __name__ == "__main__":
    main()
