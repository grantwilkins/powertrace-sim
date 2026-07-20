"""Training-only decode-batch feature ablation for the power surface.

Adds one node-level coordinate, TP * log1p(time-averaged decode batch), and
jointly refits the existing surface on finite dense training bins. Held-out
power is read only during final scoring.

Usage: uv run python power-test/batch_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "feature-test"))

from evaluation_core import trace_metrics  # noqa: E402
from fit_power_surface import (  # noqa: E402
    DELAY_GRID_S,
    PRIMARY_METRICS,
    coefficients_in_design_order,
    filtered_design,
    fit_hardware,
    interpolate_nan,
    load_cache,
    nnls_rms_scaled,
    run_slices,
)
from power_surface import predict, surface_design  # noqa: E402
from response_chain import apply_chain  # noqa: E402

HOLDOUT_ROLES = (
    "test_indomain",
    "holdout_rate",
    "holdout_twin",
    "holdout_model",
    "dtype_calibration",
)
METRICS = PRIMARY_METRICS + ("mean_bias_pct", "nrmse_mean")
PLOT_PATH = BASE / "batch_ablation_heldout.png"


def batch_feature(d: dict) -> np.ndarray:
    """Node-level concave decode-batch utilization coordinate."""
    batch = np.asarray(d["batch"], float)
    tp = np.asarray(d["tp"], float)
    if np.any(batch < 0.0):
        raise ValueError("Decode batch cannot be negative")
    return tp * np.log1p(batch)


def fit_batch_hardware(cache: dict, hardware: str) -> tuple[dict, dict]:
    baseline, design, names, context = fit_hardware(cache, hardware)
    sub, slices = context
    dt_s = float(cache["dt_s"])
    y = sub["power"].astype(float)
    train_role = list(cache["role_names"]).index("train")
    families = np.asarray(cache["family_names"])[sub["family_idx"]].astype(str)
    train = (
        (sub["role_idx"] == train_role)
        & ~np.char.startswith(families, "moe")
        & np.isfinite(y)
    )
    augmented = np.column_stack([design, batch_feature(sub)])
    candidates = []
    for delay_s in DELAY_GRID_S:
        filtered = filtered_design(
            augmented, slices, dt_s, hardware, delay_s
        )
        coefficients = nnls_rms_scaled(filtered[train], y[train])
        rmse = float(
            np.sqrt(np.mean((filtered[train] @ coefficients - y[train]) ** 2))
        )
        candidates.append((rmse, delay_s, coefficients))
    rmse, delay_s, coefficients = min(candidates, key=lambda item: item[0])
    fit = {
        "hardware": hardware,
        "delay_s": float(delay_s),
        "coefficients": dict(zip(names, coefficients[:-1].tolist())),
        "batch_beta_w": float(coefficients[-1]),
        "train_rmse": rmse,
        "baseline_train_rmse": float(
            baseline["delay_grid_rmse"][f"{baseline['delay_s']:g}"]
        ),
        "fit_population": "finite dense training bins only",
    }
    return baseline, fit


def batch_prediction(
    d: dict, hardware: str, names: list[str], fit: dict, dt_s: float
) -> np.ndarray:
    design, actual_names = surface_design(d, hardware)
    if actual_names != names:
        raise ValueError("Batch fit and prediction design columns differ")
    coefficients = coefficients_in_design_order(fit, names)
    raw = design @ coefficients + fit["batch_beta_w"] * batch_feature(d)
    return apply_chain(raw, dt_s, hardware, fit["delay_s"])


def evaluate(cache: dict, fits: dict[str, tuple[dict, dict]]) -> list[dict]:
    rows = []
    role_names = list(map(str, cache["role_names"]))
    scored = {role_names.index(role) for role in HOLDOUT_ROLES}
    dt_s = float(cache["dt_s"])
    for hardware, (baseline, batch_fit) in fits.items():
        hw = cache["hw_idx"] == list(cache["hw_names"]).index(hardware)
        sub = {
            key: value[hw]
            for key, value in cache.items()
            if isinstance(value, np.ndarray) and value.shape == hw.shape
        }
        design, names = surface_design(sub, hardware)
        baseline_coefficients = coefficients_in_design_order(baseline, names)
        for run_id, lo, hi in run_slices(sub["run_id"]):
            role_idx = int(sub["role_idx"][lo])
            if role_idx not in scored:
                continue
            measured, _ = interpolate_nan(sub["power"][lo:hi].astype(float))
            run_data = {
                key: value[lo:hi]
                for key, value in sub.items()
                if isinstance(value, np.ndarray)
                and value.shape == sub["run_id"].shape
            }
            baseline_power = apply_chain(
                predict(
                    design[lo:hi],
                    baseline_coefficients,
                    sub["tp"][lo:hi],
                    hardware,
                ),
                dt_s,
                hardware,
                baseline["delay_s"],
            )
            augmented_power = batch_prediction(
                run_data, hardware, names, batch_fit, dt_s
            )
            family = str(cache["family_names"][sub["family_idx"][lo]])
            row = {
                "hardware": hardware,
                "role": role_names[role_idx],
                "run_id": run_id,
                "model": str(cache["model_names"][sub["model_idx"][lo]]),
                "tp": int(sub["tp"][lo]),
                "rate": float(sub["rate"][lo]),
                "dense": not family.startswith("moe"),
                "dt_s": dt_s,
                "measured_power_w": measured,
                "baseline_power_w": baseline_power,
                "batch_power_w": augmented_power,
            }
            for method, power in (
                ("baseline", baseline_power),
                ("batch", augmented_power),
            ):
                metrics = trace_metrics(measured, power, native_dt=dt_s)
                row[method] = {
                    metric: float(metrics.get(metric, float("nan")))
                    for metric in METRICS
                }
            rows.append(row)
    return rows


def summarize(rows: list[dict], selector) -> dict:
    selected = [row for row in rows if selector(row)]
    return {
        "runs": len(selected),
        **{
            method: {
                metric: float(np.nanmedian([
                    row[method][metric] for row in selected
                ]))
                for metric in ("energy_error_pct", "acf_mae", "nrmse_range")
            }
            for method in ("baseline", "batch")
        },
    }


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{label}: {summary['runs']} runs")
    for method in ("baseline", "batch"):
        values = summary[method]
        print(
            f"  {method:8s} energy={values['energy_error_pct']:.3f}% "
            f"acf_mae={values['acf_mae']:.4f} "
            f"nrmse_range={values['nrmse_range']:.4f}"
        )


def _moving_average(values: np.ndarray, bins: int) -> np.ndarray:
    left = (bins - 1) // 2
    right = bins // 2
    padded = np.pad(np.asarray(values, float), (left, right), mode="edge")
    return np.convolve(padded, np.full(bins, 1.0 / bins), mode="valid")


def save_plot(rows: list[dict], path: Path = PLOT_PATH) -> None:
    import matplotlib.pyplot as plt

    selected = []
    for hardware in ("A100", "H100"):
        matches = [
            row for row in rows
            if row["hardware"] == hardware
            and row["role"] == "holdout_rate"
            and row["model"] == "llama-3-70b"
            and row["tp"] == 8
            and row["rate"] == 4.0
        ]
        selected.append(min(matches, key=lambda row: row["run_id"]))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, row in zip(axes, selected):
        time_s = np.arange(len(row["measured_power_w"])) * row["dt_s"]
        bins = int(round(5.0 / row["dt_s"]))
        tp = row["tp"]
        axis.plot(
            time_s, row["measured_power_w"] / tp,
            color="0.75", linewidth=0.5, label="Measured (250 ms)",
        )
        for values, color, label, width in (
            (row["measured_power_w"], "black", "Measured (5 s mean)", 1.5),
            (row["baseline_power_w"], "#0072B2", "Current surface", 1.8),
            (row["batch_power_w"], "#D55E00", "Surface + decode batch", 1.8),
        ):
            axis.plot(
                time_s, _moving_average(values / tp, bins),
                color=color, linewidth=width, label=label,
            )
        axis.set_title(
            f"{row['hardware']} · Llama-3-70B · TP8 · 4 req/s · "
            f"held-out run {row['run_id']}"
        )
        axis.set_ylabel("Power per GPU (W)")
        axis.grid(alpha=0.2)
        axis.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Held-out power: one-feature decode-batch ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = load_cache()
    fits = {
        hardware: fit_batch_hardware(cache, hardware)
        for hardware in map(str, cache["hw_names"])
    }
    for hardware, (_, fit) in fits.items():
        print(
            f"{hardware}: beta_batch={fit['batch_beta_w']:.3f} W "
            f"train_RMSE={fit['baseline_train_rmse']:.3f}"
            f"->{fit['train_rmse']:.3f} W delay={fit['delay_s']:.2f}s"
        )
    rows = evaluate(cache, fits)
    for hardware in map(str, cache["hw_names"]):
        dense = lambda row, hw=hardware: (
            row["hardware"] == hw and row["dense"]
        )
        problem = lambda row, hw=hardware: (
            dense(row, hw)
            and row["role"] == "holdout_rate"
            and row["tp"] == 8
            and row["rate"] == 4.0
            and "70b" in row["model"]
        )
        control = lambda row, hw=hardware: (
            dense(row, hw)
            and not problem(row, hw)
        )
        print_summary(
            f"{hardware} rate-4 70B TP8",
            summarize(rows, problem),
        )
        print_summary(
            f"{hardware} other dense holdouts",
            summarize(rows, control),
        )
    save_plot(rows)
    print(f"\nSaved held-out comparison to {PLOT_PATH}")


if __name__ == "__main__":
    main()
