"""Rate-2 leaky-work-dose fit and frozen rate-4 temporal evaluation.

The driver is predicted dynamic node power above idle and fabric floors.
Cooling, threshold, and jump use original rate-2 training traces only. The
rate-4 power traces are read only when the frozen prediction is scored.

Usage: uv run python power-test/dose_ablation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from changepoint_ablation import (
    ROW_METRICS,
    _metrics,
    build_records,
    frozen_problem_record,
)
from fit_power_surface import load_cache

TAU_GRID_S = (120.0, 240.0, 480.0, 960.0, 1920.0, float("inf"))
PLOT_PATH = Path(__file__).resolve().with_name("dose_ablation_heldout.png")


def leaky_work_dose(
    dynamic_node_power_w: np.ndarray, *, dt_s: float, tau_s: float
) -> np.ndarray:
    """Recent dynamic node energy in joules, with exponential cooling."""
    power = np.asarray(dynamic_node_power_w, float).reshape(-1)
    if not np.all(np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("Dynamic node power must be finite and non-negative")
    if dt_s <= 0.0 or tau_s <= 0.0:
        raise ValueError("dt_s and tau_s must be positive")
    if np.isinf(tau_s):
        return np.cumsum(power) * float(dt_s)
    rho = np.exp(-float(dt_s) / float(tau_s))
    input_scale_s = float(tau_s) * (1.0 - rho)
    dose = np.empty_like(power)
    previous = 0.0
    for index, value in enumerate(power):
        previous = rho * previous + input_scale_s * value
        dose[index] = previous
    return dose


def _crossing_bin(dose: np.ndarray, threshold_j: float) -> int:
    crossings = np.flatnonzero(dose >= threshold_j)
    return int(crossings[0]) if crossings.size else len(dose)


def fit_dose_step(
    records: list[dict], *, tau_grid_s: tuple[float, ...] = TAU_GRID_S
) -> dict:
    """Fit cooling, dose threshold, and rate-scaled jump on training records."""
    if len(records) < 2:
        raise ValueError("Dose fit requires at least two training records")
    if not tau_grid_s:
        raise ValueError("Dose tau grid cannot be empty")

    candidates = []
    for tau_s in tau_grid_s:
        doses = [
            leaky_work_dose(
                record["dynamic_node_power_w"],
                dt_s=record["duration_s"] / len(record["baseline"]),
                tau_s=tau_s,
            )
            for record in records
        ]
        split_doses = [
            dose[min(record["split_bin"], len(dose) - 1)]
            for record, dose in zip(records, doses)
        ]
        errors = []
        for index, (record, dose) in enumerate(zip(records, doses)):
            threshold = float(np.median(
                split_doses[:index] + split_doses[index + 1:]
            ))
            dt_s = record["duration_s"] / len(record["baseline"])
            predicted_s = _crossing_bin(dose, threshold) * dt_s
            errors.append(abs(predicted_s - record["split_s"]))
        candidates.append((float(np.median(errors)), tau_s, split_doses))

    timing_mae_s, tau_s, split_doses = min(
        candidates, key=lambda candidate: (candidate[0], -candidate[1])
    )
    return {
        "tau_s": float(tau_s),
        "threshold_j": float(np.median(split_doses)),
        "jump_w_per_gpu_per_rate": float(np.median([
            record["jump_w"] / (record["tp"] * record["rate"])
            for record in records
        ])),
        "train_timing_mae_s": timing_mae_s,
        "timing_grid_mae_s": {
            ("cumulative" if np.isinf(tau) else f"{tau:g}"): error
            for error, tau, _ in candidates
        },
        "train_run_ids": [record["run_id"] for record in records],
    }


def apply_dose_step(record: dict, fit: dict) -> tuple[np.ndarray, float]:
    """Apply the dose-triggered, energy-centered temporal correction."""
    dt_s = record["duration_s"] / len(record["baseline"])
    dose = leaky_work_dose(
        record["dynamic_node_power_w"], dt_s=dt_s, tau_s=fit["tau_s"]
    )
    active = dose >= fit["threshold_j"]
    jump_w = (
        fit["jump_w_per_gpu_per_rate"] * record["tp"] * record["rate"]
    )
    correction = jump_w * (active.astype(float) - np.mean(active))
    return (
        record["baseline"] + correction,
        _crossing_bin(dose, fit["threshold_j"]) * dt_s,
    )


def evaluate(records: list[dict]) -> list[dict]:
    rows = []
    for hardware in sorted({record["hardware"] for record in records}):
        support = [
            record
            for record in records
            if record["hardware"] == hardware
            and record["family"] == "dense-70b"
            and record["tp"] == 8
            and record["rate"] == 2.0
            and record["role"] == "train"
        ]
        fit = fit_dose_step(support)
        targets = [
            record
            for record in records
            if record["hardware"] == hardware and frozen_problem_record(record)
        ]
        for record in targets:
            dt_s = record["duration_s"] / len(record["baseline"])
            prediction, trigger_s = apply_dose_step(record, fit)
            rows.append({
                "hardware": hardware,
                "run_id": record["run_id"],
                "model": record["model"],
                "tp": record["tp"],
                "dt_s": dt_s,
                "fit": fit,
                "trigger_s": trigger_s,
                "observed_split_s": record["split_s"],
                "measured_power_w": record["measured"],
                "baseline_power_w": record["baseline"],
                "dose_power_w": prediction,
                "baseline": _metrics(
                    record["measured"], record["baseline"], dt_s
                ),
                "dose": _metrics(record["measured"], prediction, dt_s),
            })
    return rows


def _medians(rows: list[dict], method: str) -> dict:
    return {
        metric: float(np.median([row[method][metric] for row in rows]))
        for metric in ROW_METRICS
    }


def plot_rows(rows: list[dict]) -> list[dict]:
    """One frozen Llama-70B rate-4 trace per available hardware."""
    selected = []
    for hardware in sorted({row["hardware"] for row in rows}):
        matches = [
            row for row in rows
            if row["hardware"] == hardware and row["model"] == "llama-3-70b"
        ]
        if matches:
            selected.append(min(matches, key=lambda row: row["run_id"]))
    return selected


def _moving_average(values: np.ndarray, bins: int) -> np.ndarray:
    left = (bins - 1) // 2
    right = bins // 2
    padded = np.pad(np.asarray(values, float), (left, right), mode="edge")
    return np.convolve(padded, np.full(bins, 1.0 / bins), mode="valid")


def save_plot(rows: list[dict], path: Path = PLOT_PATH) -> None:
    import matplotlib.pyplot as plt

    selected = plot_rows(rows)
    if not selected:
        raise ValueError("No held-out Llama-70B rows are available to plot")
    fig, axes = plt.subplots(
        len(selected), 1, figsize=(11, 3.5 * len(selected)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, selected):
        dt_s = row["dt_s"]
        time_s = np.arange(len(row["measured_power_w"])) * dt_s
        tp = row["tp"]
        measured = row["measured_power_w"] / tp
        baseline = row["baseline_power_w"] / tp
        dose = row["dose_power_w"] / tp
        axis.plot(
            time_s, measured, color="0.75", linewidth=0.5,
            label="Measured (250 ms)",
        )
        axis.plot(
            time_s, _moving_average(measured, 20), color="black",
            linewidth=1.5, label="Measured (5 s mean)",
        )
        axis.plot(
            time_s, _moving_average(baseline, 20), color="#0072B2",
            linewidth=1.8, label="Model without dose",
        )
        axis.plot(
            time_s, _moving_average(dose, 20), color="#D55E00",
            linewidth=1.8, label="Model with dose",
        )
        axis.axvline(
            row["trigger_s"], color="#D55E00", linestyle="--", linewidth=1.2,
            label=f"Dose trigger ({row['trigger_s']:.1f} s)",
        )
        axis.axvline(
            row["observed_split_s"], color="black", linestyle=":",
            linewidth=1.2,
            label=f"Observed split ({row['observed_split_s']:.1f} s)",
        )
        axis.set_title(
            f"{row['hardware']} · Llama-3-70B · TP8 · 4 req/s · "
            f"held-out run {row['run_id']}"
        )
        axis.set_ylabel("Power per GPU (W)")
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        "Held-out power trace: workload-dose term versus original model",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = evaluate(build_records(load_cache()))
    for hardware in sorted({row["hardware"] for row in rows}):
        local = [row for row in rows if row["hardware"] == hardware]
        fit = local[0]["fit"]
        tau = "cumulative" if np.isinf(fit["tau_s"]) else f"{fit['tau_s']:.0f}s"
        triggers = [row["trigger_s"] for row in local]
        print(
            f"\n{hardware}: tau={tau} threshold={fit['threshold_j']/1e3:.1f}kJ "
            f"train_LOO_timing_MAE={fit['train_timing_mae_s']:.1f}s "
            f"rate-4_trigger={min(triggers):.1f}-{max(triggers):.1f}s"
        )
        print(f"  training timing grid: {fit['timing_grid_mae_s']}")
        for method in ("baseline", "dose"):
            values = _medians(local, method)
            print(
                f"  {method:8s} energy={values['energy_error_pct']:.3f}% "
                f"acf_mae={values['acf_mae']:.4f} "
                f"nrmse_range={values['nrmse_range']:.4f}"
            )
    save_plot(rows)
    print(f"\nSaved held-out trace comparison to {PLOT_PATH}")


if __name__ == "__main__":
    main()
