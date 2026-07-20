"""Training-only slow-thermal-state ablation for the arrival-only power model.

The static surface and thermal coefficient are fit jointly on finite dense
training bins. The thermal time constant and meter delay are selected by
training RMSE. Held-out power is used only for the final comparison.

Usage: uv run python power-test/thermal_ablation.py
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
from power_surface import predict, surface_design, thermal_state  # noqa: E402
from response_chain import apply_chain  # noqa: E402

TAU_GRID_S = (30.0, 60.0, 120.0, 240.0, 480.0, 960.0)
HOLDOUT_ROLES = (
    "test_indomain",
    "holdout_rate",
    "holdout_twin",
    "holdout_model",
    "dtype_calibration",
)
METRICS = PRIMARY_METRICS + ("mean_bias_pct", "nrmse_mean")


def dynamic_driver(design, names, coefficients) -> np.ndarray:
    """Predicted board power above the idle and always-on fabric floor."""
    coefficients = np.asarray(coefficients, float)
    floor = (
        design[:, names.index("tp")] * coefficients[names.index("tp")]
        + design[:, names.index("tp_link")]
        * coefficients[names.index("tp_link")]
    )
    return np.maximum(design @ coefficients - floor, 0.0)


def heat_driver(driver, tp, mode: str) -> np.ndarray:
    if mode == "all_load":
        return driver
    if mode == "tp8_chassis":
        return driver * (np.asarray(tp) == 8)
    raise ValueError(f"Unknown heat-driver mode {mode!r}")


def fit_thermal_hardware(
    cache: dict, hardware: str, driver_mode: str
) -> tuple[dict, dict]:
    baseline, design, names, context = fit_hardware(cache, hardware)
    sub, slices = context
    dt = float(cache["dt_s"])
    y = sub["power"].astype(float)
    train_role = list(cache["role_names"]).index("train")
    families = np.asarray(cache["family_names"])[sub["family_idx"]].astype(str)
    train = (
        (sub["role_idx"] == train_role)
        & ~np.char.startswith(families, "moe")
        & np.isfinite(y)
    )

    candidates = []
    for delay_s in DELAY_GRID_S:
        filtered = filtered_design(design, slices, dt, hardware, delay_s)
        driver_coefficients = nnls_rms_scaled(filtered[train], y[train])
        driver = heat_driver(
            dynamic_driver(design, names, driver_coefficients),
            sub["tp"],
            driver_mode,
        )
        for tau_s in TAU_GRID_S:
            heat = thermal_state(driver, sub["run_id"], dt, tau_s)
            filtered_heat = filtered_design(
                heat[:, None], slices, dt, hardware, delay_s
            )[:, 0]
            augmented = np.column_stack([filtered, filtered_heat])
            coefficients = nnls_rms_scaled(augmented[train], y[train])
            rmse = float(
                np.sqrt(np.mean((augmented[train] @ coefficients - y[train]) ** 2))
            )
            candidates.append(
                (rmse, delay_s, tau_s, driver_coefficients, coefficients)
            )

    rmse, delay_s, tau_s, driver_coefficients, coefficients = min(
        candidates, key=lambda candidate: candidate[0]
    )
    thermal = {
        "hardware": hardware,
        "delay_s": float(delay_s),
        "tau_s": float(tau_s),
        "beta": float(coefficients[-1]),
        "coefficients": dict(zip(names, coefficients[:-1].tolist())),
        "driver_coefficients": dict(zip(names, driver_coefficients.tolist())),
        "driver_mode": driver_mode,
        "train_rmse": rmse,
        "baseline_train_rmse": float(
            baseline["delay_grid_rmse"][f"{baseline['delay_s']:g}"]
        ),
        "fit_population": "finite dense training bins only",
    }
    return baseline, thermal


def thermal_prediction(d, hardware, names, fit, dt) -> np.ndarray:
    design, actual_names = surface_design(d, hardware)
    if actual_names != names:
        raise ValueError("Thermal fit and prediction design columns differ")
    surface_coefficients = coefficients_in_design_order(fit, names)
    driver_coefficients = coefficients_in_design_order(
        {"coefficients": fit["driver_coefficients"]}, names
    )
    driver = heat_driver(
        dynamic_driver(design, names, driver_coefficients),
        d["tp"],
        fit["driver_mode"],
    )
    heat = thermal_state(driver, np.zeros(driver.size), dt, fit["tau_s"])
    raw = design @ surface_coefficients + fit["beta"] * heat
    return apply_chain(raw, dt, hardware, fit["delay_s"])


def evaluate(cache: dict, fits: dict[str, tuple[dict, dict]]) -> list[dict]:
    rows = []
    role_names = list(map(str, cache["role_names"]))
    scored = {role_names.index(role) for role in HOLDOUT_ROLES}
    dt = float(cache["dt_s"])
    for hardware, (baseline, thermal) in fits.items():
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
                if isinstance(value, np.ndarray) and value.shape == sub["run_id"].shape
            }
            baseline_power = apply_chain(
                predict(
                    design[lo:hi],
                    baseline_coefficients,
                    sub["tp"][lo:hi],
                    hardware,
                ),
                dt,
                hardware,
                baseline["delay_s"],
            )
            thermal_power = thermal_prediction(
                run_data, hardware, names, thermal, dt
            )
            family = str(cache["family_names"][sub["family_idx"][lo]])
            model = str(cache["model_names"][sub["model_idx"][lo]])
            row = {
                "hardware": hardware,
                "role": role_names[role_idx],
                "run_id": run_id,
                "model": model,
                "tp": int(sub["tp"][lo]),
                "rate": float(sub["rate"][lo]),
                "dense": not family.startswith("moe"),
            }
            for method, power in (
                ("baseline", baseline_power),
                ("thermal", thermal_power),
            ):
                metrics = trace_metrics(measured, power, native_dt=dt)
                row[method] = {
                    metric: float(metrics.get(metric, float("nan")))
                    for metric in METRICS
                }
            rows.append(row)
    return rows


def summarize(rows, selector) -> dict:
    selected = [row for row in rows if selector(row)]
    summary = {"runs": len(selected)}
    for method in ("baseline", "thermal"):
        summary[method] = {
            metric: float(
                np.nanmedian([row[method][metric] for row in selected])
            )
            for metric in ("energy_error_pct", "acf_mae", "nrmse_range")
        }
    return summary


def print_summary(label: str, summary: dict) -> None:
    print(f"\n{label}: {summary['runs']} runs")
    for method in ("baseline", "thermal"):
        values = summary[method]
        print(
            f"  {method:8s} energy={values['energy_error_pct']:.3f}% "
            f"acf_mae={values['acf_mae']:.4f} "
            f"nrmse_range={values['nrmse_range']:.4f}"
        )


def main() -> None:
    cache = load_cache()
    for driver_mode in ("all_load", "tp8_chassis"):
        print(f"\n==== {driver_mode} ====")
        fits = {
            hardware: fit_thermal_hardware(cache, hardware, driver_mode)
            for hardware in map(str, cache["hw_names"])
        }
        for hardware, (baseline, thermal) in fits.items():
            print(
                f"{hardware}: baseline_rmse={thermal['baseline_train_rmse']:.3f} W "
                f"thermal_rmse={thermal['train_rmse']:.3f} W "
                f"tau={thermal['tau_s']:.0f} s beta={thermal['beta']:.4f} "
                f"delay={thermal['delay_s']:.2f} s"
            )

        rows = evaluate(cache, fits)
        for hardware in map(str, cache["hw_names"]):
            local = lambda row, hw=hardware: row["hardware"] == hw and row["dense"]
            problem = lambda row, hw=hardware: (
                local(row, hw)
                and row["role"] == "holdout_rate"
                and row["tp"] == 8
                and row["rate"] == 4.0
                and "70b" in row["model"]
            )
            control = lambda row, hw=hardware: (
                local(row, hw)
                and row["role"] == "holdout_rate"
                and row["tp"] == 4
                and row["rate"] == 4.0
                and "70b" in row["model"]
            )
            print_summary(
                f"{hardware} all dense holdouts",
                summarize(rows, local),
            )
            print_summary(
                f"{hardware} rate-4 70B TP8",
                summarize(rows, problem),
            )
            print_summary(
                f"{hardware} rate-4 70B TP4 control",
                summarize(rows, control),
            )
            for row in filter(problem, rows):
                print(
                    f"  run {row['run_id']} {row['model']}: "
                    f"ACF {row['baseline']['acf_mae']:.4f}"
                    f"->{row['thermal']['acf_mae']:.4f}, "
                    f"NRMSE {row['baseline']['nrmse_range']:.4f}"
                    f"->{row['thermal']['nrmse_range']:.4f}"
                )


if __name__ == "__main__":
    main()
