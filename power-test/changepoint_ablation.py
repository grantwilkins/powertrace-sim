"""Retrospective all-run changepoint inventory and held-out step ablation.

The physics surface is refit in memory on dense training bins. Changepoints
are detected in measured-minus-predicted residuals for diagnosis only. Step
parameters used for a test trace come exclusively from other traces.

Usage: uv run python power-test/changepoint_ablation.py
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
    coefficients_in_design_order,
    fit_hardware,
    interpolate_nan,
    load_cache,
    run_slices,
)
from power_surface import predict, surface_design  # noqa: E402
from response_chain import apply_chain  # noqa: E402
from thermal_ablation import dynamic_driver  # noqa: E402

MIN_SEGMENT_S = 120.0
ROW_METRICS = ("energy_error_pct", "acf_mae", "nrmse_range")


def best_mean_step(
    values: np.ndarray, *, dt_s: float, min_segment_s: float
) -> dict:
    """Best retrospective two-mean split under a minimum segment duration."""
    signal = np.asarray(values, float).reshape(-1)
    if not np.all(np.isfinite(signal)):
        raise ValueError("Changepoint signal must be finite")
    if dt_s <= 0.0 or min_segment_s <= 0.0:
        raise ValueError("dt_s and min_segment_s must be positive")
    minimum = int(np.ceil(min_segment_s / dt_s))
    if signal.size < 2 * minimum:
        raise ValueError("Signal is too short for two minimum-length segments")

    split = np.arange(minimum, signal.size - minimum + 1)
    sums = np.r_[0.0, np.cumsum(signal)]
    squares = np.r_[0.0, np.cumsum(signal * signal)]
    left_n = split.astype(float)
    right_n = signal.size - split
    left_sum = sums[split]
    right_sum = sums[-1] - left_sum
    left_sse = squares[split] - left_sum * left_sum / left_n
    right_sse = squares[-1] - squares[split] - right_sum * right_sum / right_n
    best_index = int(np.argmin(left_sse + right_sse))
    best_split = int(split[best_index])
    before = float(left_sum[best_index] / left_n[best_index])
    after = float(right_sum[best_index] / right_n[best_index])
    total_sse = float(np.sum((signal - np.mean(signal)) ** 2))
    step_sse = float(left_sse[best_index] + right_sse[best_index])
    explained = 0.0 if total_sse == 0.0 else 1.0 - step_sse / total_sse
    return {
        "split_bin": best_split,
        "split_s": best_split * float(dt_s),
        "before_w": before,
        "after_w": after,
        "jump_w": after - before,
        "step_r2": float(explained),
    }


def _metrics(measured, predicted, dt_s) -> dict:
    values = trace_metrics(measured, predicted, native_dt=dt_s)
    return {name: float(values.get(name, float("nan"))) for name in ROW_METRICS}


def build_records(cache: dict) -> list[dict]:
    records = []
    dt_s = float(cache["dt_s"])
    role_names = list(map(str, cache["role_names"]))
    model_names = np.asarray(cache["model_names"]).astype(str)
    family_names = np.asarray(cache["family_names"]).astype(str)
    for hardware in map(str, cache["hw_names"]):
        fit, _, _, _ = fit_hardware(cache, hardware)
        hw = cache["hw_idx"] == list(cache["hw_names"]).index(hardware)
        sub = {
            key: value[hw]
            for key, value in cache.items()
            if isinstance(value, np.ndarray) and value.shape == hw.shape
        }
        design, names = surface_design(sub, hardware)
        coefficients = coefficients_in_design_order(fit, names)
        for run_id, lo, hi in run_slices(sub["run_id"]):
            measured, _ = interpolate_nan(sub["power"][lo:hi].astype(float))
            raw = predict(
                design[lo:hi], coefficients, sub["tp"][lo:hi], hardware
            )
            dynamic_power = dynamic_driver(
                design[lo:hi], names, coefficients
            )
            baseline = apply_chain(
                raw, dt_s, hardware, float(fit["delay_s"])
            )
            residual = measured - baseline
            step = best_mean_step(
                residual, dt_s=dt_s, min_segment_s=MIN_SEGMENT_S
            )
            model_idx = int(sub["model_idx"][lo])
            family_idx = int(sub["family_idx"][lo])
            records.append(
                {
                    "hardware": hardware,
                    "role": role_names[int(sub["role_idx"][lo])],
                    "run_id": run_id,
                    "model": model_names[model_idx],
                    "family": family_names[family_idx],
                    "tp": int(sub["tp"][lo]),
                    "rate": float(sub["rate"][lo]),
                    "duration_s": (hi - lo) * dt_s,
                    "mean_baseline_w": float(np.mean(baseline)),
                    "dynamic_node_power_w": dynamic_power,
                    "measured": measured,
                    "baseline": baseline,
                    "residual": residual,
                    **step,
                }
            )
    return records


def fit_step(records: list[dict], *, relative: bool = False) -> dict:
    """Median split and segment levels from training records only."""
    if not records:
        raise ValueError("Step fit requires at least one training record")
    split_s = float(np.median([record["split_s"] for record in records]))
    before, after = [], []
    for record in records:
        split = int(round(split_s / (record["duration_s"] / len(record["residual"]))))
        split = min(max(split, 1), len(record["residual"]) - 1)
        scale = record["baseline"] if relative else 1.0
        adjusted = record["residual"] / scale
        before.append(float(np.mean(adjusted[:split])))
        after.append(float(np.mean(adjusted[split:])))
    before_level = float(np.median(before))
    after_level = float(np.median(after))
    return {
        "split_s": split_s,
        "before": before_level,
        "jump": after_level - before_level,
        "relative": relative,
        "train_run_ids": [record["run_id"] for record in records],
    }


def apply_step(record: dict, fit: dict) -> np.ndarray:
    dt_s = record["duration_s"] / len(record["baseline"])
    time_s = np.arange(len(record["baseline"])) * dt_s
    correction = fit["before"] + fit["jump"] * (time_s >= fit["split_s"])
    if fit["relative"]:
        correction = correction * record["baseline"]
    return record["baseline"] + correction


def score_folds(
    folds: list[tuple[str, list[dict], list[dict]]], *, relative: bool = False
) -> list[dict]:
    rows = []
    for label, train, test in folds:
        fit = fit_step(train, relative=relative)
        for record in test:
            dt_s = record["duration_s"] / len(record["baseline"])
            rows.append(
                {
                    "fold": label,
                    "run_id": record["run_id"],
                    "hardware": record["hardware"],
                    "model": record["model"],
                    "fit": fit,
                    "baseline": _metrics(
                        record["measured"], record["baseline"], dt_s
                    ),
                    "step": _metrics(
                        record["measured"], apply_step(record, fit), dt_s
                    ),
                }
            )
    return rows


def rate4_70b_tp8(record: dict) -> bool:
    return (
        record["family"] == "dense-70b"
        and record["tp"] == 8
        and record["rate"] == 4.0
    )


def frozen_problem_record(record: dict) -> bool:
    return (
        rate4_70b_tp8(record)
        and record["role"] == "holdout_rate"
    )


def cell_inventory(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        key = (
            record["hardware"],
            record["model"],
            record["tp"],
            record["rate"],
        )
        groups.setdefault(key, []).append(record)
    rows = []
    for key, group in groups.items():
        rows.append(
            {
                "hardware": key[0],
                "model": key[1],
                "tp": key[2],
                "rate": key[3],
                "runs": len(group),
                "split_s": float(np.median([r["split_s"] for r in group])),
                "jump_w_per_gpu": float(
                    np.median([r["jump_w"] / r["tp"] for r in group])
                ),
                "step_r2": float(np.median([r["step_r2"] for r in group])),
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(row["jump_w_per_gpu"]) * row["step_r2"],
        reverse=True,
    )


def leave_one_repeat_folds(records: list[dict]) -> list:
    folds = []
    for test in records:
        train = [
            record
            for record in records
            if record["hardware"] == test["hardware"]
            and record["model"] == test["model"]
            and record["run_id"] != test["run_id"]
        ]
        folds.append((f"repeat:{test['run_id']}", train, [test]))
    return folds


def leave_one_model_folds(records: list[dict]) -> list:
    folds = []
    for hardware in sorted({record["hardware"] for record in records}):
        local = [record for record in records if record["hardware"] == hardware]
        for model in sorted({record["model"] for record in local}):
            train = [record for record in local if record["model"] != model]
            test = [record for record in local if record["model"] == model]
            if train and test:
                folds.append((f"model:{hardware}:{model}", train, test))
    return folds


def cross_hardware_folds(records: list[dict]) -> list:
    folds = []
    for model in sorted({record["model"] for record in records}):
        local = [record for record in records if record["model"] == model]
        for hardware in sorted({record["hardware"] for record in local}):
            train = [record for record in local if record["hardware"] != hardware]
            test = [record for record in local if record["hardware"] == hardware]
            if train and test:
                folds.append((f"hardware:{model}:{hardware}", train, test))
    return folds


def metric_medians(rows: list[dict]) -> dict:
    return {
        method: {
            metric: float(np.median([row[method][metric] for row in rows]))
            for metric in ROW_METRICS
        }
        for method in ("baseline", "step")
    }


def print_scores(label: str, rows: list[dict]) -> None:
    medians = metric_medians(rows)
    print(f"\n{label}: {len(rows)} test runs")
    for method in ("baseline", "step"):
        values = medians[method]
        print(
            f"  {method:8s} energy={values['energy_error_pct']:.3f}% "
            f"acf_mae={values['acf_mae']:.4f} "
            f"nrmse_range={values['nrmse_range']:.4f}"
        )


def collateral_scores(records: list[dict], step_cells: list[dict]) -> list[dict]:
    rows = []
    for hardware in sorted({record["hardware"] for record in step_cells}):
        train = [
            record for record in step_cells if record["hardware"] == hardware
        ]
        fit = fit_step(train)
        controls = [
            record
            for record in records
            if record["hardware"] == hardware
            and record["family"].startswith("dense")
            and record["tp"] == 8
            and not rate4_70b_tp8(record)
        ]
        for record in controls:
            dt_s = record["duration_s"] / len(record["baseline"])
            rows.append(
                {
                    "hardware": hardware,
                    "run_id": record["run_id"],
                    "baseline": _metrics(
                        record["measured"], record["baseline"], dt_s
                    ),
                    "step": _metrics(
                        record["measured"], apply_step(record, fit), dt_s
                    ),
                }
            )
    return rows


def original_split_extrapolation(records: list[dict]) -> list[dict]:
    """Rate-2 training support, linear-in-rate jump, frozen rate-4 targets."""
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
        jump_w = (
            2.0
            * float(np.median([record["jump_w"] / record["tp"] for record in support]))
            * 8.0
        )
        fit = {
            "split_s": float(np.median([record["split_s"] for record in support])),
            "before": -0.5 * jump_w,
            "jump": jump_w,
            "relative": False,
            "train_run_ids": [record["run_id"] for record in support],
        }
        targets = [
            record
            for record in records
            if record["hardware"] == hardware and frozen_problem_record(record)
        ]
        for record in targets:
            dt_s = record["duration_s"] / len(record["baseline"])
            rows.append(
                {
                    "hardware": hardware,
                    "run_id": record["run_id"],
                    "fit": fit,
                    "baseline": _metrics(
                        record["measured"], record["baseline"], dt_s
                    ),
                    "step": _metrics(
                        record["measured"], apply_step(record, fit), dt_s
                    ),
                }
            )
    return rows


def main() -> None:
    records = build_records(load_cache())
    inventory = cell_inventory(records)
    print(f"Detected retrospective mean steps in {len(records)} runs.")
    print("\nTop 20 cells by |jump/GPU| x step-R2:")
    for row in inventory[:20]:
        print(
            f"  {row['hardware']} {row['model']:27s} TP{row['tp']} "
            f"rate={row['rate']:5g} n={row['runs']} "
            f"t={row['split_s']:6.1f}s jump={row['jump_w_per_gpu']:+7.2f}W/GPU "
            f"R2={row['step_r2']:.3f}"
        )

    step_cells = [record for record in records if rate4_70b_tp8(record)]
    print("\nRate-4 70B TP8 detected steps:")
    for record in step_cells:
        print(
            f"  run={record['run_id']} {record['hardware']} "
            f"{record['model']:27s} t={record['split_s']:.1f}s "
            f"jump={record['jump_w']:+.1f}W "
            f"({record['jump_w']/record['tp']:+.2f}W/GPU) "
            f"R2={record['step_r2']:.3f}"
        )

    repeat_rows = score_folds(leave_one_repeat_folds(step_cells))
    model_rows = score_folds(leave_one_model_folds(step_cells))
    hardware_rows = score_folds(
        cross_hardware_folds(step_cells), relative=True
    )
    for label, rows in (
        ("Leave-one-repeat-out", repeat_rows),
        ("Leave-one-model-out", model_rows),
        ("Cross-hardware relative correction", hardware_rows),
    ):
        print_scores(label, rows)
        for hardware in sorted({row["hardware"] for row in rows}):
            print_scores(
                f"{label} {hardware}",
                [row for row in rows if row["hardware"] == hardware],
            )
    train_split_rows = original_split_extrapolation(records)
    for hardware in sorted({row["hardware"] for row in train_split_rows}):
        print_scores(
            f"{hardware} original-train rate-2 -> frozen rate-4",
            [row for row in train_split_rows if row["hardware"] == hardware],
        )
    print_scores(
        "Ungated correction on other dense TP8 runs",
        collateral_scores(records, step_cells),
    )


if __name__ == "__main__":
    main()
