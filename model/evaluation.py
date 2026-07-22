"""Trace metrics for standard selected-model inference outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _column(path: str | Path, name: str) -> np.ndarray:
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or name not in rows[0]:
        raise ValueError(f"{path} has no {name!r} samples")
    values = np.asarray([float(row[name]) for row in rows])
    if not np.isfinite(values).all():
        raise ValueError(f"{path}:{name} contains non-finite values")
    return values


def _acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    centered = values - values.mean()
    variance = float(centered @ centered)
    if variance == 0.0:
        return np.zeros(max_lag)
    return np.asarray([
        float(centered[:-lag] @ centered[lag:] / variance)
        for lag in range(1, max_lag + 1)
    ])


def evaluate_power_csv(
    measured_csv: str | Path, predicted_csv: str | Path, *,
    measured_column: str = "node_gpu_power_w",
    predicted_column: str = "node_gpu_power_w", dt_s: float = 0.25,
) -> dict[str, float]:
    measured = _column(measured_csv, measured_column)
    predicted = _column(predicted_csv, predicted_column)
    if measured.shape != predicted.shape:
        raise ValueError("measured and predicted traces differ in length")
    energy = float(measured.sum() * dt_s)
    energy_error = 100.0 * abs(float((predicted - measured).sum() * dt_s)) / energy
    rmse = float(np.sqrt(np.mean((predicted - measured) ** 2)))
    span = float(np.ptp(measured))
    factor = int(round(1.0 / dt_s))
    n = measured.size // factor * factor
    if n < 61 * factor:
        raise ValueError("ACF evaluation requires at least 61 seconds")
    measured_1s = measured[:n].reshape(-1, factor).mean(axis=1)
    predicted_1s = predicted[:n].reshape(-1, factor).mean(axis=1)
    actual_acf = _acf(measured_1s, 60)
    predicted_acf = _acf(predicted_1s, 60)
    denominator = float(np.sum((actual_acf - actual_acf.mean()) ** 2))
    acf_r2 = 1.0 - float(np.sum((predicted_acf - actual_acf) ** 2)) / denominator
    return {
        "energy_error_pct": energy_error,
        "signed_bias_pct": 100.0 * float(predicted.mean() - measured.mean()) / float(measured.mean()),
        "rmse_w": rmse,
        "nrmse_range": rmse / span if span > 0 else float("nan"),
        "acf_mae": float(np.mean(np.abs(predicted_acf - actual_acf))),
        "acf_r2": acf_r2,
    }


def write_evaluation(metrics: dict[str, float], out_json: str | Path) -> None:
    output = Path(out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": "powertrace-evaluation-v1", "metrics": metrics,
        "soft_dtw_status": "report-only; computed by the paper evaluator",
    }, indent=2, sort_keys=True) + "\n")

