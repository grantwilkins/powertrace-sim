"""Small deterministic baselines for the frozen feature-test harness."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.optimize import nnls


def _mask(run_ids: np.ndarray, runs: Iterable[int]) -> np.ndarray:
    return np.isin(run_ids, list(runs))


def _causal_activity_history(
    run_ids: np.ndarray,
    activity: np.ndarray,
    delta_activity: np.ndarray,
    dt_s: float,
    taps_s: tuple[float, ...],
) -> np.ndarray:
    taps = [int(round(tap / dt_s)) for tap in taps_s]
    if any(tap < 0 for tap in taps) or not np.allclose(np.asarray(taps) * dt_s, taps_s):
        raise ValueError("taps_s must be nonnegative multiples of dt_s")
    columns = []
    for values in (np.log1p(activity), delta_activity):
        for tap in taps:
            history = np.zeros_like(values, dtype=float)
            for run in np.unique(run_ids):
                at = np.flatnonzero(run_ids == run)
                if tap == 0:
                    history[at] = values[at]
                elif tap < at.size:
                    history[at[tap:]] = values[at[:-tap]]
            columns.append(history)
    return np.column_stack(columns)


def fit_causal_activity_ridge(
    run_ids: np.ndarray,
    activity: np.ndarray,
    delta_activity: np.ndarray,
    power: np.ndarray,
    fit_runs: Iterable[int],
    *,
    dt_s: float,
    ridge: float,
    taps_s: tuple[float, ...] = (0, 1, 2, 4, 8),
) -> dict:
    """Fit standalone B1 on exact causal ``A_t`` and ``delta_A_t`` histories."""
    fit_runs = tuple(map(int, fit_runs))
    run_ids = np.asarray(run_ids)
    activity = np.asarray(activity, dtype=float)
    delta_activity = np.asarray(delta_activity, dtype=float)
    power = np.asarray(power, dtype=float)
    if not (run_ids.shape == activity.shape == delta_activity.shape == power.shape):
        raise ValueError("B1 inputs must have identical one-dimensional shapes")
    if np.any(activity < 0) or ridge < 0:
        raise ValueError("activity and ridge must be nonnegative")
    train = _mask(run_ids, fit_runs)
    if not train.any():
        raise ValueError("B1 requires at least one fitting row")
    history = _causal_activity_history(run_ids, activity, delta_activity, dt_s, taps_s)
    mean, scale = history[train].mean(0), history[train].std(0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = (history - mean) / scale
    x_mean, y_mean = x[train].mean(0), power[train].mean()
    centered = x[train] - x_mean
    beta = np.linalg.solve(
        centered.T @ centered + float(ridge) * np.eye(centered.shape[1]),
        centered.T @ (power[train] - y_mean),
    )
    return {
        "candidate": "B1",
        "taps_s": tuple(float(tap) for tap in taps_s),
        "dt_s": float(dt_s),
        "history_mean": mean,
        "history_scale": scale,
        "coefficients": beta,
        "intercept": float(y_mean - x_mean @ beta),
        "training_runs": tuple(sorted(set(fit_runs))),
        "transfer_eligible": True,
    }


def predict_causal_activity_ridge(
    run_ids: np.ndarray,
    activity: np.ndarray,
    delta_activity: np.ndarray,
    fit: dict,
) -> np.ndarray:
    history = _causal_activity_history(
        np.asarray(run_ids), np.asarray(activity, float), np.asarray(delta_activity, float),
        float(fit["dt_s"]), tuple(fit["taps_s"]),
    )
    x = (history - fit["history_mean"]) / fit["history_scale"]
    return float(fit["intercept"]) + x @ fit["coefficients"]


def fit_same_configuration_physics_oracle(
    design: np.ndarray,
    power: np.ndarray,
    tp: np.ndarray,
    busy: np.ndarray,
    run_ids: np.ndarray,
    configuration: np.ndarray,
    fit_runs: Iterable[int],
    *,
    transfer: bool = False,
    cap_quantile: float = 0.995,
) -> dict:
    """Fit B4 independently per configuration; B4 is never transfer-eligible."""
    fit_runs = tuple(map(int, fit_runs))
    if transfer:
        raise ValueError("B4 is a same-configuration oracle and is ineligible for transfer")
    design, power, tp = np.asarray(design, float), np.asarray(power, float), np.asarray(tp, float)
    busy, run_ids, configuration = np.asarray(busy, bool), np.asarray(run_ids), np.asarray(configuration)
    if design.ndim != 2 or any(x.shape != power.shape for x in (tp, busy, run_ids, configuration)):
        raise ValueError("B4 row arrays must align with the physics design")
    train = _mask(run_ids, fit_runs)
    fits = {}
    for config in np.unique(configuration[train]):
        rows = train & (configuration == config)
        scale = np.sqrt(np.mean(design[rows] ** 2, axis=0))
        keep = scale > 1e-12
        scaled, _ = nnls(design[rows][:, keep] / scale[keep], power[rows])
        coefficients = np.zeros(design.shape[1])
        coefficients[keep] = scaled / scale[keep]
        cap_rows = rows & busy
        if not cap_rows.any():
            raise ValueError(f"B4 configuration {config!r} has no busy fitting rows")
        fits[str(config)] = {
            "coefficients": coefficients,
            "cap_w_per_gpu": float(np.quantile(power[cap_rows] / tp[cap_rows], cap_quantile)),
        }
    return {
        "candidate": "B4",
        "configurations": fits,
        "training_runs": tuple(sorted(set(fit_runs))),
        "cap_quantile": float(cap_quantile),
        "transfer_eligible": False,
    }


def predict_same_configuration_physics_oracle(
    design: np.ndarray,
    tp: np.ndarray,
    configuration: np.ndarray,
    fit: dict,
) -> np.ndarray:
    design, tp, configuration = np.asarray(design, float), np.asarray(tp, float), np.asarray(configuration)
    predicted = np.empty(design.shape[0], dtype=float)
    for config in np.unique(configuration):
        if str(config) not in fit["configurations"]:
            raise ValueError(f"B4 cannot predict unseen configuration {config!r}")
        rows = configuration == config
        params = fit["configurations"][str(config)]
        predicted[rows] = np.minimum(
            design[rows] @ params["coefficients"], params["cap_w_per_gpu"] * tp[rows]
        )
    return predicted
