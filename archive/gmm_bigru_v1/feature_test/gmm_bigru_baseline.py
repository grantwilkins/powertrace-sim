"""Frozen same-configuration B2 GMM+BiGRU baseline for S0."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from model.classifiers.gmm_bigru import build_state_labels, fit_power_gmm
from model.classifiers.trace_generation import generate_gmm_bigru_trace
from model.pipeline.training import train_one_config


def _mask(run_ids: np.ndarray, runs: Iterable[int]) -> np.ndarray:
    return np.isin(run_ids, list(runs))


def _features(activity: np.ndarray, delta_activity: np.ndarray, mean=None, scale=None):
    raw = np.column_stack([activity, delta_activity]).astype(np.float64)
    if mean is None:
        mean, scale = raw.mean(0), raw.std(0)
        scale = np.where(scale > 1e-12, scale, 1.0)
    return ((raw - mean) / scale).astype(np.float32), mean, scale


def _fit_configuration(payload):
    config_id, run_ids, activity, delta_activity, power, train_runs, development_runs, options = payload
    torch.set_num_threads(1)
    tr, dv = _mask(run_ids, train_runs), _mask(run_ids, development_runs)
    features, mean, scale = _features(activity[tr], delta_activity[tr])
    gmm = fit_power_gmm(power[tr], k=options["k"], random_state=options["seed"])

    def traces(mask, normalized=None):
        output = []
        for run in sorted(map(int, np.unique(run_ids[mask]))):
            rows = mask & (run_ids == run)
            x = (normalized[run_ids[mask] == run] if normalized is not None
                 else _features(activity[rows], delta_activity[rows], mean, scale)[0])
            output.append({"features_norm": x, "state_labels": build_state_labels(power[rows], gmm)})
        return output

    trained = train_one_config(
        config_id=config_id,
        config_data={"train": traces(tr, features), "val": traces(dv)},
        k=options["k"], input_dim=2, hidden_dim=options["hidden_dim"],
        num_layers=options["num_layers"], n_epochs=options["epochs"],
        lr=options["learning_rate"], patience=options["patience"],
        seed=options["seed"], device=torch.device("cpu"),
    )
    model = trained["model"].cpu().eval()
    return config_id, {
        "model": model,
        "gmm": {key: gmm[key] for key in ("k", "means", "variances", "weights")},
        "feature_mean": mean,
        "feature_scale": scale,
        "power_range": (float(power[tr].min()), float(power[tr].max())),
        "best_epoch": trained["best_epoch"],
        "best_val_loss": trained["best_val_loss"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def fit_s0_gmm_bigru(
    run_ids: np.ndarray,
    configuration: np.ndarray,
    activity: np.ndarray,
    delta_activity: np.ndarray,
    power: np.ndarray,
    train_runs: Iterable[int],
    development_runs: Iterable[int],
    *,
    k: int = 10,
    hidden_dim: int = 64,
    num_layers: int = 1,
    epochs: int = 1000,
    patience: int = 50,
    learning_rate: float = 1e-3,
    seed: int = 42,
    transfer: bool = False,
) -> dict:
    """Fit one K10-F2 BiGRU per configuration using repeat 0/1 only."""
    if transfer:
        raise ValueError("B2 is an S0 same-configuration baseline and is ineligible for transfer")
    train_runs, development_runs = tuple(map(int, train_runs)), tuple(map(int, development_runs))
    if set(train_runs) & set(development_runs):
        raise ValueError("B2 train and development runs must be disjoint")
    arrays = [np.asarray(x) for x in (run_ids, configuration, activity, delta_activity, power)]
    if any(x.ndim != 1 or x.shape != arrays[0].shape for x in arrays):
        raise ValueError("B2 inputs must be aligned one-dimensional arrays")
    run_ids, configuration = arrays[0], arrays[1].astype(str)
    activity, delta_activity, power = (np.asarray(x, float) for x in arrays[2:])
    if np.any(activity < 0) or not all(np.all(np.isfinite(x)) for x in (activity, delta_activity, power)):
        raise ValueError("B2 activity must be nonnegative and all numeric inputs finite")
    train, development = _mask(run_ids, train_runs), _mask(run_ids, development_runs)
    if not train.any() or not development.any():
        raise ValueError("B2 requires both repeat-0 training and repeat-1 development rows")

    options = {"k": k, "hidden_dim": hidden_dim, "num_layers": num_layers,
               "epochs": epochs, "patience": patience,
               "learning_rate": learning_rate, "seed": seed}
    payloads = []
    for config_id in sorted(set(configuration[train])):
        tr, dv = train & (configuration == config_id), development & (configuration == config_id)
        if not dv.any():
            raise ValueError(f"B2 configuration {config_id!r} has no development run")
        rows = tr | dv
        payloads.append((config_id, run_ids[rows], activity[rows], delta_activity[rows],
                         power[rows], train_runs, development_runs, options))
    if len(payloads) == 1:
        fitted = map(_fit_configuration, payloads)
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(payloads))) as pool:
            fitted = list(pool.map(_fit_configuration, payloads))
    fits = dict(fitted)
    return {
        "candidate": "B2",
        "configurations": fits,
        "k": int(k),
        "hidden_dim": int(hidden_dim),
        "seed": int(seed),
        "training_runs": tuple(sorted(set(train_runs))),
        "development_runs": tuple(sorted(set(development_runs))),
        "parameter_count": sum(item["parameter_count"] for item in fits.values()),
        "device": "cpu",
        "transfer_eligible": False,
    }


def predict_s0_gmm_bigru(
    run_ids: np.ndarray,
    configuration: np.ndarray,
    activity: np.ndarray,
    delta_activity: np.ndarray,
    predict_runs: Iterable[int],
    fit: dict,
    *,
    seed: int | None = None,
    decode_mode: str = "stochastic",
) -> np.ndarray:
    """Predict requested repeat-2 runs, preserving their native 250 ms rows."""
    run_ids, configuration = np.asarray(run_ids), np.asarray(configuration).astype(str)
    activity, delta_activity = np.asarray(activity, float), np.asarray(delta_activity, float)
    if not (run_ids.shape == configuration.shape == activity.shape == delta_activity.shape):
        raise ValueError("B2 prediction inputs must have identical shapes")
    predicted = np.full(activity.size, np.nan)
    base_seed = int(fit["seed"] if seed is None else seed)
    for run in sorted(map(int, predict_runs)):
        rows = run_ids == run
        if not rows.any():
            raise ValueError(f"B2 prediction run {run} is absent")
        configs = np.unique(configuration[rows])
        if configs.size != 1 or configs[0] not in fit["configurations"]:
            raise ValueError(f"B2 cannot predict unseen configuration for run {run}")
        params = fit["configurations"][configs[0]]
        features = _features(
            activity[rows], delta_activity[rows], params["feature_mean"], params["feature_scale"]
        )[0]
        with torch.no_grad():
            logits = params["model"](torch.from_numpy(features).unsqueeze(0)).squeeze(0)
        generated = generate_gmm_bigru_trace(
            logits,
            params["gmm"],
            seed=base_seed + run,
            decode_mode=decode_mode,
            clamp_range=params["power_range"],
        )
        predicted[rows] = generated["power_w"]
    return predicted
