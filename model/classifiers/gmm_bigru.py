from __future__ import annotations

from typing import Dict, Mapping, Sequence, Union

import numpy as np

from model.utils.gaussian_mixture import make_gaussian_mixture

EPS = 1e-12
ArrayLike = Union[np.ndarray, Sequence[float]]


def _as_1d_finite(values: ArrayLike, *, allow_empty: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        if allow_empty:
            return arr
        raise ValueError("Expected non-empty array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Array contains non-finite values.")
    return arr


def _label_map_from_params(gmm_params: Mapping[str, object]) -> np.ndarray:
    if "label_map" in gmm_params:
        label_map = np.asarray(gmm_params["label_map"], dtype=np.int64).reshape(-1)
        return label_map
    if "order" in gmm_params:
        order = np.asarray(gmm_params["order"], dtype=np.int64).reshape(-1)
        out = np.empty_like(order)
        out[order] = np.arange(order.size, dtype=np.int64)
        return out
    k = int(gmm_params.get("k", 0))
    return np.arange(max(0, k), dtype=np.int64)


def fit_power_gmm(
    power_values: ArrayLike,
    k: int = 10,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    reg_covar: float = 1e-6,
) -> Dict[str, object]:
    """Fit a 1D power GMM and return sorted component parameters."""
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    y = _as_1d_finite(power_values, allow_empty=False)
    if y.size < k:
        raise ValueError(f"Need at least k samples to fit GMM; got n={y.size}, k={k}")
    x = y.reshape(-1, 1)

    model = make_gaussian_mixture(
        n_components=k,
        covariance_type="full",
        random_state=int(random_state),
        n_init=int(max(1, n_init)),
        max_iter=int(max(10, max_iter)),
        reg_covar=float(max(reg_covar, 1e-12)),
    )
    model.fit(x)

    means = np.asarray(model.means_, dtype=np.float64).reshape(-1)
    order = np.argsort(means).astype(np.int64)
    label_map = np.empty((k,), dtype=np.int64)
    label_map[order] = np.arange(k, dtype=np.int64)

    cov = np.asarray(model.covariances_, dtype=np.float64)
    if cov.ndim == 3:
        variances = cov.reshape(k, -1)[:, 0]
    elif cov.ndim == 2:
        variances = cov[:, 0]
    else:
        variances = cov.reshape(k)
    variances = np.clip(variances, a_min=1e-12, a_max=None)

    weights = np.asarray(model.weights_, dtype=np.float64).reshape(-1)
    aic = float(model.aic(x))
    bic = float(model.bic(x))

    return {
        "model": model,
        "k": int(k),
        "covariance_type": "full",
        "order": order.astype(np.int64),
        "label_map": label_map.astype(np.int64),
        "means": means[order].astype(np.float64),
        "variances": variances[order].astype(np.float64),
        "weights": weights[order].astype(np.float64),
        "aic": float(aic),
        "bic": float(bic),
    }


def gmm_params_to_json_dict(gmm_params: Mapping[str, object]) -> Dict[str, object]:
    return {
        "k": int(gmm_params["k"]),
        "covariance_type": str(gmm_params.get("covariance_type", "full")),
        "means": np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1).tolist(),
        "variances": np.asarray(gmm_params["variances"], dtype=np.float64)
        .reshape(-1)
        .tolist(),
        "weights": np.asarray(gmm_params["weights"], dtype=np.float64)
        .reshape(-1)
        .tolist(),
        "order": np.asarray(gmm_params["order"], dtype=np.int64).reshape(-1).tolist(),
        "label_map": np.asarray(gmm_params["label_map"], dtype=np.int64)
        .reshape(-1)
        .tolist(),
        "aic": float(gmm_params.get("aic", float("nan"))),
        "bic": float(gmm_params.get("bic", float("nan"))),
    }


def load_gmm_params_json_dict(payload: Mapping[str, object]) -> Dict[str, object]:
    means = np.asarray(payload.get("means", []), dtype=np.float64).reshape(-1)
    variances = np.asarray(payload.get("variances", []), dtype=np.float64).reshape(-1)
    weights = np.asarray(payload.get("weights", []), dtype=np.float64).reshape(-1)
    order = np.asarray(payload.get("order", []), dtype=np.int64).reshape(-1)
    label_map = np.asarray(payload.get("label_map", []), dtype=np.int64).reshape(-1)
    k = int(payload.get("k", means.size))
    if means.size != k or variances.size != k or weights.size != k:
        raise ValueError("Invalid GMM payload: array lengths must match k.")
    if order.size != k and order.size != 0:
        raise ValueError("Invalid GMM payload: order length must match k.")
    if label_map.size != k and label_map.size != 0:
        raise ValueError("Invalid GMM payload: label_map length must match k.")
    if order.size == 0:
        order = np.arange(k, dtype=np.int64)
    if label_map.size == 0:
        label_map = np.arange(k, dtype=np.int64)

    return {
        "k": int(k),
        "covariance_type": str(payload.get("covariance_type", "full")),
        "means": means.astype(np.float64),
        "variances": np.clip(variances, a_min=1e-12, a_max=None).astype(np.float64),
        "weights": weights.astype(np.float64),
        "order": order.astype(np.int64),
        "label_map": label_map.astype(np.int64),
        "aic": float(payload.get("aic", float("nan"))),
        "bic": float(payload.get("bic", float("nan"))),
    }


def build_state_labels(
    power_values: ArrayLike, gmm_params: Mapping[str, object]
) -> np.ndarray:
    """Predict sorted GMM state labels for 1D power targets."""
    y = _as_1d_finite(power_values, allow_empty=True)
    if y.size == 0:
        return np.zeros((0,), dtype=np.int64)

    model = gmm_params.get("model")
    if model is None or not hasattr(model, "predict"):
        raise ValueError(
            "gmm_params['model'] must be a fitted Gaussian mixture model with predict()."
        )
    raw_labels = model.predict(y.reshape(-1, 1)).astype(np.int64)
    label_map = _label_map_from_params(gmm_params)
    if label_map.size == 0:
        return raw_labels
    if np.max(raw_labels, initial=-1) >= label_map.size:
        raise ValueError("Label map smaller than predicted labels.")
    return label_map[raw_labels].astype(np.int64)


def predict_sorted_gmm_labels_from_params(
    power_values: np.ndarray, gmm_params: Mapping[str, object]
) -> np.ndarray:
    y = np.asarray(power_values, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.zeros((0,), dtype=np.int64)

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.clip(
        np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1),
        a_min=1e-12,
        a_max=None,
    )
    weights = np.asarray(
        gmm_params.get("weights", np.ones_like(means)), dtype=np.float64
    ).reshape(-1)
    if means.size == 0:
        raise ValueError("GMM means are empty")
    if variances.size != means.size or weights.size != means.size:
        raise ValueError("GMM parameter shape mismatch")

    weights = np.clip(weights, a_min=1e-12, a_max=None)
    weights = weights / np.sum(weights)

    x = y.reshape(-1, 1)
    log_norm = -0.5 * (
        np.log(2.0 * np.pi * variances).reshape(1, -1)
        + ((x - means.reshape(1, -1)) ** 2) / variances.reshape(1, -1)
    )
    log_prob = log_norm + np.log(weights).reshape(1, -1)
    return np.argmax(log_prob, axis=1).astype(np.int64)
