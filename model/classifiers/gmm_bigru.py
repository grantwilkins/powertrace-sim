from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from model.classifiers.feature_utils import (
    compute_delta_active_requests,
    compute_inference_features,
    normalize_delta_active_requests,
)

EPS = 1e-12
ArrayLike = Union[np.ndarray, Sequence[float]]


def _validate_feature_set(feature_set: str) -> str:
    value = str(feature_set).strip().lower()
    if value not in {"f2", "f3"}:
        raise ValueError(f"feature_set must be one of {{'f2','f3'}}; got {feature_set}")
    return value


def _safe_std(value: float) -> float:
    out = float(value)
    if (not np.isfinite(out)) or out <= 0.0:
        return 1e-6
    return max(out, 1e-6)


def _as_1d_finite(values: ArrayLike, *, allow_empty: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        if allow_empty:
            return arr
        raise ValueError("Expected non-empty array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Array contains non-finite values.")
    return arr


def _extract_norm_value(norm: Mapping[str, float], *keys: str) -> float:
    for key in keys:
        if key in norm:
            return float(norm[key])
    raise KeyError(f"Missing required norm key; expected one of {keys}")


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


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / np.clip(denom, a_min=EPS, a_max=None)


def _median_filter_states(states: np.ndarray, window: int) -> np.ndarray:
    z = np.asarray(states, dtype=np.int64).reshape(-1)
    n = int(z.size)
    if n == 0:
        return z

    w = int(max(1, window))
    if w < 3:
        return z.copy()
    if w % 2 == 0:
        w += 1
    half = w // 2

    out = np.zeros_like(z)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = int(np.median(z[lo:hi]))
    return out


def fit_power_gmm(
    power_values: ArrayLike,
    k: int = 10,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    reg_covar: float = 1e-6,
) -> Dict[str, object]:
    """
    Fit a 1D power GMM and return sorted component parameters.

    Returns a payload with:
    - model: fitted sklearn GaussianMixture (unsorted internal labels)
    - order: array mapping sorted_label -> original_label
    - label_map: array mapping original_label -> sorted_label
    - sorted arrays: means, variances, weights
    - aic, bic: model selection scores for this fitted K
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")

    y = _as_1d_finite(power_values, allow_empty=False)
    if y.size < k:
        raise ValueError(f"Need at least k samples to fit GMM; got n={y.size}, k={k}")
    x = y.reshape(-1, 1)

    model = GaussianMixture(
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
        "variances": np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1).tolist(),
        "weights": np.asarray(gmm_params["weights"], dtype=np.float64).reshape(-1).tolist(),
        "order": np.asarray(gmm_params["order"], dtype=np.int64).reshape(-1).tolist(),
        "label_map": np.asarray(gmm_params["label_map"], dtype=np.int64).reshape(-1).tolist(),
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


def build_state_labels(power_values: ArrayLike, gmm_params: Mapping[str, object]) -> np.ndarray:
    """
    Predict sorted GMM state labels for 1D power targets.
    """
    y = _as_1d_finite(power_values, allow_empty=True)
    if y.size == 0:
        return np.zeros((0,), dtype=np.int64)

    model = gmm_params.get("model")
    if not isinstance(model, GaussianMixture):
        raise ValueError("gmm_params['model'] must be a fitted sklearn GaussianMixture.")
    raw_labels = model.predict(y.reshape(-1, 1)).astype(np.int64)
    label_map = _label_map_from_params(gmm_params)
    if label_map.size == 0:
        return raw_labels
    if np.max(raw_labels, initial=-1) >= label_map.size:
        raise ValueError("Label map smaller than predicted labels.")
    return label_map[raw_labels].astype(np.int64)


def build_features_from_active(
    active_requests: ArrayLike,
    t_arrive_log: Optional[ArrayLike],
    norm: Mapping[str, float],
    feature_set: str = "f2",
    max_length: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Build normalized training features.

    For aligned length L:
    - A_t uses active[1:L+1]
    - ΔA_t uses active[1:L+1] - active[:L]
    - log1p(T_arrive_t) uses t_arrive_log[1:L+1] for F3
    """
    feat = _validate_feature_set(feature_set)
    active = _as_1d_finite(active_requests, allow_empty=True)
    if active.size == 0:
        d = 3 if feat == "f3" else 2
        return {
            "features_norm": np.zeros((0, d), dtype=np.float32),
            "A_raw": np.zeros((0,), dtype=np.float64),
            "A_norm": np.zeros((0,), dtype=np.float32),
            "delta_A_raw": np.zeros((0,), dtype=np.float64),
            "delta_A_norm": np.zeros((0,), dtype=np.float32),
            "t_arrive_norm": np.zeros((0,), dtype=np.float32),
        }

    L = int(max(0, active.size - 1))
    if max_length is not None:
        L = int(min(L, int(max(0, max_length))))

    t_log: Optional[np.ndarray] = None
    if feat == "f3":
        if t_arrive_log is None:
            raise ValueError("feature_set='f3' requires t_arrive_log.")
        t_log = _as_1d_finite(t_arrive_log, allow_empty=True)
        L = int(min(L, max(0, t_log.size - 1)))

    if L <= 0:
        d = 3 if feat == "f3" else 2
        return {
            "features_norm": np.zeros((0, d), dtype=np.float32),
            "A_raw": np.zeros((0,), dtype=np.float64),
            "A_norm": np.zeros((0,), dtype=np.float32),
            "delta_A_raw": np.zeros((0,), dtype=np.float64),
            "delta_A_norm": np.zeros((0,), dtype=np.float32),
            "t_arrive_norm": np.zeros((0,), dtype=np.float32),
        }

    A_raw = active[1 : L + 1]
    delta_raw = A_raw - active[:L]

    active_mean = _extract_norm_value(norm, "active_mean", "A_mean")
    active_std = _safe_std(_extract_norm_value(norm, "active_std", "A_std"))
    dA_mean = _extract_norm_value(norm, "delta_A_mean")
    dA_std = _safe_std(_extract_norm_value(norm, "delta_A_std"))

    A_norm = ((A_raw - active_mean) / active_std).astype(np.float32)
    delta_norm = normalize_delta_active_requests(delta_raw, mean=dA_mean, std=dA_std).astype(np.float32)

    cols = [A_norm, delta_norm]
    t_norm = np.zeros((L,), dtype=np.float32)
    if feat == "f3":
        assert t_log is not None  # mypy
        t_mean = _extract_norm_value(norm, "t_arrive_log_mean", "T_arrive_log_mean")
        t_std = _safe_std(_extract_norm_value(norm, "t_arrive_log_std", "T_arrive_log_std"))
        t_norm = ((t_log[1 : L + 1] - t_mean) / t_std).astype(np.float32)
        cols.append(t_norm)

    features = np.stack(cols, axis=-1).astype(np.float32)
    return {
        "features_norm": features,
        "A_raw": A_raw.astype(np.float64),
        "A_norm": A_norm,
        "delta_A_raw": delta_raw.astype(np.float64),
        "delta_A_norm": delta_norm,
        "t_arrive_norm": t_norm,
    }


def build_rollout_features_from_requests(
    requests: Sequence[Dict[str, object]],
    throughput: Mapping[str, float],
    norm: Mapping[str, float],
    T: Optional[int] = None,
    dt: float = 0.25,
    feature_set: str = "f2",
) -> Dict[str, np.ndarray]:
    """
    Build rollout-time normalized features from request logs and throughput rates.
    """
    feat = _validate_feature_set(feature_set)
    lambda_prefill = _extract_norm_value(
        throughput,
        "lambda_prefill",
        "prefill_rate_median_toks_per_s",
    )
    lambda_decode = _extract_norm_value(
        throughput,
        "lambda_decode",
        "decode_rate_median_toks_per_s",
    )
    if lambda_prefill <= 0.0 or lambda_decode <= 0.0:
        raise ValueError("Throughput rates must be positive.")

    active_mean = _extract_norm_value(norm, "active_mean", "A_mean")
    active_std = _safe_std(_extract_norm_value(norm, "active_std", "A_std"))
    t_mean = _extract_norm_value(norm, "t_arrive_log_mean", "T_arrive_log_mean")
    t_std = _safe_std(_extract_norm_value(norm, "t_arrive_log_std", "T_arrive_log_std"))
    dA_mean = _extract_norm_value(norm, "delta_A_mean")
    dA_std = _safe_std(_extract_norm_value(norm, "delta_A_std"))

    base = compute_inference_features(
        requests=requests,
        config={
            "lambda_prefill": float(lambda_prefill),
            "lambda_decode": float(lambda_decode),
            "A_mean": float(active_mean),
            "A_std": float(active_std),
            "T_arrive_log_mean": float(t_mean),
            "T_arrive_log_std": float(t_std),
        },
        T=T,
        dt=float(dt),
    )
    if base.ndim != 2 or base.shape[1] != 2:
        raise ValueError(f"Expected base features with shape (T,2); got {base.shape}")

    A_norm = np.asarray(base[:, 0], dtype=np.float32)
    t_arrive_norm = np.asarray(base[:, 1], dtype=np.float32)
    A_raw = (A_norm.astype(np.float64) * float(active_std)) + float(active_mean)
    delta_raw = compute_delta_active_requests(A_raw).astype(np.float64)
    delta_norm = normalize_delta_active_requests(delta_raw, mean=dA_mean, std=dA_std).astype(np.float32)

    cols = [A_norm, delta_norm]
    if feat == "f3":
        cols.append(t_arrive_norm)
    features = np.stack(cols, axis=-1).astype(np.float32)

    return {
        "features_norm": features,
        "A_raw": A_raw.astype(np.float64),
        "A_norm": A_norm.astype(np.float32),
        "delta_A_raw": delta_raw.astype(np.float64),
        "delta_A_norm": delta_norm.astype(np.float32),
        "t_arrive_norm": t_arrive_norm.astype(np.float32),
    }


def generate_gmm_bigru_trace(
    logits: Union[np.ndarray, torch.Tensor],
    gmm_params: Mapping[str, object],
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate a power trace from classifier logits and state-conditional Gaussian params.
    """
    if isinstance(logits, torch.Tensor):
        z = logits.detach().cpu().numpy()
    else:
        z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 2:
        raise ValueError(f"logits must have shape (T,K) or (1,T,K); got {z.shape}")

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
    K = int(means.size)
    if z.shape[1] != K:
        raise ValueError(f"logits K mismatch: got {z.shape[1]} but GMM has {K}")

    probs = _softmax_np(z)
    mode = str(decode_mode).strip().lower()
    if mode not in {"stochastic", "argmax"}:
        raise ValueError(f"decode_mode must be 'stochastic' or 'argmax'; got {decode_mode}")

    rng = np.random.default_rng(seed)
    if mode == "argmax":
        states_raw = np.argmax(probs, axis=-1).astype(np.int64)
    else:
        states_raw = np.array(
            [rng.choice(K, p=probs_t) for probs_t in probs],
            dtype=np.int64,
        )

    states = _median_filter_states(states_raw, int(median_filter_window))

    std = np.sqrt(np.clip(variances, a_min=1e-12, a_max=None))
    power = rng.normal(loc=means[states], scale=std[states]).astype(np.float64)

    if clamp_range is not None:
        lo, hi = float(clamp_range[0]), float(clamp_range[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            margin = 0.05 * (hi - lo)
            power = np.clip(power, lo - margin, hi + margin)

    return {
        "power_w": power.astype(np.float64),
        "states": states.astype(np.int64),
        "states_raw": states_raw.astype(np.int64),
        "probs": probs.astype(np.float64),
    }
