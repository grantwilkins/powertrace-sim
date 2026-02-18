#!/usr/bin/env python3
"""
Power trace architectural investigation runner.

Implements:
- Phase 0 diagnostics (transition matrices, feature-transition MI, BIC sweep)
- Phase 1 semi-Markov baseline + optional comparison vs BiGRU
- Phase 2 BiGRU feature ablation (F0/F1/F2/F3)
- Phase 3 soft-label training (KL with GMM responsibilities)
- Phase 4 uni vs bi GRU architecture comparison (gated)

Outputs are written under:
- results/diagnostics/
- results/semi_markov/
- results/feature_ablation/
- results/soft_labels/
- results/architecture/
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

# Non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

K_DEFAULT = 6
DT_DEFAULT = 0.25
EPS = 1e-12

FEATURE_F3 = ["prefill_tokens", "decode_tokens", "active_requests"]
FEATURE_F2 = ["active_requests", "n_new"]
FEATURE_F1 = ["active_requests"]
FEATURE_F0 = ["constant_one"]

PHASE2_FEATURE_SETS = {
    "F3": FEATURE_F3,
    "F2": FEATURE_F2,
    "F1": FEATURE_F1,
    "F0": FEATURE_F0,
}

LOGISTIC_FEATURES_DEFAULT = ["n_new", "active_requests", "prefill_tokens", "decode_tokens"]


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigKey:
    model_name: str
    hardware: str
    tp: int
    model_size_b: int

    @property
    def id(self) -> str:
        return f"model={self.model_name}|hardware={self.hardware}|tp={self.tp}|size_b={self.model_size_b}"

    @property
    def slug(self) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.model_name)
        return f"{safe}_{self.hardware.lower()}_tp{self.tp}"


@dataclass
class TraceRecord:
    trace_uid: int
    config: ConfigKey
    timestamps: np.ndarray
    power: np.ndarray
    prefill_tokens: np.ndarray
    decode_tokens: np.ndarray
    active_requests: np.ndarray
    n_new: np.ndarray
    input_tokens_in_window: np.ndarray
    output_tokens_in_window: np.ndarray
    schedule_x6: np.ndarray


@dataclass
class GMMArtifacts:
    model: GaussianMixture
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    order: np.ndarray
    inv_order: np.ndarray


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class LogisticArtifacts:
    model: LogisticRegression
    mean: np.ndarray
    std: np.ndarray
    feature_names: List[str]
    max_mi: float


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _binary_entropy_nats(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    p = float(np.mean(y))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p + EPS) + (1.0 - p) * math.log(1.0 - p + EPS))


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std <= 0.0 or b_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros((nlags + 1,), dtype=np.float64)
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        out = np.zeros((nlags + 1,), dtype=np.float64)
        out[0] = 1.0
        return out
    out = np.zeros((nlags + 1,), dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, nlags + 1):
        if lag >= len(x):
            out[lag] = 0.0
        else:
            out[lag] = float(np.dot(x[:-lag], x[lag:]) / denom)
    return out


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    n1 = a.size
    n2 = b.size
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    data = np.concatenate([a, b])
    data.sort()
    cdf1 = np.searchsorted(a, data, side="right") / max(1, n1)
    cdf2 = np.searchsorted(b, data, side="right") / max(1, n2)
    d = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic Kolmogorov p-value approximation
    en = math.sqrt((n1 * n2) / max(1.0, float(n1 + n2)))
    p_approx = 2.0 * math.exp(-2.0 * (en * d) ** 2)
    p_approx = float(min(1.0, max(0.0, p_approx)))
    return d, p_approx


def _integrate_trapezoid(y: np.ndarray, dx: float) -> float:
    """
    NumPy compatibility wrapper:
    - NumPy >= 2: np.trapezoid
    - Older NumPy: np.trapz
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    return float(np.trapz(y, dx=dx))


def _safe_mutual_info(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    if X.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(np.unique(y)) < 2:
        return np.zeros((X.shape[1],), dtype=np.float64)
    try:
        mi = mutual_info_classif(
            X,
            y,
            discrete_features=False,
            random_state=seed,
        )
        return np.asarray(mi, dtype=np.float64)
    except Exception:
        return np.zeros((X.shape[1],), dtype=np.float64)


def _safe_ks(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if a.size < 2 or b.size < 2:
        return float("nan"), float("nan")
    try:
        stat, pval = _ks_2samp(a, b)
        return float(stat), float(pval)
    except Exception:
        return float("nan"), float("nan")


def _zscore_cols(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


def _histogram_requests(
    bin_ts: np.ndarray,
    req_ts: np.ndarray,
    in_tok: np.ndarray,
    out_tok: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicates model.core.utils.histogram_requests behavior."""
    dt = np.median(np.diff(bin_ts)) if len(bin_ts) > 1 else DT_DEFAULT

    if len(bin_ts) > 1:
        if not np.all(np.diff(bin_ts) > 0):
            bin_ts = np.unique(bin_ts)
            if len(bin_ts) <= 1:
                bin_ts = np.array([0.0, dt]) if len(bin_ts) == 0 else np.array([bin_ts[0], bin_ts[0] + dt])
    else:
        bin_ts = np.array([0.0, dt]) if len(bin_ts) == 0 else np.array([bin_ts[0], bin_ts[0] + dt])

    edges = np.append(bin_ts, bin_ts[-1] + dt)

    if req_ts.size == 0:
        z = np.zeros((len(bin_ts),), dtype=np.float32)
        return z, z.copy(), z.copy()

    new_req_cnt, _ = np.histogram(req_ts, edges)
    new_in_tok, _ = np.histogram(req_ts, edges, weights=in_tok)
    new_out_tok, _ = np.histogram(req_ts, edges, weights=out_tok)
    return new_req_cnt.astype(np.float32), new_in_tok.astype(np.float32), new_out_tok.astype(np.float32)


def _canonical_model_name(npz_path: str) -> str:
    stem = Path(npz_path).stem
    prefix = "vllm-benchmark_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    model_part = stem.rsplit("_", 1)[0]

    # Normalize to weight naming conventions
    mapping = {
        "deepseek-r1-8b": "deepseek-r1-distill-8b",
        "deepseek-r1-70b": "deepseek-r1-distill-70b",
    }
    return mapping.get(model_part, model_part)


def _hardware_from_path(npz_path: str) -> str:
    stem = Path(npz_path).stem
    return stem.rsplit("_", 1)[-1].upper()


def _parse_model_size_b(model_name: str) -> int:
    match = re.search(r"-(\d+)b$", model_name)
    if match:
        return int(match.group(1))
    return -1


def _collect_npz_paths(npz_args: List[str], npz_dir: Optional[str], pattern: str) -> List[str]:
    paths: List[str] = []
    for p in npz_args:
        paths.append(os.path.expanduser(p))
    if npz_dir:
        paths.extend(sorted(glob.glob(os.path.join(os.path.expanduser(npz_dir), pattern))))
    deduped: List[str] = []
    seen = set()
    for p in paths:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _feature_array(trace: TraceRecord, feature_name: str) -> np.ndarray:
    if feature_name == "prefill_tokens":
        return trace.prefill_tokens
    if feature_name == "decode_tokens":
        return trace.decode_tokens
    if feature_name == "active_requests":
        return trace.active_requests
    if feature_name == "n_new":
        return trace.n_new
    if feature_name == "input_tokens_in_window":
        return trace.input_tokens_in_window
    if feature_name == "output_tokens_in_window":
        return trace.output_tokens_in_window
    if feature_name == "constant_one":
        return np.ones_like(trace.power, dtype=np.float64)
    raise KeyError(f"Unknown feature: {feature_name}")


def _feature_matrix(trace: TraceRecord, feature_names: Sequence[str]) -> np.ndarray:
    cols = [_feature_array(trace, f).astype(np.float64) for f in feature_names]
    return np.stack(cols, axis=1)


def _fit_gmm(y: np.ndarray, k: int, seed: int, sort_components: bool = True) -> GMMArtifacts:
    y2 = y.reshape(-1, 1).astype(np.float64)
    gm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=10,
        random_state=seed,
        reg_covar=1e-6,
    )
    gm.fit(y2)

    means_raw = gm.means_.reshape(-1)
    stds_raw = np.sqrt(gm.covariances_.reshape(-1))
    weights_raw = gm.weights_.reshape(-1)

    if sort_components:
        order = np.argsort(means_raw)
    else:
        order = np.arange(k)
    inv_order = np.zeros_like(order)
    inv_order[order] = np.arange(k)

    means = means_raw[order]
    stds = stds_raw[order]
    weights = weights_raw[order]

    return GMMArtifacts(
        model=gm,
        means=means,
        stds=stds,
        weights=weights,
        order=order,
        inv_order=inv_order,
    )


def _predict_labels(gmm_art: GMMArtifacts, y: np.ndarray) -> np.ndarray:
    raw = gmm_art.model.predict(y.reshape(-1, 1).astype(np.float64))
    return gmm_art.inv_order[raw]


def _predict_responsibilities(gmm_art: GMMArtifacts, y: np.ndarray) -> np.ndarray:
    probs_raw = gmm_art.model.predict_proba(y.reshape(-1, 1).astype(np.float64))
    return probs_raw[:, gmm_art.order]


def _compute_transition_matrix(label_seqs: Sequence[np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    counts = np.zeros((k, k), dtype=np.int64)
    occupancy = np.zeros((k,), dtype=np.int64)

    for z in label_seqs:
        if z.size == 0:
            continue
        occupancy += np.bincount(z, minlength=k)
        if z.size > 1:
            for zt, zt1 in zip(z[:-1], z[1:]):
                counts[int(zt), int(zt1)] += 1

    T = np.zeros((k, k), dtype=np.float64)
    row_sums = counts.sum(axis=1)
    for i in range(k):
        if row_sums[i] > 0:
            T[i] = counts[i] / row_sums[i]
        else:
            T[i, i] = 1.0

    diag = np.diag(T)
    sojourn = 1.0 / (1.0 - diag + 1e-12)

    occ_total = occupancy.sum()
    if occ_total > 0:
        occ_w = occupancy.astype(np.float64) / float(occ_total)
        weighted_diag = float(np.sum(occ_w * diag))

        finite_mask = np.isfinite(sojourn)
        if np.any(finite_mask):
            w_f = occ_w[finite_mask]
            w_f = w_f / (np.sum(w_f) + EPS)
            weighted_sojourn = float(np.sum(w_f * sojourn[finite_mask]))
        else:
            weighted_sojourn = float("inf")
    else:
        weighted_diag = float("nan")
        weighted_sojourn = float("nan")

    return T, counts, occupancy.astype(np.float64), sojourn, weighted_diag, weighted_sojourn


def _median_smooth_same_state(power: np.ndarray, states: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return power.copy()
    out = power.copy()
    n = len(power)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        mask = states[lo:hi] == states[i]
        if np.any(mask):
            out[i] = np.median(power[lo:hi][mask])
    return out


def _compute_fidelity_metrics(
    real_traces: Sequence[np.ndarray],
    synthetic_traces: Sequence[np.ndarray],
    dt: float,
    max_lags: int,
) -> Dict[str, float]:
    real = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in real_traces])
    syn = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in synthetic_traces])

    if real.size == 0 or syn.size == 0:
        return {
            "ks_statistic": float("nan"),
            "acf_r2": float("nan"),
            "nrmse": float("nan"),
            "p95_error_pct": float("nan"),
            "p99_error_pct": float("nan"),
            "delta_energy_pct": float("nan"),
            "abs_energy_error_pct": float("nan"),
            "num_points": 0.0,
        }

    ks_stat, _ = _ks_2samp(real, syn)

    nlags = max(1, min(max_lags, (len(real) // 2) - 1 if len(real) > 3 else 1))
    try:
        acf_real = _acf(real, nlags=nlags)
        acf_syn = _acf(syn, nlags=nlags)
        corr = _pearson_corr(acf_real, acf_syn)
        acf_r2 = float(corr * corr)
    except Exception:
        acf_r2 = float("nan")

    denom = float(np.max(real) - np.min(real))
    if denom <= 0:
        nrmse = float("nan")
    else:
        nrmse = float(np.sqrt(np.mean((real - syn) ** 2)) / denom)

    p95_r = float(np.percentile(real, 95))
    p95_s = float(np.percentile(syn, 95))
    p99_r = float(np.percentile(real, 99))
    p99_s = float(np.percentile(syn, 99))

    p95_error = float(np.abs(p95_r - p95_s) / (np.abs(p95_r) + EPS) * 100.0)
    p99_error = float(np.abs(p99_r - p99_s) / (np.abs(p99_r) + EPS) * 100.0)

    energy_real = _integrate_trapezoid(real, dx=dt)
    energy_syn = _integrate_trapezoid(syn, dx=dt)
    delta_energy = float((energy_syn - energy_real) / (np.abs(energy_real) + EPS) * 100.0)

    return {
        "ks_statistic": float(ks_stat),
        "acf_r2": acf_r2,
        "nrmse": nrmse,
        "p95_error_pct": p95_error,
        "p99_error_pct": p99_error,
        "delta_energy_pct": delta_energy,
        "abs_energy_error_pct": float(abs(delta_energy)),
        "num_points": float(len(real)),
    }


def _metric_score(row: Dict[str, float]) -> float:
    """Lower is better composite score for selection/gating."""
    return (
        float(row.get("ks_statistic", 1.0))
        + float(1.0 - row.get("acf_r2", 0.0))
        + float(row.get("nrmse", 1.0))
        + float(row.get("p95_error_pct", 100.0)) / 100.0
        + float(row.get("p99_error_pct", 100.0)) / 100.0
        + float(abs(row.get("delta_energy_pct", 100.0))) / 100.0
    )


def _split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> SplitIndices:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    if n <= 2:
        # Degenerate fallback
        n_train = max(1, n - 1)
        n_val = 0
    else:
        n_train = max(1, int(round(n * train_frac)))
        n_val = int(round(n * val_frac))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        if n_train >= n:
            n_train = n - 1

    n_test = n - n_train - n_val
    if n_test <= 0 and n > 1:
        if n_val > 0:
            n_val -= 1
        else:
            n_train -= 1
        n_test = 1

    train = np.sort(idx[:n_train])
    val = np.sort(idx[n_train : n_train + n_val])
    test = np.sort(idx[n_train + n_val :])
    return SplitIndices(train=train, val=val, test=test)


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError(
            "PyTorch import failed. Phases 2/3/4 and BiGRU comparison in Phase 1 require PyTorch."
        ) from exc
    return torch, nn, F


def _resolve_device(device_arg: Optional[str], requires_torch: bool):
    if not requires_torch:
        return None
    torch, _, _ = _import_torch()
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_trace_groups(npz_paths: Sequence[str]) -> Tuple[Dict[str, ConfigKey], Dict[str, List[TraceRecord]], int]:
    configs: Dict[str, ConfigKey] = {}
    traces_by_config: Dict[str, List[TraceRecord]] = {}

    uid = 0
    skipped = 0

    for npz_path in npz_paths:
        d = np.load(npz_path, allow_pickle=True)
        n = d["timestamps"].shape[0]

        model_name_default = _canonical_model_name(npz_path)
        hw_default = _hardware_from_path(npz_path)

        for i in range(n):
            ts_raw = np.asarray(d["timestamps"][i])
            mask = ts_raw > 0
            if np.sum(mask) < 2:
                skipped += 1
                continue

            timestamps = ts_raw[mask].astype(np.float64)
            power = np.asarray(d["power_traces"][i])[mask].astype(np.float64)
            prefill = np.asarray(d["prefill_tokens"][i])[mask].astype(np.float64)
            decode = np.asarray(d["decode_tokens"][i])[mask].astype(np.float64)
            active = np.asarray(d["active_requests"][i])[mask].astype(np.float64)

            req_ts = np.asarray(d["request_timestamps"][i]).astype(np.float64) if "request_timestamps" in d else np.array([], dtype=np.float64)
            if req_ts.size > 0:
                req_mask = req_ts > 0
                req_ts = req_ts[req_mask]
                in_tok = np.asarray(d["input_tokens"][i]).astype(np.float64)[req_mask] if "input_tokens" in d else np.zeros_like(req_ts)
                out_tok = np.asarray(d["output_tokens"][i]).astype(np.float64)[req_mask] if "output_tokens" in d else np.zeros_like(req_ts)
            else:
                in_tok = np.array([], dtype=np.float64)
                out_tok = np.array([], dtype=np.float64)

            n_new, in_hist, out_hist = _histogram_requests(
                bin_ts=timestamps,
                req_ts=req_ts,
                in_tok=in_tok,
                out_tok=out_tok,
            )

            x6 = np.stack(
                [
                    n_new,
                    in_hist,
                    out_hist,
                    active,
                    prefill,
                    decode,
                ],
                axis=1,
            ).astype(np.float64)
            x6 = _zscore_cols(x6)

            tp = int(d["tensor_parallelism"][i]) if "tensor_parallelism" in d else 1
            if "hardware" in d:
                hw_value = str(d["hardware"][i]).upper()
            else:
                hw_value = hw_default

            if "model_sizes" in d:
                model_size_b = int(d["model_sizes"][i])
            else:
                model_size_b = _parse_model_size_b(model_name_default)

            config = ConfigKey(
                model_name=model_name_default,
                hardware=hw_value,
                tp=tp,
                model_size_b=model_size_b,
            )
            config_id = config.id

            configs[config_id] = config
            traces_by_config.setdefault(config_id, []).append(
                TraceRecord(
                    trace_uid=uid,
                    config=config,
                    timestamps=timestamps,
                    power=power,
                    prefill_tokens=prefill,
                    decode_tokens=decode,
                    active_requests=active,
                    n_new=np.asarray(n_new, dtype=np.float64),
                    input_tokens_in_window=np.asarray(in_hist, dtype=np.float64),
                    output_tokens_in_window=np.asarray(out_hist, dtype=np.float64),
                    schedule_x6=np.asarray(x6, dtype=np.float64),
                )
            )
            uid += 1

    return configs, traces_by_config, skipped


# -----------------------------------------------------------------------------
# Phase 0: Diagnostics
# -----------------------------------------------------------------------------


def run_phase0(
    configs: Dict[str, ConfigKey],
    traces_by_config: Dict[str, List[TraceRecord]],
    out_dir: str,
    k_states: int,
    bic_k_values: Sequence[int],
    seed: int,
) -> None:
    _ensure_dir(out_dir)

    transition_json: Dict[str, object] = {}
    transition_rows: List[Dict[str, object]] = []
    mi_rows: List[Dict[str, object]] = []
    mi_dest_rows: List[Dict[str, object]] = []
    bic_rows: List[Dict[str, object]] = []

    print("\n[Phase 0A] Empirical transition matrices")

    for config_id, traces in sorted(traces_by_config.items()):
        cfg = configs[config_id]
        y_all = np.concatenate([t.power for t in traces], axis=0)
        gmm_art = _fit_gmm(y_all, k=k_states, seed=seed, sort_components=True)

        label_seqs = [_predict_labels(gmm_art, t.power) for t in traces]
        T, _, occupancy, sojourn, weighted_diag, weighted_sojourn = _compute_transition_matrix(label_seqs, k_states)

        transition_json[config_id] = {
            "model_name": cfg.model_name,
            "hardware": cfg.hardware,
            "tp": cfg.tp,
            "model_size_b": cfg.model_size_b,
            "state_means": gmm_art.means.tolist(),
            "state_stds": gmm_art.stds.tolist(),
            "transition_matrix": T.tolist(),
            "state_occupancy": occupancy.tolist(),
            "diag": np.diag(T).tolist(),
            "sojourn": sojourn.tolist(),
            "weighted_diag_mean": weighted_diag,
            "weighted_sojourn": weighted_sojourn,
        }

        row = {
            "config_id": config_id,
            "model_name": cfg.model_name,
            "hardware": cfg.hardware,
            "tp": cfg.tp,
            "model_size_b": cfg.model_size_b,
            "num_traces": len(traces),
            "weighted_diag_mean": weighted_diag,
            "weighted_sojourn": weighted_sojourn,
        }
        for i in range(k_states):
            row[f"diag_s{i}"] = float(T[i, i])
            row[f"sojourn_s{i}"] = float(sojourn[i])
            row[f"occupancy_s{i}"] = float(occupancy[i])
        transition_rows.append(row)

        print(
            f"  {cfg.slug:40s}  diag_w={weighted_diag:.4f}  sojourn_w={weighted_sojourn:.2f}"
        )

        # ------------------------------------------------------------------
        # Phase 0B: Feature-transition MI
        # ------------------------------------------------------------------
        feature_names = ["prefill_tokens", "decode_tokens", "active_requests", "n_new"]

        X_blocks = []
        y_transition_blocks = []
        z_curr_blocks = []

        for t, z in zip(traces, label_seqs):
            if len(z) < 2:
                continue
            X = np.stack([_feature_array(t, fn)[1:] for fn in feature_names], axis=1)
            did_transition = (z[1:] != z[:-1]).astype(int)
            X_blocks.append(X)
            y_transition_blocks.append(did_transition)
            z_curr_blocks.append(z[1:])

        if not X_blocks:
            continue

        X_all = np.concatenate(X_blocks, axis=0)
        y_tr = np.concatenate(y_transition_blocks, axis=0)
        z_curr = np.concatenate(z_curr_blocks, axis=0)

        mi = _safe_mutual_info(X_all, y_tr, seed=seed)
        entropy = _binary_entropy_nats(y_tr)
        transition_rate = float(np.mean(y_tr)) if y_tr.size else 0.0

        for j, fname in enumerate(feature_names):
            values_all = X_all[:, j]
            values_trans = values_all[y_tr == 1]
            ks_stat, ks_pval = _safe_ks(values_trans, values_all)
            row = {
                "config_id": config_id,
                "model_name": cfg.model_name,
                "hardware": cfg.hardware,
                "tp": cfg.tp,
                "model_size_b": cfg.model_size_b,
                "feature": fname,
                "mi_nats": float(mi[j]),
                "did_transition_entropy_nats": float(entropy),
                "normalized_mi": float(mi[j] / (entropy + EPS)),
                "transition_rate": transition_rate,
                "ks_statistic_transition_vs_all": ks_stat,
                "ks_pvalue_transition_vs_all": ks_pval,
                "mean_when_transition": float(np.mean(values_trans)) if values_trans.size else float("nan"),
                "mean_overall": float(np.mean(values_all)) if values_all.size else float("nan"),
            }
            mi_rows.append(row)

        for dest in range(k_states):
            y_dest = ((y_tr == 1) & (z_curr == dest)).astype(int)
            mi_dest = _safe_mutual_info(X_all, y_dest, seed=seed)
            event_rate = float(np.mean(y_dest)) if y_dest.size else 0.0
            for j, fname in enumerate(feature_names):
                mi_dest_rows.append(
                    {
                        "config_id": config_id,
                        "model_name": cfg.model_name,
                        "hardware": cfg.hardware,
                        "tp": cfg.tp,
                        "model_size_b": cfg.model_size_b,
                        "destination_state": dest,
                        "is_prefill_state": int(dest in (4, 5)),
                        "feature": fname,
                        "mi_nats": float(mi_dest[j]),
                        "event_rate": event_rate,
                    }
                )

        top_idx = int(np.argmax(mi)) if mi.size else 0
        print(
            f"    MI top feature={feature_names[top_idx]} mi={float(mi[top_idx]):.4f} nats transition_rate={transition_rate:.3f}"
        )

        # ------------------------------------------------------------------
        # Phase 0C: BIC sweep for K
        # ------------------------------------------------------------------
        bic_vals = {}
        y2 = y_all.reshape(-1, 1).astype(np.float64)
        for k in bic_k_values:
            try:
                gm = GaussianMixture(
                    n_components=int(k),
                    covariance_type="diag",
                    n_init=5,
                    random_state=seed,
                    reg_covar=1e-6,
                )
                gm.fit(y2)
                bic = float(gm.bic(y2))
                bic_vals[k] = bic
            except Exception:
                bic_vals[k] = float("nan")

        finite = {k: v for k, v in bic_vals.items() if np.isfinite(v)}
        k_opt = min(finite, key=finite.get) if finite else None

        for k in bic_k_values:
            bic_rows.append(
                {
                    "config_id": config_id,
                    "model_name": cfg.model_name,
                    "hardware": cfg.hardware,
                    "tp": cfg.tp,
                    "model_size_b": cfg.model_size_b,
                    "k": int(k),
                    "bic": bic_vals[k],
                    "is_optimal": int(k_opt == k) if k_opt is not None else 0,
                }
            )

        if k_opt is not None:
            bic_k6 = bic_vals.get(6, float("nan"))
            print(f"    BIC optimal K={k_opt} (BIC@6 delta={bic_k6 - bic_vals[k_opt]:.2f})")

    transition_json_path = os.path.join(out_dir, "transition_matrices.json")
    with open(transition_json_path, "w") as f:
        json.dump(transition_json, f, indent=2)

    _write_csv(
        os.path.join(out_dir, "transition_summary.csv"),
        transition_rows,
    )
    _write_csv(
        os.path.join(out_dir, "feature_transition_mi.csv"),
        mi_rows,
    )
    _write_csv(
        os.path.join(out_dir, "feature_transition_mi_by_destination.csv"),
        mi_dest_rows,
    )
    _write_csv(
        os.path.join(out_dir, "bic_analysis.csv"),
        bic_rows,
    )

    print("\n[Phase 0] Wrote:")
    print(f"  - {transition_json_path}")
    print(f"  - {os.path.join(out_dir, 'transition_summary.csv')}")
    print(f"  - {os.path.join(out_dir, 'feature_transition_mi.csv')}")
    print(f"  - {os.path.join(out_dir, 'feature_transition_mi_by_destination.csv')}")
    print(f"  - {os.path.join(out_dir, 'bic_analysis.csv')}")


# -----------------------------------------------------------------------------
# Phase 1: Semi-Markov baseline
# -----------------------------------------------------------------------------


def _train_logistic_transition_model(
    traces: Sequence[TraceRecord],
    labels: Sequence[np.ndarray],
    feature_names: Sequence[str],
    seed: int,
    mi_threshold: float,
    mode: str,
) -> Tuple[Optional[LogisticArtifacts], float]:
    X_blocks = []
    y_blocks = []

    for t, z in zip(traces, labels):
        if len(z) < 2:
            continue
        X = np.stack([_feature_array(t, f)[1:] for f in feature_names], axis=1)
        y = (z[1:] != z[:-1]).astype(int)
        X_blocks.append(X)
        y_blocks.append(y)

    if not X_blocks:
        return None, 0.0

    X_all = np.concatenate(X_blocks, axis=0)
    y_all = np.concatenate(y_blocks, axis=0)
    if len(np.unique(y_all)) < 2:
        return None, 0.0

    mi = _safe_mutual_info(X_all, y_all, seed)
    max_mi = float(np.max(mi)) if mi.size else 0.0

    if mode == "never":
        return None, max_mi
    if mode == "auto" and max_mi < mi_threshold:
        return None, max_mi

    mu = np.mean(X_all, axis=0)
    sd = np.std(X_all, axis=0) + 1e-6
    Xn = (X_all - mu[None, :]) / sd[None, :]

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(Xn, y_all)

    return (
        LogisticArtifacts(
            model=lr,
            mean=mu,
            std=sd,
            feature_names=list(feature_names),
            max_mi=max_mi,
        ),
        max_mi,
    )


def _semi_markov_generate_trace(
    trace: TraceRecord,
    gmm_art: GMMArtifacts,
    transition_matrix: np.ndarray,
    initial_probs: np.ndarray,
    logistic_art: Optional[LogisticArtifacts],
    smoothing_window: int,
    clip_min: float,
    clip_max: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    k = len(gmm_art.means)
    T = len(trace.power)
    rng = np.random.default_rng(seed + trace.trace_uid)

    states = np.zeros((T,), dtype=np.int64)
    states[0] = int(rng.choice(np.arange(k), p=initial_probs))

    offdiag_probs = np.zeros_like(transition_matrix)
    for s in range(k):
        row = transition_matrix[s].copy()
        row[s] = 0.0
        denom = float(np.sum(row))
        if denom > 0:
            offdiag_probs[s] = row / denom
        else:
            offdiag_probs[s, s] = 1.0

    for t_idx in range(1, T):
        s_prev = int(states[t_idx - 1])

        if logistic_art is None:
            p_transition = float(max(0.0, 1.0 - transition_matrix[s_prev, s_prev]))
        else:
            x = np.array([_feature_array(trace, f)[t_idx] for f in logistic_art.feature_names], dtype=np.float64)
            xn = (x - logistic_art.mean) / logistic_art.std
            p_transition = float(logistic_art.model.predict_proba(xn.reshape(1, -1))[0, 1])

        p_transition = float(np.clip(p_transition, 0.0, 1.0))

        if rng.random() < p_transition:
            p_dest = offdiag_probs[s_prev]
            if np.sum(p_dest) > 0:
                s_next = int(rng.choice(np.arange(k), p=p_dest))
            else:
                s_next = s_prev
        else:
            s_next = s_prev

        states[t_idx] = s_next

    watts = rng.normal(gmm_art.means[states], gmm_art.stds[states])
    watts = _median_smooth_same_state(watts, states, window=smoothing_window)
    watts = np.clip(watts, clip_min, clip_max)
    return states, watts


def _resolve_weight_path(config: ConfigKey, weight_dirs: Sequence[str]) -> Optional[str]:
    filename = f"{config.model_name}_{config.hardware.lower()}_tp{config.tp}.pt"
    for wd in weight_dirs:
        p = os.path.join(wd, filename)
        if os.path.exists(p):
            return p
    return None


def _evaluate_bigru_on_traces(
    traces: Sequence[TraceRecord],
    test_indices: np.ndarray,
    config: ConfigKey,
    weight_path: str,
    device,
    k_states: int,
    smoothing_window: int,
    clip_min: float,
    clip_max: float,
    dt: float,
    max_lags: int,
    seed: int,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    torch, _, _ = _import_torch()
    from model.core.utils import load_classifier

    test_traces = [traces[i] for i in test_indices]

    # Match training-time state ordering used by historical BiGRU weights:
    # unsorted GMM over all traces in this config.
    y_all = np.concatenate([t.power for t in traces], axis=0)
    gmm_art = _fit_gmm(y_all, k=k_states, seed=seed, sort_components=False)

    classifier = load_classifier(
        weight_path,
        device=device,
        Dx=6,
        K=k_states,
    )
    classifier.eval()

    all_real = []
    all_syn = []
    example = None

    for tr in test_traces:
        x = torch.from_numpy(tr.schedule_x6[None].astype(np.float32)).to(device)
        with torch.no_grad():
            logits = classifier(x).squeeze(0)
            states = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)

        rng = np.random.default_rng(seed + tr.trace_uid + 17)
        watts = rng.normal(gmm_art.means[states], gmm_art.stds[states])
        watts = _median_smooth_same_state(watts, states, window=smoothing_window)
        watts = np.clip(watts, clip_min, clip_max)

        all_real.append(tr.power)
        all_syn.append(watts)

        if example is None:
            example = {
                "timestamps": tr.timestamps,
                "real_power": tr.power,
                "synthetic_power": watts,
            }

    metrics = _compute_fidelity_metrics(all_real, all_syn, dt=dt, max_lags=max_lags)
    return metrics, example


def _plot_example_trace(
    out_path: str,
    title: str,
    timestamps: np.ndarray,
    real: np.ndarray,
    semi: np.ndarray,
    bigru: Optional[np.ndarray],
) -> None:
    _ensure_dir(os.path.dirname(out_path) or ".")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(timestamps, real, label="Measured", linewidth=1.2)
    ax.plot(timestamps, semi, label="Semi-Markov", linewidth=1.2)
    if bigru is not None:
        ax.plot(timestamps, bigru, label="BiGRU", linewidth=1.1, alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_phase1(
    configs: Dict[str, ConfigKey],
    traces_by_config: Dict[str, List[TraceRecord]],
    out_dir: str,
    seed: int,
    k_states: int,
    train_frac: float,
    val_frac: float,
    dt: float,
    max_lags: int,
    smoothing_window: int,
    logistic_features: Sequence[str],
    logistic_mode: str,
    transition_mi_threshold: float,
    compare_bigru: bool,
    weight_dirs: Sequence[str],
    device,
    example_config_substrings: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, "plots"))

    rows = []
    comparison_rows = []
    semi_metrics_by_config: Dict[str, Dict[str, float]] = {}
    bigru_metrics_by_config: Dict[str, Dict[str, float]] = {}

    example_cache: Dict[str, Dict[str, np.ndarray]] = {}

    print("\n[Phase 1] Semi-Markov baseline")

    for config_id, traces in sorted(traces_by_config.items()):
        cfg = configs[config_id]
        n = len(traces)
        if n < 3:
            print(f"  Skipping {cfg.slug}: not enough traces ({n})")
            continue

        split = _split_indices(n, train_frac=train_frac, val_frac=val_frac, seed=seed)
        if len(split.test) == 0:
            print(f"  Skipping {cfg.slug}: no test traces after split")
            continue

        train_traces = [traces[i] for i in split.train]
        test_traces = [traces[i] for i in split.test]

        y_train = np.concatenate([t.power for t in train_traces], axis=0)
        clip_min = float(np.min(y_train))
        clip_max = float(np.max(y_train))

        gmm_art = _fit_gmm(y_train, k=k_states, seed=seed, sort_components=True)
        labels_train = [_predict_labels(gmm_art, t.power) for t in train_traces]

        Tmat, _, occupancy, _, diag_w, sojourn_w = _compute_transition_matrix(labels_train, k_states)

        init_counts = np.zeros((k_states,), dtype=np.float64)
        for z in labels_train:
            if len(z) > 0:
                init_counts[int(z[0])] += 1.0
        if np.sum(init_counts) <= 0:
            init_counts = occupancy.astype(np.float64)
        if np.sum(init_counts) <= 0:
            init_probs = np.ones((k_states,), dtype=np.float64) / float(k_states)
        else:
            init_probs = init_counts / np.sum(init_counts)

        logistic_art, max_mi = _train_logistic_transition_model(
            traces=train_traces,
            labels=labels_train,
            feature_names=logistic_features,
            seed=seed,
            mi_threshold=transition_mi_threshold,
            mode=logistic_mode,
        )

        mode_used = "logistic" if logistic_art is not None else "unconditional"

        all_real = []
        all_syn = []
        first_example = None

        for tr in test_traces:
            _, watts = _semi_markov_generate_trace(
                trace=tr,
                gmm_art=gmm_art,
                transition_matrix=Tmat,
                initial_probs=init_probs,
                logistic_art=logistic_art,
                smoothing_window=smoothing_window,
                clip_min=clip_min,
                clip_max=clip_max,
                seed=seed,
            )
            all_real.append(tr.power)
            all_syn.append(watts)

            if first_example is None:
                first_example = {
                    "timestamps": tr.timestamps,
                    "real_power": tr.power,
                    "semi_power": watts,
                }

        metrics = _compute_fidelity_metrics(all_real, all_syn, dt=dt, max_lags=max_lags)
        metrics.update(
            {
                "config_id": config_id,
                "model_name": cfg.model_name,
                "hardware": cfg.hardware,
                "tp": cfg.tp,
                "model_size_b": cfg.model_size_b,
                "num_traces": n,
                "num_train_traces": int(len(split.train)),
                "num_test_traces": int(len(split.test)),
                "transition_mode": mode_used,
                "transition_mi_max": float(max_mi),
                "diag_weighted_mean": float(diag_w),
                "sojourn_weighted": float(sojourn_w),
            }
        )
        rows.append(metrics)
        semi_metrics_by_config[config_id] = metrics

        if first_example is not None:
            example_cache[config_id] = first_example

        bigru_metrics = None
        bigru_example = None

        if compare_bigru:
            weight_path = _resolve_weight_path(cfg, weight_dirs)
            if weight_path is None:
                print(f"  {cfg.slug:40s}  no BiGRU weights found")
            else:
                try:
                    bigru_metrics, bigru_example = _evaluate_bigru_on_traces(
                        traces=traces,
                        test_indices=split.test,
                        config=cfg,
                        weight_path=weight_path,
                        device=device,
                        k_states=k_states,
                        smoothing_window=smoothing_window,
                        clip_min=clip_min,
                        clip_max=clip_max,
                        dt=dt,
                        max_lags=max_lags,
                        seed=seed,
                    )
                    bigru_metrics_by_config[config_id] = {
                        **bigru_metrics,
                        "config_id": config_id,
                        "model_name": cfg.model_name,
                        "hardware": cfg.hardware,
                        "tp": cfg.tp,
                        "model_size_b": cfg.model_size_b,
                    }
                except Exception as exc:
                    print(f"  {cfg.slug:40s}  BiGRU eval failed: {exc}")

        print(
            f"  {cfg.slug:40s}  KS={metrics['ks_statistic']:.4f}  ACF_R2={metrics['acf_r2']:.4f}  "
            f"NRMSE={metrics['nrmse']:.4f}  mode={mode_used}"
        )

        if bigru_metrics is not None:
            comp = {
                "config_id": config_id,
                "model_name": cfg.model_name,
                "hardware": cfg.hardware,
                "tp": cfg.tp,
                "model_size_b": cfg.model_size_b,
            }
            metric_keys = [
                "ks_statistic",
                "acf_r2",
                "nrmse",
                "p95_error_pct",
                "p99_error_pct",
                "delta_energy_pct",
                "abs_energy_error_pct",
            ]
            for key in metric_keys:
                comp[f"semi_{key}"] = metrics.get(key, float("nan"))
                comp[f"bigru_{key}"] = bigru_metrics.get(key, float("nan"))
                comp[f"delta_semi_minus_bigru_{key}"] = (
                    float(metrics.get(key, np.nan)) - float(bigru_metrics.get(key, np.nan))
                )
            comparison_rows.append(comp)

            if bigru_example is not None and config_id in example_cache:
                example_cache[config_id]["bigru_power"] = bigru_example["synthetic_power"]

    trace_fidelity_path = os.path.join(out_dir, "trace_fidelity.csv")
    _write_csv(trace_fidelity_path, rows)

    comparison_path = os.path.join(out_dir, "comparison_vs_bigru.csv")
    _write_csv(comparison_path, comparison_rows)

    # Representative plots
    selected_config_ids: List[str] = []
    if example_config_substrings:
        for cid in example_cache.keys():
            if any(sub.lower() in cid.lower() for sub in example_config_substrings):
                selected_config_ids.append(cid)
    else:
        # default: up to 3 configs, prioritizing reference-like names
        preferred = [
            "llama-3-8b|hardware=H100|tp=1",
            "llama-3-70b|hardware=A100|tp=4",
        ]
        for cid in example_cache.keys():
            if any(p in cid for p in preferred):
                selected_config_ids.append(cid)
        if len(selected_config_ids) < 3:
            for cid in sorted(example_cache.keys()):
                if cid not in selected_config_ids:
                    selected_config_ids.append(cid)
                    if len(selected_config_ids) >= 3:
                        break

    for cid in selected_config_ids:
        cfg = configs[cid]
        ex = example_cache[cid]
        out_plot = os.path.join(out_dir, "plots", f"{cfg.slug}_trace.png")
        _plot_example_trace(
            out_path=out_plot,
            title=f"{cfg.model_name} {cfg.hardware} TP={cfg.tp}",
            timestamps=ex["timestamps"],
            real=ex["real_power"],
            semi=ex["semi_power"],
            bigru=ex.get("bigru_power"),
        )

    print("\n[Phase 1] Wrote:")
    print(f"  - {trace_fidelity_path}")
    print(f"  - {comparison_path}")
    print(f"  - {os.path.join(out_dir, 'plots')}")

    return semi_metrics_by_config


# -----------------------------------------------------------------------------
# GRU training utilities (Phase 2/3/4)
# -----------------------------------------------------------------------------


def _build_sequence_gru_classifier(nn_mod, input_dim: int, num_states: int, hidden_dim: int, bidirectional: bool):
    class _SequenceGRUClassifier(nn_mod.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn_mod.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            out_dim = hidden_dim * (2 if bidirectional else 1)
            self.fc = nn_mod.Linear(out_dim, num_states)

        def forward(self, x):
            h, _ = self.gru(x)
            return self.fc(h)

    return _SequenceGRUClassifier()


def _fit_feature_standardizer(traces: Sequence[TraceRecord], feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    if feature_names == ["constant_one"]:
        return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
    X = np.concatenate([_feature_matrix(t, feature_names) for t in traces], axis=0)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0) + 1e-6
    return mu, sd


def _prepare_records(
    traces: Sequence[TraceRecord],
    feature_names: Sequence[str],
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
    gmm_art: GMMArtifacts,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for tr in traces:
        x = _feature_matrix(tr, feature_names)
        x = (x - feat_mu[None, :]) / feat_sd[None, :]
        z = _predict_labels(gmm_art, tr.power)
        q = _predict_responsibilities(gmm_art, tr.power)
        records.append(
            {
                "trace": tr,
                "x": x.astype(np.float32),
                "z": z.astype(np.int64),
                "q": q.astype(np.float32),
                "y": tr.power.astype(np.float64),
            }
        )
    return records


def _evaluate_sequence_loss(
    model,
    records: Sequence[Dict[str, object]],
    device,
    label_mode: str,
    label_smoothing: float,
) -> float:
    torch, _, F = _import_torch()
    if len(records) == 0:
        return float("nan")

    model.eval()
    total = 0.0
    with torch.no_grad():
        for rec in records:
            x = torch.from_numpy(rec["x"][None]).to(device)
            logits = model(x).squeeze(0)
            if label_mode == "hard":
                z = torch.from_numpy(rec["z"]).long().to(device)
                loss = F.cross_entropy(logits, z, label_smoothing=label_smoothing)
            else:
                q = torch.from_numpy(rec["q"]).float().to(device)
                loss = F.kl_div(F.log_softmax(logits, dim=-1), q, reduction="batchmean")
            total += float(loss.item())
    return total / len(records)


def _train_sequence_model(
    train_records: Sequence[Dict[str, object]],
    val_records: Sequence[Dict[str, object]],
    input_dim: int,
    k_states: int,
    hidden_dim: int,
    bidirectional: bool,
    epochs: int,
    lr: float,
    device,
    label_mode: str,
    label_smoothing: float,
    seed: int,
) -> Tuple[object, List[Dict[str, float]]]:
    torch, nn_mod, F = _import_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_sequence_gru_classifier(
        nn_mod=nn_mod,
        input_dim=input_dim,
        num_states=k_states,
        hidden_dim=hidden_dim,
        bidirectional=bidirectional,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[Dict[str, float]] = []

    best_val = float("inf")
    best_state = None

    idxs = np.arange(len(train_records))

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(idxs)
        train_loss_total = 0.0

        for i in idxs:
            rec = train_records[int(i)]
            x = torch.from_numpy(rec["x"][None]).to(device)
            logits = model(x).squeeze(0)

            if label_mode == "hard":
                z = torch.from_numpy(rec["z"]).long().to(device)
                loss = F.cross_entropy(logits, z, label_smoothing=label_smoothing)
            else:
                q = torch.from_numpy(rec["q"]).float().to(device)
                loss = F.kl_div(F.log_softmax(logits, dim=-1), q, reduction="batchmean")

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_loss_total += float(loss.item())

        train_loss = train_loss_total / max(1, len(train_records))
        val_loss = _evaluate_sequence_loss(
            model=model,
            records=val_records,
            device=device,
            label_mode=label_mode,
            label_smoothing=label_smoothing,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"    epoch={epoch:4d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def _generate_with_sequence_model(
    model,
    records: Sequence[Dict[str, object]],
    gmm_art: GMMArtifacts,
    device,
    smoothing_window: int,
    clip_min: float,
    clip_max: float,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    torch, _, _ = _import_torch()
    model.eval()

    real_list = []
    syn_list = []
    example = None

    with torch.no_grad():
        for rec in records:
            tr: TraceRecord = rec["trace"]
            x = torch.from_numpy(rec["x"][None]).to(device)
            logits = model(x).squeeze(0)
            states = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)

            rng = np.random.default_rng(seed + tr.trace_uid + 1000)
            watts = rng.normal(gmm_art.means[states], gmm_art.stds[states])
            watts = _median_smooth_same_state(watts, states, window=smoothing_window)
            watts = np.clip(watts, clip_min, clip_max)

            real_list.append(rec["y"])
            syn_list.append(watts)

            if example is None:
                example = {
                    "timestamps": tr.timestamps,
                    "real_power": rec["y"],
                    "synthetic_power": watts,
                }

    return real_list, syn_list, example


def _select_reference_config_id(
    configs: Dict[str, ConfigKey],
    model_name: str,
    hardware: str,
    tp: int,
) -> Optional[str]:
    target_hw = hardware.upper()
    for cid, cfg in configs.items():
        if cfg.model_name == model_name and cfg.hardware.upper() == target_hw and int(cfg.tp) == int(tp):
            return cid
    return None


# -----------------------------------------------------------------------------
# Phase 2: Feature ablation on BiGRU
# -----------------------------------------------------------------------------


def run_phase2(
    configs: Dict[str, ConfigKey],
    traces_by_config: Dict[str, List[TraceRecord]],
    out_dir: str,
    seed: int,
    k_states: int,
    train_frac: float,
    val_frac: float,
    dt: float,
    max_lags: int,
    smoothing_window: int,
    epochs: int,
    lr: float,
    label_smoothing: float,
    device,
    reference_model: str,
    reference_hardware: str,
    reference_tp: int,
) -> Tuple[Optional[str], Dict[str, float]]:
    if device is None:
        raise RuntimeError("Phase 2 requires PyTorch. Re-run with an environment where torch imports successfully.")
    _ensure_dir(out_dir)
    curves_dir = os.path.join(out_dir, "training_curves")
    _ensure_dir(curves_dir)

    ref_id = _select_reference_config_id(
        configs=configs,
        model_name=reference_model,
        hardware=reference_hardware,
        tp=reference_tp,
    )
    if ref_id is None:
        print("\n[Phase 2] Reference config not found; skipping.")
        return None, {}

    cfg = configs[ref_id]
    traces = traces_by_config[ref_id]
    n = len(traces)
    if n < 3:
        print(f"\n[Phase 2] Not enough traces for {cfg.slug}; skipping.")
        return None, {}

    split = _split_indices(n=n, train_frac=train_frac, val_frac=val_frac, seed=seed)
    train_traces = [traces[i] for i in split.train]
    val_traces = [traces[i] for i in split.val]
    test_traces = [traces[i] for i in split.test]

    y_train = np.concatenate([t.power for t in train_traces], axis=0)
    clip_min = float(np.min(y_train))
    clip_max = float(np.max(y_train))

    gmm_art = _fit_gmm(y_train, k=k_states, seed=seed, sort_components=True)

    print(f"\n[Phase 2] Feature ablation on {cfg.slug}")
    print(f"  train={len(train_traces)} val={len(val_traces)} test={len(test_traces)}")

    rows: List[Dict[str, object]] = []
    score_map: Dict[str, float] = {}

    for feature_set_id, feature_names in PHASE2_FEATURE_SETS.items():
        print(f"  -> {feature_set_id} features={feature_names}")

        feat_mu, feat_sd = _fit_feature_standardizer(train_traces, feature_names)

        train_records = _prepare_records(train_traces, feature_names, feat_mu, feat_sd, gmm_art)
        val_records = _prepare_records(val_traces, feature_names, feat_mu, feat_sd, gmm_art)
        test_records = _prepare_records(test_traces, feature_names, feat_mu, feat_sd, gmm_art)

        model, history = _train_sequence_model(
            train_records=train_records,
            val_records=val_records,
            input_dim=len(feature_names),
            k_states=k_states,
            hidden_dim=64,
            bidirectional=True,
            epochs=epochs,
            lr=lr,
            device=device,
            label_mode="hard",
            label_smoothing=label_smoothing,
            seed=seed,
        )

        real, syn, _ = _generate_with_sequence_model(
            model=model,
            records=test_records,
            gmm_art=gmm_art,
            device=device,
            smoothing_window=smoothing_window,
            clip_min=clip_min,
            clip_max=clip_max,
            seed=seed,
        )
        metrics = _compute_fidelity_metrics(real, syn, dt=dt, max_lags=max_lags)

        row = {
            "config_id": ref_id,
            "model_name": cfg.model_name,
            "hardware": cfg.hardware,
            "tp": cfg.tp,
            "feature_set": feature_set_id,
            "features": ",".join(feature_names),
            "input_dim": len(feature_names),
            "label_mode": "hard",
            **metrics,
        }
        rows.append(row)

        score = _metric_score(metrics)
        score_map[feature_set_id] = score

        curve_path = os.path.join(curves_dir, f"{feature_set_id}_curve.csv")
        _write_csv(curve_path, history, fieldnames=["epoch", "train_loss", "val_loss"])

        print(
            f"     KS={metrics['ks_statistic']:.4f} ACF_R2={metrics['acf_r2']:.4f} NRMSE={metrics['nrmse']:.4f} score={score:.4f}"
        )

    out_csv = os.path.join(out_dir, "feature_ablation_results.csv")
    _write_csv(out_csv, rows)

    if score_map:
        best_feature_set = min(score_map, key=score_map.get)
    else:
        best_feature_set = None

    print("\n[Phase 2] Wrote:")
    print(f"  - {out_csv}")
    print(f"  - {curves_dir}")
    if best_feature_set is not None:
        print(f"  Best feature set: {best_feature_set}")

    return best_feature_set, score_map


# -----------------------------------------------------------------------------
# Phase 3: Soft labels
# -----------------------------------------------------------------------------


def run_phase3(
    configs: Dict[str, ConfigKey],
    traces_by_config: Dict[str, List[TraceRecord]],
    out_dir: str,
    seed: int,
    k_states: int,
    train_frac: float,
    val_frac: float,
    dt: float,
    max_lags: int,
    smoothing_window: int,
    epochs: int,
    lr: float,
    label_smoothing: float,
    device,
    reference_model: str,
    reference_hardware: str,
    reference_tp: int,
    feature_set_id: str,
) -> Tuple[Optional[str], Dict[str, float]]:
    if device is None:
        raise RuntimeError("Phase 3 requires PyTorch. Re-run with an environment where torch imports successfully.")
    _ensure_dir(out_dir)
    curves_dir = os.path.join(out_dir, "training_curves")
    _ensure_dir(curves_dir)

    if feature_set_id not in PHASE2_FEATURE_SETS:
        print(f"\n[Phase 3] Unknown feature set {feature_set_id}; skipping.")
        return None, {}

    ref_id = _select_reference_config_id(
        configs=configs,
        model_name=reference_model,
        hardware=reference_hardware,
        tp=reference_tp,
    )
    if ref_id is None:
        print("\n[Phase 3] Reference config not found; skipping.")
        return None, {}

    cfg = configs[ref_id]
    traces = traces_by_config[ref_id]
    n = len(traces)
    if n < 3:
        print(f"\n[Phase 3] Not enough traces for {cfg.slug}; skipping.")
        return None, {}

    split = _split_indices(n=n, train_frac=train_frac, val_frac=val_frac, seed=seed)
    train_traces = [traces[i] for i in split.train]
    val_traces = [traces[i] for i in split.val]
    test_traces = [traces[i] for i in split.test]

    y_train = np.concatenate([t.power for t in train_traces], axis=0)
    clip_min = float(np.min(y_train))
    clip_max = float(np.max(y_train))

    gmm_art = _fit_gmm(y_train, k=k_states, seed=seed, sort_components=True)

    feature_names = PHASE2_FEATURE_SETS[feature_set_id]
    feat_mu, feat_sd = _fit_feature_standardizer(train_traces, feature_names)

    train_records = _prepare_records(train_traces, feature_names, feat_mu, feat_sd, gmm_art)
    val_records = _prepare_records(val_traces, feature_names, feat_mu, feat_sd, gmm_art)
    test_records = _prepare_records(test_traces, feature_names, feat_mu, feat_sd, gmm_art)

    print(f"\n[Phase 3] Soft-label experiment on {cfg.slug} with {feature_set_id}")

    rows: List[Dict[str, object]] = []
    score_map: Dict[str, float] = {}

    for label_mode in ["hard", "soft"]:
        print(f"  -> training label_mode={label_mode}")
        model, history = _train_sequence_model(
            train_records=train_records,
            val_records=val_records,
            input_dim=len(feature_names),
            k_states=k_states,
            hidden_dim=64,
            bidirectional=True,
            epochs=epochs,
            lr=lr,
            device=device,
            label_mode=label_mode,
            label_smoothing=(label_smoothing if label_mode == "hard" else 0.0),
            seed=seed,
        )

        real, syn, _ = _generate_with_sequence_model(
            model=model,
            records=test_records,
            gmm_art=gmm_art,
            device=device,
            smoothing_window=smoothing_window,
            clip_min=clip_min,
            clip_max=clip_max,
            seed=seed,
        )
        metrics = _compute_fidelity_metrics(real, syn, dt=dt, max_lags=max_lags)

        row = {
            "config_id": ref_id,
            "model_name": cfg.model_name,
            "hardware": cfg.hardware,
            "tp": cfg.tp,
            "feature_set": feature_set_id,
            "features": ",".join(feature_names),
            "input_dim": len(feature_names),
            "label_mode": label_mode,
            **metrics,
        }
        rows.append(row)

        score = _metric_score(metrics)
        score_map[label_mode] = score

        curve_path = os.path.join(curves_dir, f"{feature_set_id}_{label_mode}_curve.csv")
        _write_csv(curve_path, history, fieldnames=["epoch", "train_loss", "val_loss"])

        print(
            f"     mode={label_mode} KS={metrics['ks_statistic']:.4f} ACF_R2={metrics['acf_r2']:.4f} NRMSE={metrics['nrmse']:.4f} score={score:.4f}"
        )

    out_csv = os.path.join(out_dir, "soft_label_results.csv")
    _write_csv(out_csv, rows)

    best_label_mode = min(score_map, key=score_map.get) if score_map else None

    print("\n[Phase 3] Wrote:")
    print(f"  - {out_csv}")
    print(f"  - {curves_dir}")
    if best_label_mode is not None:
        print(f"  Best label mode: {best_label_mode}")

    return best_label_mode, score_map


# -----------------------------------------------------------------------------
# Phase 4: Uni vs Bi GRU
# -----------------------------------------------------------------------------


def run_phase4(
    configs: Dict[str, ConfigKey],
    traces_by_config: Dict[str, List[TraceRecord]],
    out_dir: str,
    seed: int,
    k_states: int,
    train_frac: float,
    val_frac: float,
    dt: float,
    max_lags: int,
    smoothing_window: int,
    epochs: int,
    lr: float,
    label_smoothing: float,
    device,
    reference_model: str,
    reference_hardware: str,
    reference_tp: int,
    feature_set_id: str,
    label_mode: str,
) -> None:
    if device is None:
        raise RuntimeError("Phase 4 requires PyTorch. Re-run with an environment where torch imports successfully.")
    _ensure_dir(out_dir)
    curves_dir = os.path.join(out_dir, "training_curves")
    _ensure_dir(curves_dir)

    if feature_set_id not in PHASE2_FEATURE_SETS:
        print(f"\n[Phase 4] Unknown feature set {feature_set_id}; skipping.")
        return
    if label_mode not in {"hard", "soft"}:
        print(f"\n[Phase 4] Unknown label mode {label_mode}; skipping.")
        return

    ref_id = _select_reference_config_id(
        configs=configs,
        model_name=reference_model,
        hardware=reference_hardware,
        tp=reference_tp,
    )
    if ref_id is None:
        print("\n[Phase 4] Reference config not found; skipping.")
        return

    cfg = configs[ref_id]
    traces = traces_by_config[ref_id]
    n = len(traces)
    if n < 3:
        print(f"\n[Phase 4] Not enough traces for {cfg.slug}; skipping.")
        return

    split = _split_indices(n=n, train_frac=train_frac, val_frac=val_frac, seed=seed)
    train_traces = [traces[i] for i in split.train]
    val_traces = [traces[i] for i in split.val]
    test_traces = [traces[i] for i in split.test]

    y_train = np.concatenate([t.power for t in train_traces], axis=0)
    clip_min = float(np.min(y_train))
    clip_max = float(np.max(y_train))
    gmm_art = _fit_gmm(y_train, k=k_states, seed=seed, sort_components=True)

    feature_names = PHASE2_FEATURE_SETS[feature_set_id]
    feat_mu, feat_sd = _fit_feature_standardizer(train_traces, feature_names)

    train_records = _prepare_records(train_traces, feature_names, feat_mu, feat_sd, gmm_art)
    val_records = _prepare_records(val_traces, feature_names, feat_mu, feat_sd, gmm_art)
    test_records = _prepare_records(test_traces, feature_names, feat_mu, feat_sd, gmm_art)

    variants = [
        ("Bi-64", 64, True),
        ("Uni-64", 64, False),
        ("Uni-128", 128, False),
    ]

    rows: List[Dict[str, object]] = []

    print(f"\n[Phase 4] Architecture sweep on {cfg.slug} using {feature_set_id} + {label_mode}")

    for variant_id, hidden_dim, bidirectional in variants:
        print(f"  -> {variant_id}")

        model, history = _train_sequence_model(
            train_records=train_records,
            val_records=val_records,
            input_dim=len(feature_names),
            k_states=k_states,
            hidden_dim=hidden_dim,
            bidirectional=bidirectional,
            epochs=epochs,
            lr=lr,
            device=device,
            label_mode=label_mode,
            label_smoothing=(label_smoothing if label_mode == "hard" else 0.0),
            seed=seed,
        )

        real, syn, _ = _generate_with_sequence_model(
            model=model,
            records=test_records,
            gmm_art=gmm_art,
            device=device,
            smoothing_window=smoothing_window,
            clip_min=clip_min,
            clip_max=clip_max,
            seed=seed,
        )
        metrics = _compute_fidelity_metrics(real, syn, dt=dt, max_lags=max_lags)

        rows.append(
            {
                "config_id": ref_id,
                "model_name": cfg.model_name,
                "hardware": cfg.hardware,
                "tp": cfg.tp,
                "variant": variant_id,
                "hidden_dim": hidden_dim,
                "bidirectional": int(bidirectional),
                "feature_set": feature_set_id,
                "label_mode": label_mode,
                **metrics,
            }
        )

        curve_path = os.path.join(curves_dir, f"{variant_id}_curve.csv")
        _write_csv(curve_path, history, fieldnames=["epoch", "train_loss", "val_loss"])

        print(
            f"     KS={metrics['ks_statistic']:.4f} ACF_R2={metrics['acf_r2']:.4f} NRMSE={metrics['nrmse']:.4f}"
        )

    out_csv = os.path.join(out_dir, "architecture_results.csv")
    _write_csv(out_csv, rows)

    print("\n[Phase 4] Wrote:")
    print(f"  - {out_csv}")
    print(f"  - {curves_dir}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run revised power trace architectural investigation phases (0-4)."
    )

    parser.add_argument("--npz", action="append", default=[], help="Path to an NPZ file (repeatable).")
    parser.add_argument("--npz-dir", default=None, help="Directory containing NPZ files.")
    parser.add_argument("--pattern", default="*.npz", help="Glob pattern under --npz-dir.")

    parser.add_argument(
        "--phases",
        nargs="+",
        default=["0"],
        help="Phases to run: 0 1 2 3 4 or 'all'.",
    )

    parser.add_argument("--results-root", default="results", help="Root output directory.")

    parser.add_argument("--k-states", type=int, default=K_DEFAULT, help="Default number of GMM states.")
    parser.add_argument("--bic-k-min", type=int, default=3)
    parser.add_argument("--bic-k-max", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)

    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    parser.add_argument("--max-lags", type=int, default=50)
    parser.add_argument("--smoothing-window", type=int, default=5)

    parser.add_argument(
        "--semi-markov-transition-mode",
        default="auto",
        choices=["auto", "always", "never"],
        help="Transition trigger model for semi-Markov baseline.",
    )
    parser.add_argument(
        "--semi-markov-mi-threshold",
        type=float,
        default=0.005,
        help="Use logistic transition model in auto mode when max MI >= threshold (nats).",
    )
    parser.add_argument(
        "--semi-markov-logistic-features",
        default=",".join(LOGISTIC_FEATURES_DEFAULT),
        help="Comma-separated feature names for logistic transition model.",
    )

    parser.add_argument(
        "--skip-bigru-compare",
        action="store_true",
        help="Skip BiGRU comparison in Phase 1.",
    )
    parser.add_argument(
        "--bigru-weights-dirs",
        nargs="+",
        default=["model/best_weights", "model/gru_classifier_weights", "model/new_weights"],
        help="Directories to search for BiGRU weights.",
    )

    parser.add_argument("--device", default=None, help="Torch device (cpu, cuda, cuda:0, ...).")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    parser.add_argument("--reference-model", default="llama-3-8b")
    parser.add_argument("--reference-hardware", default="H100")
    parser.add_argument("--reference-tp", type=int, default=1)

    parser.add_argument(
        "--phase4-improvement-margin",
        type=float,
        default=0.0,
        help="Minimum score improvement needed (BiGRU vs semi-Markov) to run Phase 4 unless --force-phase4.",
    )
    parser.add_argument("--force-phase4", action="store_true")

    parser.add_argument(
        "--example-configs",
        nargs="*",
        default=[],
        help="Optional substrings to choose representative configs for Phase 1 trace plots.",
    )

    args = parser.parse_args()

    phases = set(args.phases)
    if "all" in phases:
        phases = {"0", "1", "2", "3", "4"}

    npz_dir = args.npz_dir
    if not args.npz and npz_dir is None:
        npz_dir = "model/training_data"

    npz_paths = _collect_npz_paths(args.npz, npz_dir, args.pattern)
    if not npz_paths:
        raise SystemExit("No NPZ files found. Use --npz and/or --npz-dir.")

    logistic_features = [f.strip() for f in args.semi_markov_logistic_features.split(",") if f.strip()]
    if not logistic_features:
        logistic_features = list(LOGISTIC_FEATURES_DEFAULT)

    requires_torch = (
        ("2" in phases)
        or ("3" in phases)
        or ("4" in phases)
        or (("1" in phases) and (not args.skip_bigru_compare))
    )

    device = _resolve_device(args.device, requires_torch=requires_torch)

    print("Loading traces...")
    configs, traces_by_config, skipped = load_trace_groups(npz_paths)
    print(f"Loaded {len(configs)} configs across {sum(len(v) for v in traces_by_config.values())} traces")
    print(f"Skipped traces with insufficient bins: {skipped}")
    print(f"Device: {device}")

    diagnostics_dir = os.path.join(args.results_root, "diagnostics")
    semi_dir = os.path.join(args.results_root, "semi_markov")
    ablation_dir = os.path.join(args.results_root, "feature_ablation")
    soft_dir = os.path.join(args.results_root, "soft_labels")
    arch_dir = os.path.join(args.results_root, "architecture")

    bic_k_values = list(range(args.bic_k_min, args.bic_k_max + 1))

    semi_metrics_by_config: Dict[str, Dict[str, float]] = {}
    best_feature_set: Optional[str] = None
    best_label_mode: Optional[str] = None
    phase2_scores: Dict[str, float] = {}
    phase3_scores: Dict[str, float] = {}

    if "0" in phases:
        run_phase0(
            configs=configs,
            traces_by_config=traces_by_config,
            out_dir=diagnostics_dir,
            k_states=args.k_states,
            bic_k_values=bic_k_values,
            seed=args.seed,
        )

    if "1" in phases:
        semi_metrics_by_config = run_phase1(
            configs=configs,
            traces_by_config=traces_by_config,
            out_dir=semi_dir,
            seed=args.seed,
            k_states=args.k_states,
            train_frac=args.train_fraction,
            val_frac=args.val_fraction,
            dt=args.dt,
            max_lags=args.max_lags,
            smoothing_window=args.smoothing_window,
            logistic_features=logistic_features,
            logistic_mode=args.semi_markov_transition_mode,
            transition_mi_threshold=args.semi_markov_mi_threshold,
            compare_bigru=(not args.skip_bigru_compare),
            weight_dirs=args.bigru_weights_dirs,
            device=device,
            example_config_substrings=args.example_configs,
        )

    if "2" in phases:
        best_feature_set, phase2_scores = run_phase2(
            configs=configs,
            traces_by_config=traces_by_config,
            out_dir=ablation_dir,
            seed=args.seed,
            k_states=args.k_states,
            train_frac=args.train_fraction,
            val_frac=args.val_fraction,
            dt=args.dt,
            max_lags=args.max_lags,
            smoothing_window=args.smoothing_window,
            epochs=args.epochs,
            lr=args.lr,
            label_smoothing=args.label_smoothing,
            device=device,
            reference_model=args.reference_model,
            reference_hardware=args.reference_hardware,
            reference_tp=args.reference_tp,
        )

    if "3" in phases:
        if best_feature_set is None:
            # fallback to file if Phase 2 not executed in this run
            phase2_csv = os.path.join(ablation_dir, "feature_ablation_results.csv")
            if os.path.exists(phase2_csv):
                rows = []
                with open(phase2_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)
                if rows:
                    best_row = min(rows, key=lambda r: _metric_score({
                        "ks_statistic": float(r.get("ks_statistic", "nan")),
                        "acf_r2": float(r.get("acf_r2", "nan")),
                        "nrmse": float(r.get("nrmse", "nan")),
                        "p95_error_pct": float(r.get("p95_error_pct", "nan")),
                        "p99_error_pct": float(r.get("p99_error_pct", "nan")),
                        "delta_energy_pct": float(r.get("delta_energy_pct", "nan")),
                    }))
                    best_feature_set = best_row.get("feature_set")

        if best_feature_set is None:
            print("\n[Phase 3] Missing best feature set; skipping.")
        else:
            best_label_mode, phase3_scores = run_phase3(
                configs=configs,
                traces_by_config=traces_by_config,
                out_dir=soft_dir,
                seed=args.seed,
                k_states=args.k_states,
                train_frac=args.train_fraction,
                val_frac=args.val_fraction,
                dt=args.dt,
                max_lags=args.max_lags,
                smoothing_window=args.smoothing_window,
                epochs=args.epochs,
                lr=args.lr,
                label_smoothing=args.label_smoothing,
                device=device,
                reference_model=args.reference_model,
                reference_hardware=args.reference_hardware,
                reference_tp=args.reference_tp,
                feature_set_id=best_feature_set,
            )

    if "4" in phases:
        if best_feature_set is None:
            # fallback: load from phase2
            phase2_csv = os.path.join(ablation_dir, "feature_ablation_results.csv")
            if os.path.exists(phase2_csv):
                rows = []
                with open(phase2_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if rows:
                    best_row = min(rows, key=lambda r: _metric_score({
                        "ks_statistic": float(r.get("ks_statistic", "nan")),
                        "acf_r2": float(r.get("acf_r2", "nan")),
                        "nrmse": float(r.get("nrmse", "nan")),
                        "p95_error_pct": float(r.get("p95_error_pct", "nan")),
                        "p99_error_pct": float(r.get("p99_error_pct", "nan")),
                        "delta_energy_pct": float(r.get("delta_energy_pct", "nan")),
                    }))
                    best_feature_set = best_row.get("feature_set")

        if best_label_mode is None:
            # fallback: load from phase3
            phase3_csv = os.path.join(soft_dir, "soft_label_results.csv")
            if os.path.exists(phase3_csv):
                rows = []
                with open(phase3_csv, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if rows:
                    best_row = min(rows, key=lambda r: _metric_score({
                        "ks_statistic": float(r.get("ks_statistic", "nan")),
                        "acf_r2": float(r.get("acf_r2", "nan")),
                        "nrmse": float(r.get("nrmse", "nan")),
                        "p95_error_pct": float(r.get("p95_error_pct", "nan")),
                        "p99_error_pct": float(r.get("p99_error_pct", "nan")),
                        "delta_energy_pct": float(r.get("delta_energy_pct", "nan")),
                    }))
                    best_label_mode = best_row.get("label_mode")

        if best_feature_set is None or best_label_mode is None:
            print("\n[Phase 4] Missing prerequisites (best feature set/label mode); skipping.")
        else:
            # Gate Phase 4 by semi-Markov vs BiGRU score if possible
            should_run = args.force_phase4

            if not should_run:
                ref_id = _select_reference_config_id(
                    configs=configs,
                    model_name=args.reference_model,
                    hardware=args.reference_hardware,
                    tp=args.reference_tp,
                )
                semi_score = None
                bigru_score = None

                if ref_id is not None:
                    # Semi score from Phase 1 output if available
                    semi_csv = os.path.join(semi_dir, "trace_fidelity.csv")
                    if os.path.exists(semi_csv):
                        with open(semi_csv, "r", newline="") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                if row.get("config_id") == ref_id:
                                    semi_score = _metric_score({
                                        "ks_statistic": float(row.get("ks_statistic", "nan")),
                                        "acf_r2": float(row.get("acf_r2", "nan")),
                                        "nrmse": float(row.get("nrmse", "nan")),
                                        "p95_error_pct": float(row.get("p95_error_pct", "nan")),
                                        "p99_error_pct": float(row.get("p99_error_pct", "nan")),
                                        "delta_energy_pct": float(row.get("delta_energy_pct", "nan")),
                                    })
                                    break

                    # BiGRU score from best available (phase3 first, then phase2)
                    if phase3_scores:
                        bigru_score = min(phase3_scores.values())
                    elif phase2_scores:
                        bigru_score = min(phase2_scores.values())

                if semi_score is not None and bigru_score is not None:
                    should_run = (bigru_score + args.phase4_improvement_margin) < semi_score
                    print(
                        "\n[Phase 4 gate] "
                        f"semi_score={semi_score:.4f} bigru_score={bigru_score:.4f} "
                        f"margin={args.phase4_improvement_margin:.4f} -> run={should_run}"
                    )
                else:
                    print("\n[Phase 4 gate] Missing scores; skipping by default (use --force-phase4 to override).")
                    should_run = False

            if should_run:
                run_phase4(
                    configs=configs,
                    traces_by_config=traces_by_config,
                    out_dir=arch_dir,
                    seed=args.seed,
                    k_states=args.k_states,
                    train_frac=args.train_fraction,
                    val_frac=args.val_fraction,
                    dt=args.dt,
                    max_lags=args.max_lags,
                    smoothing_window=args.smoothing_window,
                    epochs=args.epochs,
                    lr=args.lr,
                    label_smoothing=args.label_smoothing,
                    device=device,
                    reference_model=args.reference_model,
                    reference_hardware=args.reference_hardware,
                    reference_tp=args.reference_tp,
                    feature_set_id=best_feature_set,
                    label_mode=best_label_mode,
                )
            else:
                print("[Phase 4] Skipped.")


if __name__ == "__main__":
    main()
