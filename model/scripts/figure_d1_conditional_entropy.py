#!/usr/bin/env python3
"""
Figure D1: Held-out sequence classification analysis.

Trains BiGRU sequence classifiers on regime labels z_t for:
- F2 = (A_t, Delta A_t)
- Full-6D = (new_req_count, new_input_tokens, new_output_tokens,
             active_requests, prefill_tokens, decode_tokens)

Evaluates held-out sequence classification performance:
- Test cross-entropy
- Test accuracy

This directly compares predictive utility of F2 vs Full-6D in the model class
used by the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch

from model.classifiers.gru import GRUClassifier

# Keep this in sync with methods figure BIC policy.
BIC_CONFIGS = {
    "bic_config1": {
        "config_id": "llama-3-8b_A100_tp1",
        "title": "Llama-3.1-8B / A100 / TP=1",
    },
    "bic_config2": {
        "config_id": "llama-3-8b_H100_tp2",
        "title": "Llama-3.1-8B / H100 / TP=2",
    },
    "bic_config3": {
        "config_id": "gpt-oss-120b_H100_tp8",
        "title": "GPT-OSS-120B / H100 / TP=8 (MoE Proxy)",
    },
    "bic_config4": {
        "config_id": "deepseek-r1-distill-70b_H100_tp4",
        "title": "DeepSeek-R1-Distill-70B / H100 / TP=4 (Dense)",
    },
}

SUBSET_ORDER = ["A_t", "ΔA_t", "F2", "Full-6D"]
SUBSET_ESTIMATOR = {
    "A_t": "plugin_binned",
    "ΔA_t": "plugin_binned",
    "F2": "plugin_binned",
    "Full-6D": "knn_joint_posterior",
}
SEQ_SUBSET_ORDER = ["F2", "Full-6D"]


@dataclass
class TraceFeatures:
    z: np.ndarray
    a_t: np.ndarray
    delta_a_t: np.ndarray
    f2: np.ndarray
    f6: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_paths() -> Dict[str, str]:
    root = _repo_root()
    return {
        "experimental_manifest": str(root / "results" / "experimental_continuous_v1" / "manifest.json"),
        "run_manifest": str(root / "results" / "continuous_v1_gmm_bigru" / "k10_f2" / "run_manifest.json"),
        "pair_manifest_csv": str(root / "results" / "stage0" / "pair_manifest.csv"),
        "throughput_db": str(root / "model" / "config" / "throughput_database.json"),
        "out_figure": str(root / "figures" / "figure_d1_conditional_entropy.pdf"),
        "out_per_config_csv": str(root / "results" / "eval_paper" / "figure_d1_conditional_entropy_per_config.csv"),
        "out_json": str(root / "results" / "eval_paper" / "figure_d1_conditional_entropy_manifest.json"),
    }


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_existing_path(path_str: str, base_dir: str) -> Optional[str]:
    raw = Path(path_str)
    if raw.is_absolute():
        return str(raw) if raw.exists() else None
    local = Path(path_str)
    if local.exists():
        return str(local)
    from_base = Path(base_dir) / raw
    if from_base.exists():
        return str(from_base)
    return None


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    _ensure_parent(path)
    with open(path, "w") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    _ensure_parent(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def parse_config_id(config_id: str) -> Tuple[str, str, int]:
    m = re.match(r"^(?P<model>.+)_(?P<hw>A100|H100)_tp(?P<tp>\d+)$", str(config_id).strip())
    if m is None:
        raise ValueError(f"Invalid config_id format: {config_id}")
    return str(m.group("model")), str(m.group("hw")), int(m.group("tp"))


def parse_pair_key(pair_key: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for seg in str(pair_key).strip().split("|"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            out[str(k).strip()] = str(v).strip()
        else:
            out["model"] = str(seg).strip()
    return out


def deterministic_pairkey_json_path(config_id: str, pair_key: str) -> Optional[Path]:
    """
    Resolve benchmark-style fallback path from pair key and config id.

    Expected pattern:
      data/extraneous-data/benchmark-{model}-{hw}/tp{tp}/
      {pair_model}_tp{pair_tp}_rate{rate}_iter{iter}_{date}.json
    """
    try:
        model_name, hw, tp = parse_config_id(config_id)
    except Exception:
        return None

    parts = parse_pair_key(pair_key)
    required = {"model", "tp", "rate", "iter", "date"}
    if not required.issubset(parts):
        return None

    file_name = (
        f"{parts['model']}_tp{parts['tp']}_rate{parts['rate']}_iter{parts['iter']}_{parts['date']}.json"
    )

    root = _repo_root()
    candidates = [
        root
        / "data"
        / "extraneous-data"
        / f"benchmark-{model_name}-{hw.lower()}"
        / f"tp{tp}"
        / file_name,
        root
        / "data"
        / "extraneous-data"
        / f"benchmark-{parts['model']}-{hw.lower()}"
        / f"tp{parts['tp']}"
        / file_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_pair_manifest_map(pair_manifest_csv: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    base_dir = str(Path(pair_manifest_csv).resolve().parent)
    with open(pair_manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip() != "matched":
                continue
            key = str(row.get("pair_key", "")).strip()
            json_path_raw = str(row.get("json_path", "")).strip()
            if key == "" or json_path_raw == "":
                continue
            json_path = _resolve_existing_path(json_path_raw, base_dir)
            if json_path is not None:
                out[key] = json_path
    return out


def _synthesize_request_timestamps(payload: Mapping[str, object], n: int) -> Optional[List[float]]:
    if n <= 0:
        return []

    duration = _to_float(payload.get("duration"))
    if duration is not None and duration > 0:
        step = float(duration) / float(max(n, 1))
        if step > 0:
            values = (np.arange(n, dtype=np.float64) + 0.5) * step + 1.0
            return [float(x) for x in values]

    request_rate = _to_float(payload.get("request_rate"))
    poisson_rate = _to_float(payload.get("poisson_rate"))
    rate = request_rate if request_rate is not None else poisson_rate
    if rate is not None and rate > 0:
        step = 1.0 / float(rate)
        values = (np.arange(n, dtype=np.float64) + 1.0) * step + 1.0
        return [float(x) for x in values]
    return None


def build_requests_from_json(
    request_json_path: str | Path,
    *,
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
    require_recorded_timestamps: bool,
) -> List[Dict[str, float]]:
    payload = _load_json(request_json_path)
    required = ("input_lens", "output_lens")
    missing = [k for k in required if not isinstance(payload.get(k), list)]
    if missing:
        raise ValueError(f"request json missing arrays: {missing}")

    input_lens = payload["input_lens"]
    output_lens = payload["output_lens"]
    n_base = int(min(len(input_lens), len(output_lens)))

    request_timestamps_raw = payload.get("request_timestamps")
    if isinstance(request_timestamps_raw, list):
        n = int(min(n_base, len(request_timestamps_raw)))
        request_timestamps = request_timestamps_raw[:n]
    else:
        if bool(require_recorded_timestamps):
            raise ValueError("request json missing arrays: ['request_timestamps']")
        n = int(n_base)
        synth = _synthesize_request_timestamps(payload, n)
        if synth is None:
            raise ValueError("request json missing arrays: ['request_timestamps']")
        request_timestamps = synth

    if n <= 0:
        raise ValueError("request arrays are empty after alignment")

    arrivals = np.asarray(request_timestamps[:n], dtype=np.float64) - float(power_start_epoch_s)

    if arrivals.size > 0:
        arr_min = float(np.min(arrivals))
        arr_max = float(np.max(arrivals))
        out_of_window = arr_min < (-float(dt)) or arr_max > (float(trace_duration_s) + float(dt))
        if out_of_window:
            median_arr = float(np.median(arrivals))
            tz_hours = int(np.round(median_arr / 3600.0))
            arrivals_tz = arrivals - (float(tz_hours) * 3600.0)
            tz_min = float(np.min(arrivals_tz))
            tz_max = float(np.max(arrivals_tz))
            if (
                tz_hours != 0
                and tz_min >= (-2.0 * float(dt))
                and tz_max <= (float(trace_duration_s) + 2.0 * float(dt))
            ):
                arrivals = arrivals_tz
            else:
                arrivals = arrivals - arr_min

    requests: List[Dict[str, float]] = []
    for i in range(n):
        a = float(arrivals[i])
        nin = float(input_lens[i])
        nout = float(output_lens[i])
        if not (np.isfinite(a) and np.isfinite(nin) and np.isfinite(nout)):
            continue
        requests.append(
            {
                "arrival_time": float(a),
                "input_tokens": float(max(0.0, nin)),
                "output_tokens": float(max(0.0, nout)),
            }
        )

    if not requests:
        raise ValueError("no valid requests after filtering")
    return requests


def _prepare_gmm_arrays(payload: Mapping[str, object]) -> Dict[str, np.ndarray]:
    means_sorted = np.asarray(payload.get("means", []), dtype=np.float64).reshape(-1)
    variances_sorted = np.asarray(payload.get("variances", []), dtype=np.float64).reshape(-1)
    weights_sorted = np.asarray(payload.get("weights", []), dtype=np.float64).reshape(-1)
    k = int(payload.get("k", means_sorted.size))

    if means_sorted.size != k or variances_sorted.size != k or weights_sorted.size != k:
        raise ValueError("Invalid GMM payload: means/variances/weights lengths must match k")

    variances_sorted = np.clip(variances_sorted, a_min=1e-12, a_max=None)
    weights_sorted = np.clip(weights_sorted, a_min=1e-12, a_max=None)
    weights_sorted = weights_sorted / np.sum(weights_sorted)

    order = np.asarray(payload.get("order", []), dtype=np.int64).reshape(-1)
    label_map = np.asarray(payload.get("label_map", []), dtype=np.int64).reshape(-1)

    if order.size == k and label_map.size == k:
        means_raw = np.empty((k,), dtype=np.float64)
        vars_raw = np.empty((k,), dtype=np.float64)
        weights_raw = np.empty((k,), dtype=np.float64)
        means_raw[order] = means_sorted
        vars_raw[order] = variances_sorted
        weights_raw[order] = weights_sorted
        return {
            "means_raw": means_raw,
            "variances_raw": vars_raw,
            "weights_raw": weights_raw,
            "label_map": label_map.astype(np.int64),
        }

    return {
        "means_raw": means_sorted,
        "variances_raw": variances_sorted,
        "weights_raw": weights_sorted,
        "label_map": np.arange(k, dtype=np.int64),
    }


def predict_regime_labels_from_gmm(power_values: np.ndarray, gmm_payload: Mapping[str, object]) -> np.ndarray:
    y = np.asarray(power_values, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return np.zeros((0,), dtype=np.int64)

    gmm = _prepare_gmm_arrays(gmm_payload)
    means = gmm["means_raw"]
    variances = gmm["variances_raw"]
    weights = gmm["weights_raw"]

    y2 = y.reshape(-1, 1)
    log_prob = (
        -0.5 * np.log(2.0 * np.pi * variances.reshape(1, -1))
        -0.5 * ((y2 - means.reshape(1, -1)) ** 2) / variances.reshape(1, -1)
        + np.log(weights.reshape(1, -1))
    )
    raw_labels = np.argmax(log_prob, axis=1).astype(np.int64)
    label_map = gmm["label_map"]
    if np.max(raw_labels, initial=-1) >= label_map.size:
        raise ValueError("Internal label mapping mismatch while predicting z_t")
    return label_map[raw_labels].astype(np.int64)


def histogram_requests(
    bin_ts: np.ndarray,
    req_ts: np.ndarray,
    in_tok: np.ndarray,
    out_tok: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.asarray(bin_ts, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        z = np.zeros((0,), dtype=np.float64)
        return z, z, z

    dt = float(np.median(np.diff(ts))) if ts.size > 1 else 0.25
    if ts.size > 1 and not np.all(np.diff(ts) > 0):
        ts = np.unique(ts)
        if ts.size <= 1:
            ts = np.array([0.0, dt], dtype=np.float64) if ts.size == 0 else np.array([ts[0], ts[0] + dt], dtype=np.float64)
    elif ts.size <= 1:
        ts = np.array([0.0, dt], dtype=np.float64) if ts.size == 0 else np.array([ts[0], ts[0] + dt], dtype=np.float64)

    edges = np.append(ts, ts[-1] + dt)

    req = np.asarray(req_ts, dtype=np.float64).reshape(-1)
    nin = np.asarray(in_tok, dtype=np.float64).reshape(-1)
    nout = np.asarray(out_tok, dtype=np.float64).reshape(-1)
    n = int(min(req.size, nin.size, nout.size))
    req, nin, nout = req[:n], nin[:n], nout[:n]

    cnt, _ = np.histogram(req, edges)
    tok_in, _ = np.histogram(req, edges, weights=nin)
    tok_out, _ = np.histogram(req, edges, weights=nout)
    return cnt.astype(np.float64), tok_in.astype(np.float64), tok_out.astype(np.float64)


def _add_interval_weight(diff: np.ndarray, centers: np.ndarray, start: float, end: float, weight: float) -> None:
    if weight <= 0.0:
        return
    n = int(centers.size)
    if n <= 0:
        return
    s = int(np.searchsorted(centers, start, side="left"))
    e = int(np.searchsorted(centers, end, side="right") - 1)
    if e < 0 or s >= n:
        return
    left = max(0, min(n - 1, s))
    right = max(0, min(n - 1, e))
    if right < left:
        return
    diff[left] += float(weight)
    diff[right + 1] -= float(weight)


def reconstruct_phase_tokens(
    bin_ts: np.ndarray,
    requests: Sequence[Mapping[str, float]],
    lambda_prefill: float,
    lambda_decode: float,
) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(bin_ts, dtype=np.float64).reshape(-1)
    n = int(centers.size)
    if n <= 0:
        z = np.zeros((0,), dtype=np.float64)
        return z, z

    lp = float(lambda_prefill)
    ld = float(lambda_decode)
    if lp <= 0.0 or ld <= 0.0:
        raise ValueError("Throughput must be positive for phase token reconstruction")

    pre_diff = np.zeros((n + 1,), dtype=np.float64)
    dec_diff = np.zeros((n + 1,), dtype=np.float64)

    for req in requests:
        arr = float(req.get("arrival_time", 0.0))
        nin = max(0.0, float(req.get("input_tokens", 0.0)))
        nout = max(0.0, float(req.get("output_tokens", 0.0)))

        prefill_start = arr
        prefill_end = prefill_start + (nin / lp)
        decode_start = prefill_end
        decode_end = decode_start + (nout / ld)

        _add_interval_weight(pre_diff, centers, prefill_start, prefill_end, nin)
        _add_interval_weight(dec_diff, centers, decode_start, decode_end, nout)

    prefill_tokens = np.cumsum(pre_diff[:-1])
    decode_tokens = np.cumsum(dec_diff[:-1])
    prefill_tokens = np.clip(prefill_tokens, a_min=0.0, a_max=None)
    decode_tokens = np.clip(decode_tokens, a_min=0.0, a_max=None)
    return prefill_tokens.astype(np.float64), decode_tokens.astype(np.float64)


def entropy_from_labels(labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if y.size == 0:
        return float("nan")
    _, inv = np.unique(y, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    p = counts / max(float(np.sum(counts)), 1e-12)
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def _quantile_discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)

    q = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    edges = np.quantile(arr, q)
    edges = np.unique(edges)
    if edges.size <= 1:
        return np.zeros((arr.size,), dtype=np.int64)

    bins = np.digitize(arr, edges[1:-1], right=False).astype(np.int64)
    bins = np.clip(bins, a_min=0, a_max=max(0, edges.size - 2))
    return bins


def _encode_joint_bins(x_disc: np.ndarray) -> np.ndarray:
    x = np.asarray(x_disc, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("x_disc must be 2D")
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    bases = []
    for j in range(x.shape[1]):
        mx = int(np.max(x[:, j], initial=0))
        bases.append(mx + 1)

    state = np.zeros((x.shape[0],), dtype=np.int64)
    mult = 1
    for j, base in enumerate(bases):
        state += x[:, j] * mult
        mult *= max(1, base)
    return state


def estimate_plugin_mi_nmi(x: np.ndarray, z: np.ndarray, n_bins: int) -> Dict[str, float]:
    X = np.asarray(x, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(z, dtype=np.int64).reshape(-1)
    if X.shape[0] != y.size:
        raise ValueError("X and z length mismatch")
    if X.shape[0] == 0:
        return {"mi": float("nan"), "nmi": float("nan"), "h_z": float("nan")}

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    if X.shape[0] == 0:
        return {"mi": float("nan"), "nmi": float("nan"), "h_z": float("nan")}

    h_z = entropy_from_labels(y)
    if not np.isfinite(h_z) or h_z <= 0.0:
        return {"mi": 0.0, "nmi": 0.0, "h_z": float(0.0)}

    discs = []
    for j in range(X.shape[1]):
        discs.append(_quantile_discretize(X[:, j], n_bins=int(n_bins)))
    x_disc = np.stack(discs, axis=1)
    x_state = _encode_joint_bins(x_disc)

    cls, z_inv = np.unique(y, return_inverse=True)
    n_classes = int(cls.size)
    n_joint = int(np.max(x_state, initial=-1) + 1)

    counts_joint = np.zeros((n_joint, n_classes), dtype=np.float64)
    np.add.at(counts_joint, (x_state, z_inv), 1.0)

    total = float(np.sum(counts_joint))
    if total <= 0.0:
        return {"mi": float("nan"), "nmi": float("nan"), "h_z": float(h_z)}

    p_xz = counts_joint / total
    p_x = np.sum(p_xz, axis=1, keepdims=True)
    p_z = np.sum(p_xz, axis=0, keepdims=True)

    denom = p_x @ p_z
    mask_nz = p_xz > 0.0
    mi = float(np.sum(p_xz[mask_nz] * np.log(p_xz[mask_nz] / np.clip(denom[mask_nz], 1e-12, None))))
    if mi < 0.0 and mi > -1e-12:
        mi = 0.0
    mi = float(np.clip(mi, 0.0, h_z))
    nmi = float(np.clip(mi / h_z, 0.0, 1.0))
    return {"mi": mi, "nmi": nmi, "h_z": float(h_z)}


def _knn_posterior_probs(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    classes: np.ndarray,
    class_to_idx: Mapping[int, int],
    knn_k: int,
) -> np.ndarray:
    xt = np.asarray(x_train, dtype=np.float64)
    yt = np.asarray(y_train, dtype=np.int64).reshape(-1)
    xv = np.asarray(x_test, dtype=np.float64)
    cls = np.asarray(classes, dtype=np.int64).reshape(-1)

    if xt.ndim == 1:
        xt = xt.reshape(-1, 1)
    if xv.ndim == 1:
        xv = xv.reshape(-1, 1)
    if xt.shape[0] != yt.size:
        raise ValueError("kNN: x_train and y_train length mismatch")
    if xt.shape[1] != xv.shape[1]:
        raise ValueError("kNN: train/test feature dims mismatch")
    if xt.shape[0] <= 0 or xv.shape[0] <= 0 or cls.size <= 0:
        return np.zeros((xv.shape[0], max(0, cls.size)), dtype=np.float64)

    y_train_idx = np.array([class_to_idx.get(int(v), -1) for v in yt.tolist()], dtype=np.int64)
    valid_train = y_train_idx >= 0
    xt = xt[valid_train]
    y_train_idx = y_train_idx[valid_train]
    if xt.shape[0] <= 0:
        return np.full((xv.shape[0], cls.size), 1.0 / float(max(1, cls.size)), dtype=np.float64)

    k = int(min(max(1, knn_k), xt.shape[0]))
    tree = cKDTree(xt)
    dist, nn_idx = tree.query(xv, k=k, workers=1)
    if k == 1:
        dist = dist.reshape(-1, 1)
        nn_idx = nn_idx.reshape(-1, 1)

    y_nn = y_train_idx[nn_idx]
    weights = 1.0 / np.clip(np.asarray(dist, dtype=np.float64), 1e-12, None)

    exact_mask = np.asarray(dist, dtype=np.float64) <= 1e-12
    has_exact = np.any(exact_mask, axis=1)
    if np.any(has_exact):
        exact_weights = exact_mask[has_exact].astype(np.float64)
        exact_weights = exact_weights / np.clip(np.sum(exact_weights, axis=1, keepdims=True), 1e-12, None)
        weights[has_exact] = exact_weights

    out = np.zeros((xv.shape[0], cls.size), dtype=np.float64)
    for c_idx in range(cls.size):
        mask_c = (y_nn == c_idx).astype(np.float64)
        out[:, c_idx] = np.sum(weights * mask_c, axis=1)
    out = out / np.clip(np.sum(out, axis=1, keepdims=True), 1e-12, None)
    return out


def estimate_knn_joint_mi_nmi(
    feature_traces: Sequence[np.ndarray],
    label_traces: Sequence[np.ndarray],
    knn_k: int,
) -> Dict[str, float]:
    if len(feature_traces) == 0 or len(label_traces) == 0 or len(feature_traces) != len(label_traces):
        return {"mi": float("nan"), "nmi": float("nan"), "h_z": float("nan")}

    pooled_y = np.concatenate([np.asarray(y, dtype=np.int64).reshape(-1) for y in label_traces], axis=0)
    h_z = entropy_from_labels(pooled_y)
    if not np.isfinite(h_z) or h_z <= 0.0:
        return {"mi": 0.0, "nmi": 0.0, "h_z": float(0.0)}

    classes = np.unique(pooled_y)
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}

    total_loss = 0.0
    total_n = 0

    for i in range(len(feature_traces)):
        x_test = np.asarray(feature_traces[i], dtype=np.float64)
        y_test = np.asarray(label_traces[i], dtype=np.int64).reshape(-1)
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        if x_test.shape[0] != y_test.size or y_test.size == 0:
            continue

        x_train_parts: List[np.ndarray] = []
        y_train_parts: List[np.ndarray] = []
        for j in range(len(feature_traces)):
            if j == i:
                continue
            xj = np.asarray(feature_traces[j], dtype=np.float64)
            yj = np.asarray(label_traces[j], dtype=np.int64).reshape(-1)
            if xj.ndim == 1:
                xj = xj.reshape(-1, 1)
            if xj.shape[0] != yj.size or yj.size == 0:
                continue
            x_train_parts.append(xj)
            y_train_parts.append(yj)

        if len(x_train_parts) == 0:
            continue

        x_train = np.concatenate(x_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        if x_train.shape[0] <= 0:
            continue

        mu = np.mean(x_train, axis=0, keepdims=True)
        sd = np.std(x_train, axis=0, keepdims=True) + 1e-6
        x_train_n = (x_train - mu) / sd
        x_test_n = (x_test - mu) / sd

        p_full = _knn_posterior_probs(
            x_train=x_train_n,
            y_train=y_train,
            x_test=x_test_n,
            classes=classes,
            class_to_idx=class_to_idx,
            knn_k=int(knn_k),
        )

        y_idx = np.array([class_to_idx.get(int(v), -1) for v in y_test.tolist()], dtype=np.int64)
        valid = y_idx >= 0
        if not np.any(valid):
            continue

        probs_true = p_full[np.arange(p_full.shape[0])[valid], y_idx[valid]]
        probs_true = np.clip(probs_true, 1e-12, 1.0)

        total_loss += float(-np.sum(np.log(probs_true)))
        total_n += int(np.sum(valid))

    if total_n <= 0:
        return {"mi": float("nan"), "nmi": float("nan"), "h_z": float(h_z)}

    ce = float(total_loss / float(total_n))
    mi = float(h_z - ce)
    mi = float(np.clip(mi, 0.0, h_z))
    nmi = float(np.clip(mi / h_z, 0.0, 1.0))
    return {"mi": mi, "nmi": nmi, "h_z": float(h_z)}


def _load_throughput_entry(throughput_payload: Mapping[str, object], config_id: str) -> Optional[Dict[str, float]]:
    cfgs = throughput_payload.get("configs", {})
    if not isinstance(cfgs, dict):
        return None
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        return None

    prefill = _to_float(row.get("prefill_rate_median_toks_per_s"))
    decode = _to_float(row.get("decode_rate_median_toks_per_s"))
    if prefill is None or decode is None or prefill <= 0.0 or decode <= 0.0:
        return None
    return {
        "lambda_prefill": float(prefill),
        "lambda_decode": float(decode),
    }


def _extract_trace_row(obj: np.ndarray, idx: int) -> np.ndarray:
    return np.asarray(obj[idx], dtype=np.float64).reshape(-1)


def _determine_default_config_ids() -> List[str]:
    return [str(v["config_id"]) for v in BIC_CONFIGS.values()]


def _determine_config_labels(config_ids: Sequence[str]) -> Dict[str, str]:
    by_id: Dict[str, str] = {}
    for _, spec in BIC_CONFIGS.items():
        by_id[str(spec["config_id"])] = str(spec.get("title", spec["config_id"]))
    out: Dict[str, str] = {}
    for cid in config_ids:
        out[str(cid)] = by_id.get(str(cid), str(cid))
    return out


def _resolve_config_ids(raw: Optional[str]) -> List[str]:
    if raw is None or str(raw).strip() == "":
        return _determine_default_config_ids()
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _classify_request_error(message: str) -> str:
    text = str(message)
    if "request_timestamps" in text:
        return "missing_recorded_timestamps"
    return "request_json_invalid"


def _build_trace_features(
    *,
    power: np.ndarray,
    active_requests: np.ndarray,
    dt: float,
    gmm_payload: Mapping[str, object],
    requests: Sequence[Mapping[str, float]],
    throughput: Mapping[str, float],
) -> Optional[TraceFeatures]:
    pw = np.asarray(power, dtype=np.float64).reshape(-1)
    active = np.asarray(active_requests, dtype=np.float64).reshape(-1)

    n = int(min(pw.size, active.size))
    if n <= 2:
        return None
    pw = pw[:n]
    active = active[:n]

    if not np.all(np.isfinite(pw)) or not np.all(np.isfinite(active)):
        return None

    centers = np.arange(n, dtype=np.float64) * float(dt)

    req_ts = np.asarray([float(r["arrival_time"]) for r in requests], dtype=np.float64)
    in_tok = np.asarray([float(r["input_tokens"]) for r in requests], dtype=np.float64)
    out_tok = np.asarray([float(r["output_tokens"]) for r in requests], dtype=np.float64)

    valid_req = np.isfinite(req_ts) & np.isfinite(in_tok) & np.isfinite(out_tok)
    req_ts = req_ts[valid_req]
    in_tok = in_tok[valid_req]
    out_tok = out_tok[valid_req]

    cnt, tok_in, tok_out = histogram_requests(centers, req_ts, in_tok, out_tok)

    prefill_tokens, decode_tokens = reconstruct_phase_tokens(
        centers,
        requests=requests,
        lambda_prefill=float(throughput["lambda_prefill"]),
        lambda_decode=float(throughput["lambda_decode"]),
    )

    z = predict_regime_labels_from_gmm(pw[1:], gmm_payload)
    a_t = active[1:]
    delta = active[1:] - active[:-1]
    f2 = np.stack([a_t, delta], axis=1)

    f6_full = np.stack([cnt, tok_in, tok_out, active, prefill_tokens, decode_tokens], axis=1)
    f6 = f6_full[1:, :]

    L = int(min(z.size, a_t.size, delta.size, f2.shape[0], f6.shape[0]))
    if L <= 0:
        return None

    z = z[:L].astype(np.int64)
    a_t = a_t[:L].astype(np.float64)
    delta = delta[:L].astype(np.float64)
    f2 = f2[:L, :].astype(np.float64)
    f6 = f6[:L, :].astype(np.float64)

    return TraceFeatures(
        z=z,
        a_t=a_t,
        delta_a_t=delta,
        f2=f2,
        f6=f6,
    )


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={arr.shape}")
    return arr


def _compute_feature_norm(feature_traces: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if len(feature_traces) == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    cat = np.concatenate([_as_2d(x).astype(np.float64) for x in feature_traces], axis=0)
    mean = np.mean(cat, axis=0)
    std = np.std(cat, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_feature_norm(feature_traces: Sequence[np.ndarray], mean: np.ndarray, std: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    m = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    s = np.asarray(std, dtype=np.float32).reshape(1, -1)
    for x in feature_traces:
        arr = _as_2d(x).astype(np.float32)
        if arr.shape[1] != m.shape[1]:
            raise ValueError(f"Feature dimension mismatch: got {arr.shape[1]} expected {m.shape[1]}")
        out.append(((arr - m) / s).astype(np.float32))
    return out


def _resolve_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _resolve_legacy_f6_checkpoint(config_id: str, legacy_dirs: Sequence[str]) -> Optional[Path]:
    try:
        model_name, hw, tp = parse_config_id(config_id)
    except Exception:
        return None
    filename = f"{model_name}_{hw.lower()}_tp{int(tp)}.pt"
    for base in legacy_dirs:
        candidate = Path(base) / filename
        if candidate.exists():
            return candidate
    return None


def _load_legacy_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    try:
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    out: Dict[str, torch.Tensor] = {}
    for k, v in payload.items():
        if isinstance(k, str) and torch.is_tensor(v):
            out[k] = v.detach().cpu()
    return out


def _flatten_labels(label_traces: Sequence[np.ndarray]) -> np.ndarray:
    if len(label_traces) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate([np.asarray(y, dtype=np.int64).reshape(-1) for y in label_traces], axis=0)


def _train_sequence_classifier(
    *,
    train_x: Sequence[np.ndarray],
    train_y: Sequence[np.ndarray],
    val_x: Sequence[np.ndarray],
    val_y: Sequence[np.ndarray],
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
    device: torch.device,
    init_state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = GRUClassifier(Dx=int(input_dim), K=int(num_classes), H=int(hidden_dim), num_layers=int(max(1, num_layers))).to(device)
    loaded_init_keys = 0
    if init_state_dict is not None:
        model_state = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        for k, v in init_state_dict.items():
            if k not in model_state:
                continue
            if not torch.is_tensor(v):
                continue
            if tuple(v.shape) != tuple(model_state[k].shape):
                continue
            filtered[k] = v.detach().cpu()
        if len(filtered) > 0:
            model.load_state_dict(filtered, strict=False)
            loaded_init_keys = int(len(filtered))

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    epochs_without_improve = 0
    history: List[Dict[str, float]] = []

    idx_order = np.arange(len(train_x), dtype=np.int64)
    for epoch in range(int(max(1, epochs))):
        rng = np.random.default_rng(int(seed) + int(epoch))
        rng.shuffle(idx_order)

        model.train()
        train_losses: List[float] = []
        for idx in idx_order.tolist():
            x_np = _as_2d(train_x[idx]).astype(np.float32)
            y_np = np.asarray(train_y[idx], dtype=np.int64).reshape(-1)
            if x_np.shape[0] == 0 or y_np.size == 0 or x_np.shape[0] != y_np.size:
                continue
            if np.min(y_np) < 0 or np.max(y_np) >= int(num_classes):
                continue
            x = torch.tensor(x_np.tolist(), dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.tensor(y_np.tolist(), dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            loss = criterion(logits.reshape(-1, int(num_classes)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for x_np, y_np in zip(val_x, val_y):
                x_arr = _as_2d(x_np).astype(np.float32)
                y_arr = np.asarray(y_np, dtype=np.int64).reshape(-1)
                if x_arr.shape[0] == 0 or y_arr.size == 0 or x_arr.shape[0] != y_arr.size:
                    continue
                if np.min(y_arr) < 0 or np.max(y_arr) >= int(num_classes):
                    continue
                x = torch.tensor(x_arr.tolist(), dtype=torch.float32, device=device).unsqueeze(0)
                y = torch.tensor(y_arr.tolist(), dtype=torch.long, device=device).unsqueeze(0)
                logits = model(x)
                loss = criterion(logits.reshape(-1, int(num_classes)), y.reshape(-1))
                val_losses.append(float(loss.item()))

        mean_train = float(np.mean(train_losses)) if len(train_losses) > 0 else float("nan")
        mean_val = float(np.mean(val_losses)) if len(val_losses) > 0 else float("nan")
        history.append({"epoch": float(epoch), "train_ce": mean_train, "val_ce": mean_val})

        if np.isfinite(mean_val) and mean_val < best_val:
            best_val = float(mean_val)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= int(max(1, patience)):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_ce": float(best_val),
        "best_epoch": int(best_epoch),
        "history": history,
        "loaded_init_keys": int(loaded_init_keys),
    }


def _evaluate_sequence_classifier(
    *,
    model: GRUClassifier,
    test_x: Sequence[np.ndarray],
    test_y: Sequence[np.ndarray],
    num_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    criterion_sum = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_n = 0
    total_correct = 0
    n_used_traces = 0

    model.eval()
    with torch.no_grad():
        for x_np, y_np in zip(test_x, test_y):
            x_arr = _as_2d(x_np).astype(np.float32)
            y_arr = np.asarray(y_np, dtype=np.int64).reshape(-1)
            if x_arr.shape[0] == 0 or y_arr.size == 0 or x_arr.shape[0] != y_arr.size:
                continue
            valid = (y_arr >= 0) & (y_arr < int(num_classes))
            if not np.all(valid):
                x_arr = x_arr[valid]
                y_arr = y_arr[valid]
            if y_arr.size == 0:
                continue
            x = torch.tensor(x_arr.tolist(), dtype=torch.float32, device=device).unsqueeze(0)
            y = torch.tensor(y_arr.tolist(), dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            loss = criterion_sum(logits.reshape(-1, int(num_classes)), y.reshape(-1))
            pred = torch.argmax(logits, dim=-1)
            correct = int((pred.reshape(-1) == y.reshape(-1)).sum().item())

            total_loss += float(loss.item())
            total_n += int(y_arr.size)
            total_correct += int(correct)
            n_used_traces += 1

    if total_n <= 0:
        return {
            "test_cross_entropy": float("nan"),
            "test_accuracy": float("nan"),
            "n_test_samples": 0.0,
            "n_test_traces_used": 0.0,
        }

    return {
        "test_cross_entropy": float(total_loss / float(total_n)),
        "test_accuracy": float(total_correct / float(total_n)),
        "n_test_samples": float(total_n),
        "n_test_traces_used": float(n_used_traces),
    }


def _compute_null_baseline(
    *,
    train_labels: Sequence[np.ndarray],
    test_labels: Sequence[np.ndarray],
    num_classes: int,
) -> Dict[str, float]:
    y_train = _flatten_labels(train_labels)
    y_test = _flatten_labels(test_labels)
    if y_train.size == 0 or y_test.size == 0 or int(num_classes) <= 0:
        return {"null_test_cross_entropy": float("nan"), "null_test_accuracy": float("nan")}

    train_valid = (y_train >= 0) & (y_train < int(num_classes))
    y_train = y_train[train_valid]
    if y_train.size == 0:
        return {"null_test_cross_entropy": float("nan"), "null_test_accuracy": float("nan")}

    counts = np.bincount(y_train, minlength=int(num_classes)).astype(np.float64)
    probs = counts / np.clip(np.sum(counts), 1e-12, None)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / np.sum(probs)

    test_valid = (y_test >= 0) & (y_test < int(num_classes))
    y_test = y_test[test_valid]
    if y_test.size == 0:
        return {"null_test_cross_entropy": float("nan"), "null_test_accuracy": float("nan")}

    ce = float(-np.mean(np.log(probs[y_test])))
    acc = float(np.max(probs))
    return {"null_test_cross_entropy": ce, "null_test_accuracy": acc}


def _plot_accuracy_comparison(
    *,
    rows: Sequence[Mapping[str, object]],
    out_path: str,
) -> None:
    if len(rows) == 0:
        raise ValueError("No rows provided for plotting")

    config_ids: List[str] = []
    seen = set()
    for row in rows:
        cid = str(row["config_id"])
        if cid in seen:
            continue
        seen.add(cid)
        config_ids.append(cid)

    labels_by_id = {str(r["config_id"]): str(r["config_label"]) for r in rows}
    acc_map: Dict[str, Dict[str, float]] = {}
    for row in rows:
        cid = str(row["config_id"])
        subset = str(row["subset"])
        acc_map.setdefault(cid, {})[subset] = float(row["test_accuracy"])

    f2_vals = np.asarray([acc_map.get(cid, {}).get("F2", float("nan")) for cid in config_ids], dtype=np.float64)
    f6_vals = np.asarray([acc_map.get(cid, {}).get("Full-6D", float("nan")) for cid in config_ids], dtype=np.float64)

    x = np.arange(len(config_ids), dtype=np.float64)
    width = 0.32
    fig, ax = plt.subplots(figsize=(max(7.5, 2.0 * len(config_ids)), 4.8))

    ax.bar(
        x - (0.5 * width),
        f2_vals,
        width=width,
        color="#4e8fc6",
        edgecolor="black",
        linewidth=0.4,
        label="F2 (A_t, ΔA_t)",
    )
    ax.bar(
        x + (0.5 * width),
        f6_vals,
        width=width,
        color="#1f4e79",
        edgecolor="black",
        linewidth=0.4,
        label="Full-6D",
    )

    for i, (a2, a6) in enumerate(zip(f2_vals.tolist(), f6_vals.tolist())):
        if np.isfinite(a2):
            ax.text(x[i] - (0.5 * width), a2 + 0.01, f"{a2:.3f}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(a6):
            ax.text(x[i] + (0.5 * width), a6 + 0.01, f"{a6:.3f}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(a2) and np.isfinite(a6) and a6 > 0.0:
            ratio = float(a2 / a6)
            ax.text(x[i], min(0.99, max(a2, a6) + 0.04), f"{100.0 * ratio:.1f}% of F6", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([labels_by_id.get(cid, cid) for cid in config_ids], rotation=12, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Held-out State Prediction Accuracy")
    ax.set_xlabel("Configuration")
    ax.set_title("Figure D1: BiGRU Sequence Classification (F2 vs Full-6D)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)

    _ensure_parent(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_d1_conditional_entropy(
    *,
    experimental_manifest: str,
    run_manifest: str,
    pair_manifest_csv: str,
    throughput_db: str,
    config_ids: Optional[str],
    seed: int,
    require_recorded_timestamps: bool,
    min_valid_traces_per_config: int,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    lr: float,
    patience: int,
    device: str,
    use_legacy_f6_init: bool,
    legacy_f6_weights_dirs: Sequence[str],
    out_figure: str,
    out_per_config_csv: str,
    out_json: str,
) -> Dict[str, object]:
    config_list = _resolve_config_ids(config_ids)
    labels = _determine_config_labels(config_list)
    resolved_device = _resolve_device(device)
    resolved_legacy_dirs = [str((Path(d) if Path(d).is_absolute() else (_repo_root() / d)).resolve()) for d in legacy_f6_weights_dirs]

    exp_manifest = _load_json(experimental_manifest)
    run_manifest_payload = _load_json(run_manifest)
    throughput_payload = _load_json(throughput_db)
    pair_map = load_pair_manifest_map(pair_manifest_csv)

    exp_cfgs = exp_manifest.get("configs", {})
    run_cfgs = run_manifest_payload.get("configs", {})
    if not isinstance(exp_cfgs, dict):
        raise ValueError("experimental manifest missing configs dict")
    if not isinstance(run_cfgs, dict):
        raise ValueError("run manifest missing configs dict")

    rows: List[Dict[str, object]] = []
    selected_configs: List[Dict[str, object]] = []
    skipped_configs: Dict[str, Dict[str, object]] = {}
    split_names = ["train", "val", "test"]

    for config_id in config_list:
        label = labels.get(config_id, config_id)
        exp_row = exp_cfgs.get(config_id)
        if not isinstance(exp_row, dict):
            skipped_configs[config_id] = {"reason": "missing_dataset", "config_label": label}
            continue

        dataset_npz = exp_row.get("dataset_npz")
        split_json = exp_row.get("split_json")
        if not isinstance(dataset_npz, str) or dataset_npz.strip() == "":
            skipped_configs[config_id] = {"reason": "missing_dataset", "config_label": label}
            continue
        if not isinstance(split_json, str) or split_json.strip() == "":
            skipped_configs[config_id] = {"reason": "missing_dataset", "config_label": label, "detail": "missing split_json"}
            continue

        dataset_path = _resolve_existing_path(dataset_npz, str(Path(experimental_manifest).resolve().parent))
        split_path = _resolve_existing_path(split_json, str(Path(experimental_manifest).resolve().parent))
        if dataset_path is None:
            skipped_configs[config_id] = {"reason": "missing_dataset", "config_label": label}
            continue
        if split_path is None:
            skipped_configs[config_id] = {"reason": "missing_dataset", "config_label": label, "detail": "split_json unresolved"}
            continue

        run_row = run_cfgs.get(config_id)
        if not isinstance(run_row, dict):
            skipped_configs[config_id] = {"reason": "missing_gmm", "config_label": label}
            continue
        gmm_raw = run_row.get("gmm_params_path")
        if not isinstance(gmm_raw, str) or gmm_raw.strip() == "":
            skipped_configs[config_id] = {"reason": "missing_gmm", "config_label": label}
            continue

        gmm_path = _resolve_existing_path(gmm_raw, str(Path(run_manifest).resolve().parent))
        if gmm_path is None:
            k_guess = int(run_row.get("k", 10))
            slug = _safe_slug(config_id)
            fallback = Path(run_manifest).resolve().parent / "gmms" / f"{slug}_k{k_guess}.json"
            if fallback.exists():
                gmm_path = str(fallback)
        if gmm_path is None:
            skipped_configs[config_id] = {"reason": "missing_gmm", "config_label": label}
            continue

        throughput = _load_throughput_entry(throughput_payload, config_id)
        if throughput is None:
            skipped_configs[config_id] = {"reason": "missing_throughput", "config_label": label}
            continue

        split_payload = _load_json(split_path)
        gmm_payload = _load_json(gmm_path)

        trace_reason_counts: Dict[str, Dict[str, int]] = {
            split: {
                "missing_request_json": 0,
                "missing_recorded_timestamps": 0,
                "request_json_invalid": 0,
                "insufficient_samples": 0,
            }
            for split in split_names
        }
        split_total: Dict[str, int] = {split: 0 for split in split_names}
        split_recorded: Dict[str, int] = {split: 0 for split in split_names}
        split_traces: Dict[str, List[TraceFeatures]] = {split: [] for split in split_names}

        with np.load(dataset_path, allow_pickle=True) as data:
            pair_keys = np.asarray(data["pair_key"], dtype=object)
            power_arr = np.asarray(data["power"], dtype=object)
            active_arr = np.asarray(data["active_requests"], dtype=object)
            power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64).reshape(-1)
            dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)

            if dt_arr.size == 0:
                skipped_configs[config_id] = {"reason": "insufficient_samples", "config_label": label, "detail": "missing dt"}
                continue
            dt = float(dt_arr[0])
            n_traces = int(min(pair_keys.size, power_arr.size, active_arr.size, power_start_arr.size))

            split_indices: Dict[str, List[int]] = {}
            for split_name in split_names:
                raw_indices = split_payload.get(f"{split_name}_indices", [])
                if not isinstance(raw_indices, list):
                    skipped_configs[config_id] = {
                        "reason": "missing_dataset",
                        "config_label": label,
                        "detail": f"invalid {split_name}_indices",
                    }
                    split_indices = {}
                    break
                split_indices[split_name] = [int(i) for i in raw_indices if isinstance(i, int) and 0 <= int(i) < n_traces]
            if len(split_indices) == 0:
                continue

            for split_name in split_names:
                indices = split_indices[split_name]
                split_total[split_name] = int(len(indices))
                for idx in indices:
                    pair_key = str(pair_keys[idx])
                    power = _extract_trace_row(power_arr, idx)
                    active = _extract_trace_row(active_arr, idx)
                    power_start_epoch_s = float(power_start_arr[idx])

                    json_path: Optional[Path] = None
                    stage1 = pair_map.get(pair_key)
                    if stage1 is not None:
                        json_path = Path(stage1)
                    if json_path is None or (not json_path.exists()):
                        json_path = deterministic_pairkey_json_path(config_id, pair_key)
                    if json_path is None or (not json_path.exists()):
                        trace_reason_counts[split_name]["missing_request_json"] += 1
                        continue

                    try:
                        raw_payload = _load_json(json_path)
                        has_recorded = isinstance(raw_payload.get("request_timestamps"), list) and len(raw_payload.get("request_timestamps", [])) > 0
                        if has_recorded:
                            split_recorded[split_name] += 1
                    except Exception:
                        trace_reason_counts[split_name]["request_json_invalid"] += 1
                        continue

                    try:
                        requests = build_requests_from_json(
                            json_path,
                            power_start_epoch_s=power_start_epoch_s,
                            trace_duration_s=float(max(0, len(power) - 1) * dt),
                            dt=dt,
                            require_recorded_timestamps=bool(require_recorded_timestamps),
                        )
                    except Exception as exc:
                        reason = _classify_request_error(str(exc))
                        trace_reason_counts[split_name][reason] = trace_reason_counts[split_name].get(reason, 0) + 1
                        continue

                    tr = _build_trace_features(
                        power=power,
                        active_requests=active,
                        dt=dt,
                        gmm_payload=gmm_payload,
                        requests=requests,
                        throughput=throughput,
                    )
                    if tr is None or tr.z.size <= 0:
                        trace_reason_counts[split_name]["insufficient_samples"] += 1
                        continue
                    split_traces[split_name].append(tr)

        min_req = int(max(1, min_valid_traces_per_config))
        split_summary = {
            split: {
                "n_valid_traces": int(len(split_traces[split])),
                "n_total_split_traces": int(split_total[split]),
                "recorded_timestamp_fraction": float(split_recorded[split] / max(1, split_total[split])),
                "trace_reason_counts": trace_reason_counts[split],
            }
            for split in split_names
        }
        if any(split_total[s] <= 0 for s in split_names):
            skipped_configs[config_id] = {
                "reason": "insufficient_samples",
                "config_label": label,
                "detail": "empty split indices",
                "split_summary": split_summary,
            }
            continue
        if any(len(split_traces[s]) < min_req for s in split_names):
            skipped_configs[config_id] = {
                "reason": "insufficient_valid_traces",
                "config_label": label,
                "split_summary": split_summary,
            }
            continue

        all_labels = _flatten_labels([t.z for s in split_names for t in split_traces[s]])
        if all_labels.size == 0 or np.unique(all_labels).size < 2:
            skipped_configs[config_id] = {
                "reason": "insufficient_samples",
                "config_label": label,
                "detail": "single class labels after filtering",
                "split_summary": split_summary,
            }
            continue

        num_classes = int(max(int(gmm_payload.get("k", 0)), int(np.max(all_labels)) + 1))
        if num_classes < 2:
            skipped_configs[config_id] = {
                "reason": "insufficient_samples",
                "config_label": label,
                "detail": "num_classes < 2",
                "split_summary": split_summary,
            }
            continue

        legacy_info: Dict[str, object] = {
            "enabled": bool(use_legacy_f6_init),
            "checkpoint_path": None,
            "status": "disabled",
            "loaded_state_keys": 0,
        }
        legacy_state: Optional[Dict[str, torch.Tensor]] = None
        if bool(use_legacy_f6_init):
            legacy_ckpt = _resolve_legacy_f6_checkpoint(config_id, resolved_legacy_dirs)
            if legacy_ckpt is None:
                legacy_info["status"] = "missing_checkpoint"
            else:
                legacy_info["checkpoint_path"] = str(legacy_ckpt)
                try:
                    legacy_state = _load_legacy_state_dict(legacy_ckpt)
                    if len(legacy_state) > 0:
                        legacy_info["status"] = "loaded"
                        legacy_info["loaded_state_keys"] = int(len(legacy_state))
                    else:
                        legacy_info["status"] = "empty_state_dict"
                except Exception as exc:
                    legacy_info["status"] = f"load_failed:{type(exc).__name__}"

        subset_metrics: Dict[str, Dict[str, float]] = {}
        for subset in SEQ_SUBSET_ORDER:
            train_feats = [t.f2 if subset == "F2" else t.f6 for t in split_traces["train"]]
            val_feats = [t.f2 if subset == "F2" else t.f6 for t in split_traces["val"]]
            test_feats = [t.f2 if subset == "F2" else t.f6 for t in split_traces["test"]]
            train_labels = [t.z for t in split_traces["train"]]
            val_labels = [t.z for t in split_traces["val"]]
            test_labels = [t.z for t in split_traces["test"]]

            feat_mean, feat_std = _compute_feature_norm(train_feats)
            train_x = _apply_feature_norm(train_feats, feat_mean, feat_std)
            val_x = _apply_feature_norm(val_feats, feat_mean, feat_std)
            test_x = _apply_feature_norm(test_feats, feat_mean, feat_std)

            input_dim = int(train_x[0].shape[1]) if len(train_x) > 0 else int(feat_mean.size)
            subset_seed = int(seed) + (0 if subset == "F2" else 10000)
            train_result = _train_sequence_classifier(
                train_x=train_x,
                train_y=train_labels,
                val_x=val_x,
                val_y=val_labels,
                input_dim=input_dim,
                num_classes=int(num_classes),
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                epochs=int(epochs),
                lr=float(lr),
                patience=int(patience),
                seed=int(subset_seed),
                device=resolved_device,
                init_state_dict=(legacy_state if subset == "Full-6D" else None),
            )
            eval_result = _evaluate_sequence_classifier(
                model=train_result["model"],
                test_x=test_x,
                test_y=test_labels,
                num_classes=int(num_classes),
                device=resolved_device,
            )
            subset_metrics[subset] = {
                "best_val_ce": float(train_result["best_val_ce"]),
                "best_epoch": float(train_result["best_epoch"]),
                "test_cross_entropy": float(eval_result["test_cross_entropy"]),
                "test_accuracy": float(eval_result["test_accuracy"]),
                "n_test_samples": float(eval_result["n_test_samples"]),
                "n_test_traces_used": float(eval_result["n_test_traces_used"]),
                "input_dim": float(input_dim),
                "loaded_init_keys": float(train_result["loaded_init_keys"]),
            }
            if subset == "Full-6D":
                legacy_info["loaded_state_keys"] = int(train_result["loaded_init_keys"])
                if int(train_result["loaded_init_keys"]) > 0:
                    legacy_info["status"] = "initialized"

        null_metrics = _compute_null_baseline(
            train_labels=[t.z for t in split_traces["train"]],
            test_labels=[t.z for t in split_traces["test"]],
            num_classes=int(num_classes),
        )
        acc_f2 = float(subset_metrics["F2"]["test_accuracy"])
        acc_f6 = float(subset_metrics["Full-6D"]["test_accuracy"])
        ce_f2 = float(subset_metrics["F2"]["test_cross_entropy"])
        ce_f6 = float(subset_metrics["Full-6D"]["test_cross_entropy"])
        ce_null = float(null_metrics["null_test_cross_entropy"])

        acc_ratio_vs_f6 = float("nan")
        if np.isfinite(acc_f2) and np.isfinite(acc_f6) and acc_f6 > 0.0:
            acc_ratio_vs_f6 = float(acc_f2 / acc_f6)

        ce_retention_vs_f6 = float("nan")
        if np.isfinite(ce_null) and np.isfinite(ce_f2) and np.isfinite(ce_f6):
            denom = float(ce_null - ce_f6)
            if abs(denom) > 1e-12:
                ce_retention_vs_f6 = float((ce_null - ce_f2) / denom)

        pass_acc = bool(np.isfinite(acc_ratio_vs_f6) and acc_ratio_vs_f6 >= 0.95)
        pass_ce = bool(np.isfinite(ce_retention_vs_f6) and ce_retention_vs_f6 >= 0.95)

        for subset in SEQ_SUBSET_ORDER:
            row_metrics = subset_metrics[subset]
            rows.append(
                {
                    "config_id": config_id,
                    "config_label": label,
                    "subset": subset,
                    "test_cross_entropy": float(row_metrics["test_cross_entropy"]),
                    "test_accuracy": float(row_metrics["test_accuracy"]),
                    "best_val_cross_entropy": float(row_metrics["best_val_ce"]),
                    "best_epoch": int(row_metrics["best_epoch"]),
                    "input_dim": int(row_metrics["input_dim"]),
                    "n_classes": int(num_classes),
                    "n_train_traces": int(len(split_traces["train"])),
                    "n_val_traces": int(len(split_traces["val"])),
                    "n_test_traces": int(len(split_traces["test"])),
                    "n_test_samples": int(row_metrics["n_test_samples"]),
                    "n_test_traces_used": int(row_metrics["n_test_traces_used"]),
                    "loaded_init_keys": int(row_metrics["loaded_init_keys"]),
                    "null_test_cross_entropy": float(null_metrics["null_test_cross_entropy"]),
                    "null_test_accuracy": float(null_metrics["null_test_accuracy"]),
                    "accuracy_ratio_vs_f6": float(acc_ratio_vs_f6) if np.isfinite(acc_ratio_vs_f6) else float("nan"),
                    "ce_retention_vs_f6": float(ce_retention_vs_f6) if np.isfinite(ce_retention_vs_f6) else float("nan"),
                    "passes_relative_f6_accuracy": bool(pass_acc),
                    "passes_relative_f6_ce_retention": bool(pass_ce),
                }
            )

        selected_configs.append(
            {
                "config_id": config_id,
                "config_label": label,
                "n_classes": int(num_classes),
                "split_summary": split_summary,
                "null_metrics": {
                    "test_cross_entropy": float(null_metrics["null_test_cross_entropy"]),
                    "test_accuracy": float(null_metrics["null_test_accuracy"]),
                },
                "subset_metrics": {
                    "F2": {
                        "test_cross_entropy": float(subset_metrics["F2"]["test_cross_entropy"]),
                        "test_accuracy": float(subset_metrics["F2"]["test_accuracy"]),
                        "best_val_cross_entropy": float(subset_metrics["F2"]["best_val_ce"]),
                        "loaded_init_keys": int(subset_metrics["F2"]["loaded_init_keys"]),
                    },
                    "Full-6D": {
                        "test_cross_entropy": float(subset_metrics["Full-6D"]["test_cross_entropy"]),
                        "test_accuracy": float(subset_metrics["Full-6D"]["test_accuracy"]),
                        "best_val_cross_entropy": float(subset_metrics["Full-6D"]["best_val_ce"]),
                        "loaded_init_keys": int(subset_metrics["Full-6D"]["loaded_init_keys"]),
                    },
                },
                "legacy_f6_init": legacy_info,
                "accuracy_ratio_vs_f6": float(acc_ratio_vs_f6) if np.isfinite(acc_ratio_vs_f6) else None,
                "ce_retention_vs_f6": float(ce_retention_vs_f6) if np.isfinite(ce_retention_vs_f6) else None,
                "passes_relative_f6_accuracy": bool(pass_acc),
                "passes_relative_f6_ce_retention": bool(pass_ce),
            }
        )

    if len(rows) == 0:
        raise RuntimeError("No valid D1 rows were produced. Check skipped_configs in manifest output.")

    subset_rank = {name: i for i, name in enumerate(SEQ_SUBSET_ORDER)}
    rows = sorted(rows, key=lambda r: (config_list.index(str(r["config_id"])), subset_rank.get(str(r["subset"]), 999)))

    valid_config_ids = []
    for cid in config_list:
        if any(str(r["config_id"]) == cid for r in rows):
            valid_config_ids.append(cid)

    rows_plot = [r for r in rows if str(r["config_id"]) in valid_config_ids]
    _plot_accuracy_comparison(rows=rows_plot, out_path=out_figure)

    _write_csv(
        out_per_config_csv,
        rows,
        fieldnames=[
            "config_id",
            "config_label",
            "subset",
            "test_cross_entropy",
            "test_accuracy",
            "best_val_cross_entropy",
            "best_epoch",
            "input_dim",
            "n_classes",
            "n_train_traces",
            "n_val_traces",
            "n_test_traces",
            "n_test_samples",
            "n_test_traces_used",
            "loaded_init_keys",
            "null_test_cross_entropy",
            "null_test_accuracy",
            "accuracy_ratio_vs_f6",
            "ce_retention_vs_f6",
            "passes_relative_f6_accuracy",
            "passes_relative_f6_ce_retention",
        ],
    )

    pass_acc_list = [bool(c.get("passes_relative_f6_accuracy", False)) for c in selected_configs]
    pass_ce_list = [bool(c.get("passes_relative_f6_ce_retention", False)) for c in selected_configs]
    n_valid_configs = int(len(selected_configs))
    n_pass_acc = int(sum(1 for v in pass_acc_list if v))
    n_pass_ce = int(sum(1 for v in pass_ce_list if v))
    pass_ratio_acc = float(n_pass_acc / n_valid_configs) if n_valid_configs > 0 else float("nan")
    pass_ratio_ce = float(n_pass_ce / n_valid_configs) if n_valid_configs > 0 else float("nan")

    manifest = {
        "schema_version": "figure-d1-sequence-bigru-v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "experimental_manifest": experimental_manifest,
            "run_manifest": run_manifest,
            "pair_manifest_csv": pair_manifest_csv,
            "throughput_db": throughput_db,
            "config_ids_requested": config_list,
        },
        "config_policy": {
            "source": "bic_configs",
            "bic_configs": BIC_CONFIGS,
            "valid_configs_in_plot": valid_config_ids,
        },
        "trace_filter_policy": {
            "scope": "train_val_test_indices",
            "require_recorded_timestamps": bool(require_recorded_timestamps),
            "min_valid_traces_per_config": int(max(1, min_valid_traces_per_config)),
            "drop_on_missing_recorded_timestamps": bool(require_recorded_timestamps),
        },
        "model_policy": {
            "model": "bidirectional_gru_classifier",
            "subsets": SEQ_SUBSET_ORDER,
            "hidden_dim": int(hidden_dim),
            "num_layers": int(max(1, num_layers)),
            "epochs": int(max(1, epochs)),
            "lr": float(lr),
            "patience": int(max(1, patience)),
            "seed": int(seed),
            "device": str(resolved_device),
            "legacy_f6_init_enabled": bool(use_legacy_f6_init),
            "legacy_f6_weights_dirs": resolved_legacy_dirs,
        },
        "selected_configs": selected_configs,
        "skipped_configs": skipped_configs,
        "per_config_metrics": rows,
        "pass_summary": {
            "criterion_accuracy": "Accuracy(F2) >= 0.95 * Accuracy(Full-6D) when Accuracy(Full-6D)>0",
            "criterion_ce_retention": "(CE_null-CE_F2)/(CE_null-CE_F6) >= 0.95 when denominator!=0",
            "n_valid_configs": int(n_valid_configs),
            "n_pass_accuracy": int(n_pass_acc),
            "n_pass_ce_retention": int(n_pass_ce),
            "pass_ratio_accuracy": float(pass_ratio_acc) if np.isfinite(pass_ratio_acc) else None,
            "pass_ratio_ce_retention": float(pass_ratio_ce) if np.isfinite(pass_ratio_ce) else None,
        },
        "outputs": {
            "figure": out_figure,
            "per_config_csv": out_per_config_csv,
            "manifest_json": out_json,
        },
    }

    _write_json(out_json, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    p = argparse.ArgumentParser(description="Figure D1 held-out BiGRU comparison (F2 vs Full-6D)")
    p.add_argument("--experimental-manifest", default=defaults["experimental_manifest"])
    p.add_argument("--run-manifest", default=defaults["run_manifest"])
    p.add_argument("--pair-manifest-csv", default=defaults["pair_manifest_csv"])
    p.add_argument("--throughput-db", default=defaults["throughput_db"])
    p.add_argument("--config-ids", default=None, help="Comma-separated config IDs. Defaults to methods BIC set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--require-recorded-timestamps",
        type=str,
        default="true",
        choices=["true", "false"],
        help="If true, traces without recorded request_timestamps are dropped.",
    )
    p.add_argument("--min-valid-traces-per-config", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--use-legacy-f6-init",
        type=str,
        default="true",
        choices=["true", "false"],
        help="If true, attempt to initialize Full-6D model from legacy 6D checkpoints in model/*weights.",
    )
    p.add_argument(
        "--legacy-f6-weights-dirs",
        type=str,
        default="model/new_weights,model/best_weights,model/gru_classifier_weights",
        help="Comma-separated directories searched for legacy Full-6D checkpoints.",
    )
    p.add_argument("--out-figure", default=defaults["out_figure"])
    p.add_argument("--out-per-config-csv", default=defaults["out_per_config_csv"])
    p.add_argument("--out-json", default=defaults["out_json"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    manifest = run_d1_conditional_entropy(
        experimental_manifest=str(args.experimental_manifest),
        run_manifest=str(args.run_manifest),
        pair_manifest_csv=str(args.pair_manifest_csv),
        throughput_db=str(args.throughput_db),
        config_ids=args.config_ids,
        seed=int(args.seed),
        require_recorded_timestamps=(str(args.require_recorded_timestamps).lower() == "true"),
        min_valid_traces_per_config=int(args.min_valid_traces_per_config),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        epochs=int(args.epochs),
        lr=float(args.lr),
        patience=int(args.patience),
        device=str(args.device),
        use_legacy_f6_init=(str(args.use_legacy_f6_init).lower() == "true"),
        legacy_f6_weights_dirs=_parse_csv_list(args.legacy_f6_weights_dirs),
        out_figure=str(args.out_figure),
        out_per_config_csv=str(args.out_per_config_csv),
        out_json=str(args.out_json),
    )

    print("[figure_d1_conditional_entropy] Summary:")
    print(f"  requested_configs: {len(manifest['inputs']['config_ids_requested'])}")
    print(f"  selected_configs: {len(manifest['selected_configs'])}")
    print(f"  skipped_configs: {len(manifest['skipped_configs'])}")
    n_legacy_init = sum(
        1
        for cfg in manifest["selected_configs"]
        if str(cfg.get("legacy_f6_init", {}).get("status", "")) == "initialized"
    )
    print(f"  legacy_f6_initialized_configs: {n_legacy_init}")
    print(f"  pass_ratio_accuracy: {manifest['pass_summary']['pass_ratio_accuracy']}")
    print(f"  pass_ratio_ce_retention: {manifest['pass_summary']['pass_ratio_ce_retention']}")
    print(f"  per_config_csv: {manifest['outputs']['per_config_csv']}")
    print(f"  figure: {manifest['outputs']['figure']}")
    print(f"  manifest: {manifest['outputs']['manifest_json']}")


if __name__ == "__main__":
    main()
