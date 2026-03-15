#!/usr/bin/env python3
"""
Generate side-by-side CDF plots comparing held-out measured vs sampled power traces.

This script reproduces/updates publication-style CDF figures like:
  - Original (measured) vs Sampled (synthetic)
  - One panel per config_id (e.g., A100 TP=1 and H100 TP=1)

Sampling is driven by trained GMM+BiGRU artifacts and can run in:
  - iid mode (default, uses state-conditional GMM sampling)
  - ar1_thresholded mode (hybrid AR(1) per-state sampling)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
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
import seaborn as sns
import torch

# Allow running via: python3 scripts/eval/*.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.classifiers.feature_utils import (  # noqa: E402
    compute_delta_active_requests,
    compute_inference_features,
    normalize_delta_active_requests,
)
from model.classifiers.gmm_bigru import (  # noqa: E402
    AR1_MIN_RUN_LENGTH,
    AR1_PHI_THRESHOLD,
    estimate_ar1_params,
    extract_norm_params,
    generate_gmm_bigru_trace_ar1_thresholded,
    predict_sorted_gmm_labels_from_params,
)
from model.classifiers.metrics import compute_power_metrics  # noqa: E402
from model.classifiers.model_loading import load_gru_classifier  # noqa: E402
from model.utils.io import safe_slug, write_json  # noqa: E402
from scripts.eval.azure_defaults import MODEL_NAME_MAP  # noqa: E402
from scripts.eval.pipeline_utils import (  # noqa: E402
    resolve_checkpoint_norm_gmm_paths as _shared_resolve_checkpoint_norm_gmm_paths,
    resolve_experimental_paths as _shared_resolve_experimental_paths,
    resolve_throughput as _shared_resolve_throughput,
)

DEFAULT_CONFIG_IDS = (
    "deepseek-r1-distill-70b_A100_tp4",
    "llama-3-8b_H100_tp1",
)
PREFERRED_GPT_OSS_120B_CONFIG_IDS = (
    "gpt-oss-120b_H100_tp8",
    "gpt-oss-120b_A100_tp8",
)

CONFIG_MODEL_SIZE_RE = re.compile(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$")
EPS = 1e-12


def _safe_std(value: float) -> float:
    out = float(value)
    if (not np.isfinite(out)) or out <= 0.0:
        return 1e-6
    return max(out, 1e-6)


def _extract_norm_value(norm: Mapping[str, float], *keys: str) -> float:
    for key in keys:
        if key in norm:
            return float(norm[key])
    raise KeyError(f"Missing required norm key; expected one of {keys}")


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / np.clip(denom, a_min=EPS, a_max=None)


def _tensor_to_numpy(t: torch.Tensor, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Convert torch tensor -> numpy with fallback for environments where torch.numpy bridge is unavailable.
    """
    cpu_t = t.detach().cpu()
    try:
        return np.asarray(cpu_t.numpy(), dtype=dtype)
    except Exception:
        return np.asarray(cpu_t.tolist(), dtype=dtype)


def _median_filter_states(states: np.ndarray, window: int) -> np.ndarray:
    z = np.asarray(states, dtype=np.int64).reshape(-1)
    n = int(z.size)
    if n == 0:
        return z.copy()

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


def build_rollout_features_from_requests(
    *,
    requests: Sequence[Mapping[str, float]],
    throughput: Mapping[str, float],
    norm: Mapping[str, float],
    T: Optional[int],
    dt: float,
    feature_set: str = "f2",
) -> Dict[str, np.ndarray]:
    feat = str(feature_set).strip().lower()
    if feat == "f3":
        raise ValueError("feature_set='f3' is no longer supported; use 'f2'.")
    if feat != "f2":
        raise ValueError(f"feature_set must be 'f2'; got {feature_set}")

    lambda_prefill = _extract_norm_value(
        throughput, "lambda_prefill", "prefill_rate_median_toks_per_s"
    )
    lambda_decode = _extract_norm_value(
        throughput, "lambda_decode", "decode_rate_median_toks_per_s"
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

    a_norm = np.asarray(base[:, 0], dtype=np.float32)
    t_arrive_norm = np.asarray(base[:, 1], dtype=np.float32)
    a_raw = (a_norm.astype(np.float64) * float(active_std)) + float(active_mean)
    delta_raw = compute_delta_active_requests(a_raw).astype(np.float64)
    delta_norm = normalize_delta_active_requests(
        delta_raw, mean=dA_mean, std=dA_std
    ).astype(np.float32)

    features = np.stack([a_norm, delta_norm], axis=-1).astype(np.float32)
    return {
        "features_norm": features,
        "A_raw": a_raw.astype(np.float64),
        "A_norm": a_norm.astype(np.float32),
        "delta_A_raw": delta_raw.astype(np.float64),
        "delta_A_norm": delta_norm.astype(np.float32),
        "t_arrive_norm": t_arrive_norm.astype(np.float32),
    }


def generate_gmm_bigru_trace(
    logits: np.ndarray | torch.Tensor,
    gmm_params: Mapping[str, object],
    seed: Optional[int] = None,
    decode_mode: str = "stochastic",
    median_filter_window: int = 1,
    clamp_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, np.ndarray]:
    if isinstance(logits, torch.Tensor):
        z = _tensor_to_numpy(logits, dtype=np.float64)
    else:
        z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if z.ndim != 2:
        raise ValueError(f"logits must have shape (T,K) or (1,T,K); got {z.shape}")

    means = np.asarray(gmm_params["means"], dtype=np.float64).reshape(-1)
    variances = np.asarray(gmm_params["variances"], dtype=np.float64).reshape(-1)
    k = int(means.size)
    if z.shape[1] != k:
        raise ValueError(f"logits K mismatch: got {z.shape[1]} but GMM has {k}")

    probs = _softmax_np(z)
    mode = str(decode_mode).strip().lower()
    if mode not in {"stochastic", "argmax"}:
        raise ValueError(
            f"decode_mode must be 'stochastic' or 'argmax'; got {decode_mode}"
        )

    rng = np.random.default_rng(seed)
    if mode == "argmax":
        states_raw = np.argmax(probs, axis=-1).astype(np.int64)
    else:
        states_raw = np.asarray(
            [rng.choice(k, p=probs_t) for probs_t in probs], dtype=np.int64
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


def _ensure_dir_for_file(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _write_csv(
    path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]
) -> None:
    _ensure_dir_for_file(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


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


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _parse_config_ids(config_ids: Optional[Sequence[str]]) -> List[str]:
    if not config_ids:
        return []
    out: List[str] = []
    for token in config_ids:
        if token is None:
            continue
        out.extend([x.strip() for x in str(token).split(",") if x.strip()])
    deduped: List[str] = []
    seen = set()
    for cid in out:
        if cid in seen:
            continue
        deduped.append(cid)
        seen.add(cid)
    return deduped


def _select_single_gpt_oss_120b_config_id(
    run_cfgs: Mapping[str, object],
) -> Optional[str]:
    for cid in PREFERRED_GPT_OSS_120B_CONFIG_IDS:
        row = run_cfgs.get(cid)
        if isinstance(row, dict) and str(row.get("status", "")) == "trained":
            return str(cid)

    fallback: List[str] = []
    for key, row in run_cfgs.items():
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")) != "trained":
            continue
        k = str(key).strip()
        if k.startswith("gpt-oss-120b_"):
            fallback.append(k)
    if len(fallback) == 0:
        return None
    fallback.sort()
    return fallback[0]


def _finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _synthesize_request_timestamps(
    payload: Dict[str, object], n: int
) -> Optional[List[float]]:
    if n <= 0:
        return []

    duration = _finite_float(payload.get("duration"))
    if duration is not None and duration > 0:
        step = float(duration) / float(max(n, 1))
        if step > 0:
            values = (np.arange(n, dtype=np.float64) + 0.5) * step + 1.0
            return [float(x) for x in values]

    request_rate = _finite_float(payload.get("request_rate"))
    poisson_rate = _finite_float(payload.get("poisson_rate"))
    rate = request_rate if request_rate is not None else poisson_rate
    if rate is not None and rate > 0:
        step = 1.0 / float(rate)
        values = (np.arange(n, dtype=np.float64) + 1.0) * step + 1.0
        return [float(x) for x in values]
    return None


def _build_requests_from_stage0_json(
    request_json_path: str,
    *,
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
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
        n = int(n_base)
        synth = _synthesize_request_timestamps(payload, n)
        if synth is None:
            raise ValueError("request json missing arrays: ['request_timestamps']")
        request_timestamps = synth
    if n <= 0:
        raise ValueError("request arrays are empty after alignment")

    arrivals = np.asarray(request_timestamps[:n], dtype=np.float64) - float(
        power_start_epoch_s
    )
    if arrivals.size > 0 and (
        float(np.min(arrivals)) < -float(dt)
        or float(np.max(arrivals)) > float(trace_duration_s) + float(dt)
    ):
        arrivals = arrivals - float(np.min(arrivals))

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
    if len(requests) == 0:
        raise ValueError("no valid requests after filtering")
    return requests


def _load_pair_manifest_map(pair_manifest_csv: str) -> Dict[str, str]:
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


_extract_norm_for_eval = extract_norm_params


_resolve_throughput = _shared_resolve_throughput
_resolve_checkpoint_norm_gmm_paths = _shared_resolve_checkpoint_norm_gmm_paths
_resolve_experimental_paths = _shared_resolve_experimental_paths


_load_model = load_gru_classifier


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    xs = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    ys = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if xs.size == 0 or ys.size == 0:
        return float("nan")
    values = np.concatenate([xs, ys])
    values.sort()
    cdf_x = np.searchsorted(xs, values, side="right") / float(xs.size)
    cdf_y = np.searchsorted(ys, values, side="right") / float(ys.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = int(x.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    y = np.arange(1, n + 1, dtype=np.float64) / float(n)
    return x, y


def _display_label(config_id: str) -> str:
    m = CONFIG_MODEL_SIZE_RE.match(str(config_id).strip())
    if m is None:
        return str(config_id)
    model_family, model_size, hardware, tp = m.groups()
    model_name = MODEL_NAME_MAP.get(model_family, model_family)
    return f"{model_name}-{int(model_size)}B {hardware} TP={int(tp)}"


def _build_default_paths() -> Dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "run_manifest": str(
            repo_root
            / "results"
            / "continuous_v1_gmm_bigru"
            / "k10_f2"
            / "run_manifest.json"
        ),
        "experimental_manifest": str(
            repo_root / "results" / "experimental_continuous_v1" / "manifest.json"
        ),
        "throughput_db": str(
            repo_root / "model" / "config" / "throughput_database.json"
        ),
        "pair_manifest_csv": str(
            repo_root / "results" / "stage0" / "pair_manifest.csv"
        ),
        "out_plot_dir": str(repo_root / "figures" / "trace_power_cdf_comparison"),
        "out_cdf_csv": str(
            repo_root
            / "results"
            / "eval_paper"
            / "trace_power_cdf_comparison_points.csv"
        ),
        "out_summary_csv": str(
            repo_root / "results" / "eval_paper" / "trace_power_cdf_comparison.csv"
        ),
        "out_json": str(
            repo_root / "results" / "eval_paper" / "trace_power_cdf_comparison.json"
        ),
    }


def _collect_config_cdf(
    *,
    config_id: str,
    run_cfg_row: Dict[str, object],
    run_manifest_base: str,
    experimental_payload: Dict[str, object],
    experimental_base: str,
    throughput_payload: Dict[str, object],
    pair_map: Mapping[str, str],
    seeds: Sequence[int],
    generation_mode: str,
    decode_mode: str,
    median_filter_window: int,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint_path, norm_path, gmm_path = _resolve_checkpoint_norm_gmm_paths(
        run_cfg_row, run_manifest_base
    )
    norm_payload = _load_json(norm_path)
    norm_cfg = _extract_norm_for_eval(norm_payload)
    gmm_payload = _load_json(gmm_path)
    gmm_cfg = load_gmm_params_json_dict(gmm_payload)
    throughput = _resolve_throughput(throughput_payload, config_id)

    k = int(run_cfg_row.get("k", gmm_cfg["k"]))
    feature_set = str(
        run_cfg_row.get("feature_set", norm_payload.get("feature_set", "f2"))
    ).lower()
    if feature_set == "f3":
        raise ValueError("feature_set='f3' is no longer supported; use 'f2'.")
    if feature_set != "f2":
        raise ValueError(f"invalid feature_set for '{config_id}': {feature_set}")
    input_dim = int(run_cfg_row.get("input_dim", 2))
    hidden_dim = int(run_cfg_row.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
    num_layers = int(run_cfg_row.get("num_layers", norm_payload.get("num_layers", 1)))
    if k != int(gmm_cfg["k"]):
        raise ValueError(
            f"k mismatch between run manifest ({k}) and gmm payload ({int(gmm_cfg['k'])})"
        )

    model = _load_model(
        checkpoint_path=checkpoint_path,
        k=k,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
    )

    dataset_path, split_path = _resolve_experimental_paths(
        experimental_payload,
        config_id=config_id,
        experimental_base=experimental_base,
    )
    split_payload = _load_json(split_path)
    test_indices = [int(x) for x in split_payload.get("test_indices", [])]
    train_indices = [int(x) for x in split_payload.get("train_indices", [])]
    if len(test_indices) == 0:
        raise ValueError(f"empty test split for {config_id}")

    with np.load(dataset_path, allow_pickle=True) as data:
        pair_key_arr = np.asarray(data["pair_key"], dtype=object)
        power_arr = np.asarray(data["power"], dtype=object)
        power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
        dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)

    if dt_arr.size == 0:
        raise ValueError("dataset dt missing")
    dt = float(dt_arr[0])
    if (not np.isfinite(dt)) or dt <= 0.0:
        raise ValueError(f"invalid dt in dataset: {dt}")

    n_total = int(min(len(pair_key_arr), len(power_arr), len(power_start_arr)))

    phi = np.zeros((int(k),), dtype=np.float64)
    sigma_innov = np.zeros((int(k),), dtype=np.float64)
    sigma_marginal = np.zeros((int(k),), dtype=np.float64)
    if generation_mode == "ar1_thresholded":
        training_power_traces: List[np.ndarray] = []
        training_labels_traces: List[np.ndarray] = []
        for idx in train_indices:
            if idx < 0 or idx >= n_total:
                continue
            p_train = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
            if p_train.size <= 0:
                continue
            labels_train = predict_sorted_gmm_labels_from_params(p_train, gmm_cfg)
            training_power_traces.append(p_train.astype(np.float64))
            training_labels_traces.append(labels_train.astype(np.int64))
        if len(training_power_traces) == 0:
            raise ValueError(
                f"no valid training traces for AR1 estimation: {config_id}"
            )
        phi, sigma_innov, sigma_marginal = estimate_ar1_params(
            gmm_params=gmm_cfg,
            training_power_traces=training_power_traces,
            training_labels_traces=training_labels_traces,
            K=int(k),
            min_run_length=AR1_MIN_RUN_LENGTH,
        )

    original_chunks: List[np.ndarray] = []
    sampled_chunks: List[np.ndarray] = []
    chosen_seed_rows: List[Dict[str, object]] = []
    skipped = 0

    for trace_idx in test_indices:
        if trace_idx < 0 or trace_idx >= n_total:
            skipped += 1
            continue
        power = np.asarray(power_arr[trace_idx], dtype=np.float64).reshape(-1)
        if power.size < 2:
            skipped += 1
            continue
        pair_key = str(pair_key_arr[trace_idx])
        json_path = pair_map.get(pair_key)
        if json_path is None:
            skipped += 1
            continue

        try:
            p0 = float(power[0])
            gt = power[1:].astype(np.float64)
            requests = _build_requests_from_stage0_json(
                json_path,
                power_start_epoch_s=float(power_start_arr[trace_idx]),
                trace_duration_s=float(power.size * dt),
                dt=float(dt),
            )
            feat = build_rollout_features_from_requests(
                requests=requests,
                throughput=throughput,
                norm=norm_cfg,
                T=int(gt.size),
                dt=float(dt),
                feature_set=feature_set,
            )
            features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
            if features_norm.ndim != 2 or features_norm.shape[1] != input_dim:
                skipped += 1
                continue
            with torch.no_grad():
                x = torch.tensor(
                    features_norm.tolist(), dtype=torch.float32, device=device
                ).unsqueeze(0)
                logits = _tensor_to_numpy(model(x)[0], dtype=np.float64)

            per_seed_rows: List[Dict[str, object]] = []
            preds_by_seed: Dict[int, np.ndarray] = {}
            for seed in seeds:
                if generation_mode == "iid":
                    gen = generate_gmm_bigru_trace(
                        logits=logits,
                        gmm_params=gmm_cfg,
                        seed=int(seed),
                        decode_mode=decode_mode,
                        median_filter_window=int(median_filter_window),
                        clamp_range=(norm_cfg["power_min"], norm_cfg["power_max"]),
                    )
                else:
                    gen = generate_gmm_bigru_trace_ar1_thresholded(
                        logits=logits,
                        gmm_params=gmm_cfg,
                        phi=phi,
                        sigma_innov=sigma_innov,
                        sigma_marginal=sigma_marginal,
                        p0=float(p0),
                        seed=int(seed),
                        decode_mode=decode_mode,
                        median_filter_window=int(median_filter_window),
                        phi_threshold=float(AR1_PHI_THRESHOLD),
                        clamp_range=(norm_cfg["power_min"], norm_cfg["power_max"]),
                    )
                pred = np.asarray(gen["power_w"], dtype=np.float64).reshape(-1)
                n = int(min(gt.size, pred.size))
                if n <= 0:
                    continue
                gt_n = gt[:n]
                pred_n = pred[:n]
                metrics = compute_power_metrics(gt_n, pred_n, dt=dt, acf_max_lag=50)
                row = {
                    "seed": int(seed),
                    "n": int(n),
                    "nrmse": float(metrics["nrmse"]),
                }
                per_seed_rows.append(row)
                preds_by_seed[int(seed)] = pred_n

            if len(per_seed_rows) == 0:
                skipped += 1
                continue

            nrmse_arr = np.asarray(
                [float(r["nrmse"]) for r in per_seed_rows], dtype=np.float64
            )
            median_nrmse = float(np.median(nrmse_arr))
            best_idx = int(np.argmin(np.abs(nrmse_arr - median_nrmse)))
            chosen_seed = int(per_seed_rows[best_idx]["seed"])
            pred_sel = np.asarray(preds_by_seed[chosen_seed], dtype=np.float64).reshape(
                -1
            )
            n_sel = int(min(gt.size, pred_sel.size))
            if n_sel <= 0:
                skipped += 1
                continue

            original_chunks.append(gt[:n_sel].astype(np.float64))
            sampled_chunks.append(pred_sel[:n_sel].astype(np.float64))
            chosen_seed_rows.append(
                {
                    "config_id": config_id,
                    "trace_idx": int(trace_idx),
                    "pair_key": pair_key,
                    "chosen_seed": int(chosen_seed),
                    "chosen_nrmse": float(per_seed_rows[best_idx]["nrmse"]),
                    "median_nrmse": float(median_nrmse),
                    "num_seed_candidates": int(len(per_seed_rows)),
                    "num_points": int(n_sel),
                }
            )
        except Exception:
            skipped += 1
            continue

    if len(original_chunks) == 0 or len(sampled_chunks) == 0:
        raise ValueError(f"no valid traces evaluated for {config_id}")

    original_all = np.concatenate(original_chunks, axis=0).astype(np.float64)
    sampled_all = np.concatenate(sampled_chunks, axis=0).astype(np.float64)
    n_all = int(min(original_all.size, sampled_all.size))
    if n_all <= 0:
        raise ValueError(f"no aligned points after aggregation for {config_id}")
    original_all = original_all[:n_all]
    sampled_all = sampled_all[:n_all]

    metrics = compute_power_metrics(original_all, sampled_all, dt=dt, acf_max_lag=50)
    sorted_orig, cdf_orig = _ecdf(original_all)
    sorted_samp, cdf_samp = _ecdf(sampled_all)
    ks_stat = _ks_statistic(original_all, sampled_all)

    return {
        "config_id": config_id,
        "display_label": _display_label(config_id),
        "dt": float(dt),
        "num_test_traces": int(len(test_indices)),
        "num_eval_traces": int(len(chosen_seed_rows)),
        "num_skipped_traces": int(skipped),
        "num_points": int(n_all),
        "generation_mode": str(generation_mode),
        "decode_mode": str(decode_mode),
        "median_filter_window": int(median_filter_window),
        "summary": {
            "ks_stat": float(ks_stat),
            "nrmse": float(metrics["nrmse"]),
            "acf_r2": float(metrics["acf_r2"]),
            "p95_error_pct": float(metrics["p95_error_pct"]),
            "p99_error_pct": float(metrics["p99_error_pct"]),
            "delta_energy_pct": float(metrics["delta_energy_pct"]),
            "original_mean_w": float(np.mean(original_all)),
            "sampled_mean_w": float(np.mean(sampled_all)),
        },
        "original_sorted": sorted_orig,
        "original_cdf": cdf_orig,
        "sampled_sorted": sorted_samp,
        "sampled_cdf": cdf_samp,
        "chosen_seed_rows": chosen_seed_rows,
    }


def _plot_cdfs(
    *,
    results: Sequence[Dict[str, object]],
    out_plot_dir: str,
) -> List[Dict[str, str]]:
    if int(len(results)) <= 0:
        raise ValueError("No results to plot")

    Path(out_plot_dir).mkdir(parents=True, exist_ok=True)

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plot_files: List[Dict[str, str]] = []
    for row in results:
        sns.set_style("whitegrid")
        sns.set_context("talk", font_scale=1.6)
        fig, ax = plt.subplots(figsize=(6, 6.5))
        x_o = np.asarray(row["original_sorted"], dtype=np.float64)
        y_o = np.asarray(row["original_cdf"], dtype=np.float64)
        x_s = np.asarray(row["sampled_sorted"], dtype=np.float64)
        y_s = np.asarray(row["sampled_cdf"], dtype=np.float64)

        ax.plot(x_o, y_o, label="Original", color="#1f77b4", linewidth=2.2)
        ax.plot(x_s, y_s, label="Sampled", color="#d99000", linewidth=2.2)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Active GPU Power (W)")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.35)
        ax.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=2, frameon=False)

        slug = safe_slug(str(row["config_id"]))
        out_pdf = str(Path(out_plot_dir) / f"{slug}_power_cdf.pdf")
        out_png = str(Path(out_plot_dir) / f"{slug}_power_cdf.png")
        fig.tight_layout()
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        plot_files.append(
            {
                "config_id": str(row["config_id"]),
                "plot_pdf": out_pdf,
                "plot_png": out_png,
            }
        )
    return plot_files


def generate_power_cdf_comparison(
    *,
    run_manifest: str,
    experimental_manifest: str,
    throughput_db: str,
    pair_manifest_csv: str,
    config_ids: Sequence[str],
    generation_mode: str,
    num_seeds: int,
    base_seed: int,
    decode_mode: str,
    median_filter_window: int,
    device: str,
    out_plot_dir: str,
    out_cdf_csv: str,
    out_summary_csv: str,
    out_json: str,
) -> Dict[str, object]:
    if int(num_seeds) <= 0:
        raise ValueError("num_seeds must be >= 1")
    if generation_mode not in {"iid", "ar1_thresholded"}:
        raise ValueError("generation_mode must be one of {'iid', 'ar1_thresholded'}")
    if decode_mode not in {"stochastic", "argmax"}:
        raise ValueError("decode_mode must be one of {'stochastic','argmax'}")

    run_payload = _load_json(run_manifest)
    run_cfgs = run_payload.get("configs", {})
    if not isinstance(run_cfgs, dict):
        raise ValueError("Invalid run manifest format")
    run_manifest_base = str(Path(run_manifest).resolve().parent)

    experimental_payload = _load_json(experimental_manifest)
    experimental_base = str(Path(experimental_manifest).resolve().parent)
    throughput_payload = _load_json(throughput_db)
    pair_map = _load_pair_manifest_map(pair_manifest_csv)

    seeds = [int(base_seed) + i for i in range(int(num_seeds))]
    resolved_device = _resolve_device(device)

    results: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    gpt_oss_single_plot_files: List[Dict[str, str]] = []
    gpt_oss_single_result: Optional[Dict[str, object]] = None
    gpt_oss_single_failure: Optional[Dict[str, str]] = None

    for config_id in config_ids:
        row = run_cfgs.get(config_id)
        if not isinstance(row, dict):
            failures.append(
                {"config_id": config_id, "reason": "config_not_in_run_manifest"}
            )
            continue
        if str(row.get("status", "")) != "trained":
            failures.append(
                {
                    "config_id": config_id,
                    "reason": f"config_status_{row.get('status', 'unknown')}",
                }
            )
            continue
        try:
            cfg_result = _collect_config_cdf(
                config_id=config_id,
                run_cfg_row=row,
                run_manifest_base=run_manifest_base,
                experimental_payload=experimental_payload,
                experimental_base=experimental_base,
                throughput_payload=throughput_payload,
                pair_map=pair_map,
                seeds=seeds,
                generation_mode=generation_mode,
                decode_mode=decode_mode,
                median_filter_window=int(median_filter_window),
                device=resolved_device,
            )
            results.append(cfg_result)
        except Exception as exc:
            failures.append(
                {"config_id": config_id, "reason": f"{type(exc).__name__}:{exc}"}
            )

    if len(results) == 0:
        raise ValueError(f"No configs successfully evaluated. Failures: {failures}")

    plot_files = _plot_cdfs(
        results=results,
        out_plot_dir=out_plot_dir,
    )

    # Also emit a single GPT-OSS 120B CDF if a trained config is available.
    gpt_oss_single_config_id = _select_single_gpt_oss_120b_config_id(run_cfgs)
    if gpt_oss_single_config_id is not None:
        existing = next(
            (
                r
                for r in results
                if str(r.get("config_id", "")) == gpt_oss_single_config_id
            ),
            None,
        )
        if existing is not None:
            gpt_oss_single_result = existing
        else:
            gpt_row = run_cfgs.get(gpt_oss_single_config_id)
            if isinstance(gpt_row, dict):
                try:
                    gpt_oss_single_result = _collect_config_cdf(
                        config_id=gpt_oss_single_config_id,
                        run_cfg_row=gpt_row,
                        run_manifest_base=run_manifest_base,
                        experimental_payload=experimental_payload,
                        experimental_base=experimental_base,
                        throughput_payload=throughput_payload,
                        pair_map=pair_map,
                        seeds=seeds,
                        generation_mode=generation_mode,
                        decode_mode=decode_mode,
                        median_filter_window=int(median_filter_window),
                        device=resolved_device,
                    )
                except Exception as exc:
                    gpt_oss_single_failure = {
                        "config_id": gpt_oss_single_config_id,
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
        if gpt_oss_single_result is not None:
            gpt_oss_single_plot_files = _plot_cdfs(
                results=[gpt_oss_single_result],
                out_plot_dir=out_plot_dir,
            )

    cdf_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    chosen_seed_rows: List[Dict[str, object]] = []

    for row in results:
        config_id = str(row["config_id"])
        for x, y in zip(
            np.asarray(row["original_sorted"]), np.asarray(row["original_cdf"])
        ):
            cdf_rows.append(
                {
                    "config_id": config_id,
                    "display_label": str(row["display_label"]),
                    "series": "original",
                    "power_w": float(x),
                    "cdf": float(y),
                }
            )
        for x, y in zip(
            np.asarray(row["sampled_sorted"]), np.asarray(row["sampled_cdf"])
        ):
            cdf_rows.append(
                {
                    "config_id": config_id,
                    "display_label": str(row["display_label"]),
                    "series": "sampled",
                    "power_w": float(x),
                    "cdf": float(y),
                }
            )

        s = dict(row["summary"])
        summary_rows.append(
            {
                "config_id": config_id,
                "display_label": str(row["display_label"]),
                "generation_mode": str(row["generation_mode"]),
                "decode_mode": str(row["decode_mode"]),
                "median_filter_window": int(row["median_filter_window"]),
                "dt": float(row["dt"]),
                "num_test_traces": int(row["num_test_traces"]),
                "num_eval_traces": int(row["num_eval_traces"]),
                "num_skipped_traces": int(row["num_skipped_traces"]),
                "num_points": int(row["num_points"]),
                "ks_stat": float(s["ks_stat"]),
                "nrmse": float(s["nrmse"]),
                "acf_r2": float(s["acf_r2"]),
                "p95_error_pct": float(s["p95_error_pct"]),
                "p99_error_pct": float(s["p99_error_pct"]),
                "delta_energy_pct": float(s["delta_energy_pct"]),
                "original_mean_w": float(s["original_mean_w"]),
                "sampled_mean_w": float(s["sampled_mean_w"]),
            }
        )
        chosen_seed_rows.extend([dict(x) for x in row["chosen_seed_rows"]])

    _write_csv(
        out_cdf_csv,
        cdf_rows,
        fieldnames=["config_id", "display_label", "series", "power_w", "cdf"],
    )
    _write_csv(
        out_summary_csv,
        summary_rows,
        fieldnames=[
            "config_id",
            "display_label",
            "generation_mode",
            "decode_mode",
            "median_filter_window",
            "dt",
            "num_test_traces",
            "num_eval_traces",
            "num_skipped_traces",
            "num_points",
            "ks_stat",
            "nrmse",
            "acf_r2",
            "p95_error_pct",
            "p99_error_pct",
            "delta_energy_pct",
            "original_mean_w",
            "sampled_mean_w",
        ],
    )

    payload = {
        "status": "ok",
        "inputs": {
            "run_manifest": str(run_manifest),
            "experimental_manifest": str(experimental_manifest),
            "throughput_db": str(throughput_db),
            "pair_manifest_csv": str(pair_manifest_csv),
            "config_ids": [str(x) for x in config_ids],
            "generation_mode": str(generation_mode),
            "num_seeds": int(num_seeds),
            "base_seed": int(base_seed),
            "decode_mode": str(decode_mode),
            "median_filter_window": int(median_filter_window),
            "device": str(resolved_device),
        },
        "artifacts": {
            "plot_dir": str(out_plot_dir),
            "plot_files": plot_files,
            "gpt_oss_120b_single_plot_files": gpt_oss_single_plot_files,
            "cdf_points_csv": str(out_cdf_csv),
            "summary_csv": str(out_summary_csv),
        },
        "summary": {
            "num_requested_configs": int(len(config_ids)),
            "num_successful_configs": int(len(results)),
            "num_failed_configs": int(len(failures)),
        },
        "config_summaries": summary_rows,
        "chosen_seed_rows": chosen_seed_rows,
        "failures": failures,
        "gpt_oss_120b_single": {
            "config_id": (
                str(gpt_oss_single_result["config_id"])
                if isinstance(gpt_oss_single_result, dict)
                else None
            ),
            "summary": (
                dict(gpt_oss_single_result["summary"])
                if isinstance(gpt_oss_single_result, dict)
                else None
            ),
            "failure": gpt_oss_single_failure,
        },
    }
    write_json(out_json, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = _build_default_paths()
    parser = argparse.ArgumentParser(
        description="Generate publication-style CDF comparison plots for measured vs sampled held-out power traces."
    )
    parser.add_argument("--run-manifest", default=defaults["run_manifest"])
    parser.add_argument(
        "--experimental-manifest", default=defaults["experimental_manifest"]
    )
    parser.add_argument("--throughput-db", default=defaults["throughput_db"])
    parser.add_argument("--pair-manifest-csv", default=defaults["pair_manifest_csv"])
    parser.add_argument(
        "--config-ids",
        nargs="*",
        default=list(DEFAULT_CONFIG_IDS),
        help="Config IDs to plot (space- or comma-separated).",
    )
    parser.add_argument(
        "--generation-mode", choices=["iid", "ar1_thresholded"], default="iid"
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--decode-mode", choices=["stochastic", "argmax"], default="stochastic"
    )
    parser.add_argument("--median-filter-window", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-plot-dir", default=defaults["out_plot_dir"])
    parser.add_argument("--out-cdf-csv", default=defaults["out_cdf_csv"])
    parser.add_argument("--out-summary-csv", default=defaults["out_summary_csv"])
    parser.add_argument("--out-json", default=defaults["out_json"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_ids = _parse_config_ids(args.config_ids)
    if len(config_ids) == 0:
        raise ValueError("No config_ids provided")

    run = generate_power_cdf_comparison(
        run_manifest=args.run_manifest,
        experimental_manifest=args.experimental_manifest,
        throughput_db=args.throughput_db,
        pair_manifest_csv=args.pair_manifest_csv,
        config_ids=config_ids,
        generation_mode=args.generation_mode,
        num_seeds=int(args.num_seeds),
        base_seed=int(args.base_seed),
        decode_mode=args.decode_mode,
        median_filter_window=int(args.median_filter_window),
        device=args.device,
        out_plot_dir=args.out_plot_dir,
        out_cdf_csv=args.out_cdf_csv,
        out_summary_csv=args.out_summary_csv,
        out_json=args.out_json,
    )

    print("[generate_power_cdf_comparison] Done")
    print(f"  requested_configs: {run['summary']['num_requested_configs']}")
    print(f"  successful_configs: {run['summary']['num_successful_configs']}")
    print(f"  failed_configs: {run['summary']['num_failed_configs']}")
    print(f"  plot_dir: {run['artifacts']['plot_dir']}")
    print(f"  summary_csv: {run['artifacts']['summary_csv']}")
    print(f"  cdf_points_csv: {run['artifacts']['cdf_points_csv']}")
    gpt_single = run.get("gpt_oss_120b_single", {})
    if isinstance(gpt_single, dict):
        cid = gpt_single.get("config_id")
        if isinstance(cid, str) and cid.strip():
            print(f"  gpt_oss_120b_single_config: {cid}")
        failure = gpt_single.get("failure")
        if isinstance(failure, dict):
            reason = str(failure.get("reason", "")).strip()
            if reason:
                print(f"  gpt_oss_120b_single_failure: {reason}")


if __name__ == "__main__":
    main()
