#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from model.classifiers.gmm_bigru import (
    build_rollout_features_from_requests,
    generate_gmm_bigru_trace,
    load_gmm_params_json_dict,
)
from model.classifiers.gru import GRUClassifier
from model.scripts.eval_gmm_bigru import (
    generate_gmm_bigru_trace_ar1_thresholded,
)

COLOR_DARK = "#2c3e50"
COLOR_RED = "#e74c3c"
COLOR_ORANGE = "#e67e22"
COLOR_GREEN = "#27ae60"
COLOR_PURPLE = "#8e44ad"
COLOR_LIGHT_GRAY = "#bdc3c7"

BIC_CONFIGS = {
    "bic_config1": {
        "config_id": "llama-3-8b_A100_tp1",
        "title": "Llama-3.1-8B / A100 / TP=1",
        "legend": "Llama A100 TP1",
    },
    "bic_config2": {
        "config_id": "llama-3-8b_H100_tp2",
        "title": "Llama-3.1-8B / H100 / TP=2",
        "legend": "Llama H100 TP2",
    },
    "bic_config3": {
        "config_id": "gpt-oss-120b_H100_tp8",
        "title": "GPT-OSS-120B / H100 / TP=8 (MoE Proxy)",
        "legend": "GPT-OSS H100 TP8",
    },
    "bic_config4": {
        "config_id": "deepseek-r1-distill-70b_H100_tp4",
        "title": "DeepSeek-R1-Distill-70B / H100 / TP=4 (Dense)",
        "legend": "DeepSeek H100 TP4",
    },
}

DENSE_CONFIG_ID = "llama-3-8b_H100_tp1"
DENSE_OVERLAY_RATES = {
    "sparse": 0.25,
    "medium": 1.0,
    "high": 4.0,
}
AT_OVERLAY_RATE = (
    0.25  # Changed from 0.0625 to show continuous load instead of discrete spikes
)
# None means use the full trace for A_t overlay.
AT_OVERLAY_WINDOW_SECONDS: Optional[float] = None

MOE_PROXY_CONFIG_ID = "gpt-oss-120b_H100_tp8"
MOE_PROXY_RATE = 1.0
DEEPSEEK_DENSE_CONFIG_ID = "deepseek-r1-distill-70b_H100_tp4"

GMM_STRUCTURE_CONFIGS = {
    "dense": {
        "config_id": "llama-3-8b_A100_tp1",
        "trace_idx": 16,
        "title": "Llama-3.1-8B / A100 / TP=1",
    },
    "moe": {
        "config_id": "gpt-oss-120b_H100_tp8",
        "trace_idx": 19,
        "title": "GPT-OSS-120B / H100 / TP=8 (MoE Proxy)",
    },
}

VALIDATION_PDFS = [
    "validation_deepseek-r1-distill_8b_h100_tp8_ttft.pdf",
    "validation_deepseek-r1-distill_8b_h100_tp8_decode.pdf",
]

K_VALUES = list(range(2, 21))
DEFAULT_SIM_SEED = 42


@dataclass
class TraceData:
    config_id: str
    trace_idx: int
    pair_key: str
    rate: str
    power_start_epoch_s: float
    dt: float
    power: np.ndarray


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_pdf(fig: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


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


def _to_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def load_sharegpt_requests_and_power(
    results_csv: str,
    power_csv: str,
    *,
    dt: float = 0.25,
    num_gpus: int = 8,
) -> Tuple[List[Dict[str, float]], np.ndarray, float]:
    """
    Load ShareGPT results and aggregate TP power traces.

    Args:
        results_csv: Path to ShareGPT results CSV with actual request timestamps
        power_csv: Path to nvidia-smi power CSV (contains all GPUs)
        dt: Sampling interval in seconds for binning power
        num_gpus: Number of GPUs in TP group (for aggregation)

    Returns:
        Tuple of (requests, aggregated_power, power_start_epoch_s)
        - requests: List of dicts with arrival_time, input_tokens, output_tokens
        - aggregated_power: Power trace binned at dt intervals (sum across GPUs)
        - power_start_epoch_s: Epoch timestamp of first power measurement
    """
    from datetime import datetime

    # Load requests from ShareGPT results CSV
    requests = []
    with open(results_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            req_time = float(row["Request Time"])
            input_tokens = float(row["Input Tokens"])
            output_tokens = float(row["Output Tokens"])
            requests.append(
                {
                    "arrival_time": req_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )

    if not requests:
        raise ValueError(f"No requests loaded from {results_csv}")

    # Load power CSV and aggregate across GPUs
    timestamps = []
    powers = []
    with open(power_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            ts_str = row[0].strip()
            power_str = row[1].strip().replace(" W", "")
            try:
                # Parse timestamp like "2025/04/25 03:47:11.710"
                dt_obj = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
                epoch_s = dt_obj.timestamp()
                power_w = float(power_str)
                timestamps.append(epoch_s)
                powers.append(power_w)
            except Exception:
                continue

    if not timestamps:
        raise ValueError(f"No power data loaded from {power_csv}")

    timestamps = np.array(timestamps)
    powers = np.array(powers)

    # Group by timestamp and sum across GPUs
    # nvidia-smi logs each GPU sequentially, so every num_gpus rows is one timestamp
    unique_ts = []
    aggregated_powers = []

    i = 0
    while i < len(timestamps):
        # Take up to num_gpus consecutive measurements
        end_idx = min(i + num_gpus, len(timestamps))
        group_ts = timestamps[i:end_idx]
        group_pw = powers[i:end_idx]

        # Use the median timestamp of the group
        ts = np.median(group_ts)
        # Sum power across GPUs
        total_power = np.sum(group_pw)

        unique_ts.append(ts)
        aggregated_powers.append(total_power)

        i = end_idx

    unique_ts = np.array(unique_ts)
    aggregated_powers = np.array(aggregated_powers)

    # Bin into dt intervals
    power_start_epoch_s = float(unique_ts[0])
    power_end_epoch_s = float(unique_ts[-1])
    duration_s = power_end_epoch_s - power_start_epoch_s
    num_bins = int(np.ceil(duration_s / dt)) + 1

    binned_power = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for ts, pw in zip(unique_ts, aggregated_powers):
        rel_time = ts - power_start_epoch_s
        bin_idx = int(np.floor(rel_time / dt))
        if 0 <= bin_idx < num_bins:
            binned_power[bin_idx] += pw
            bin_counts[bin_idx] += 1

    # Average power within each bin
    mask = bin_counts > 0
    binned_power[mask] /= bin_counts[mask]

    # Convert request times to relative times
    requests_relative = []
    for req in requests:
        rel_time = req["arrival_time"] - power_start_epoch_s
        requests_relative.append(
            {
                "arrival_time": rel_time,
                "input_tokens": req["input_tokens"],
                "output_tokens": req["output_tokens"],
            }
        )

    return requests_relative, binned_power, power_start_epoch_s


def normalize_rate(value: object) -> Optional[float]:
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def rate_matches(value: object, target: float, atol: float = 1e-9) -> bool:
    val = normalize_rate(value)
    if val is None:
        return False
    return abs(val - float(target)) <= float(atol)


def rate_is_one(value: object) -> bool:
    return rate_matches(value, 1.0)


def is_moe_config_id(config_id: str) -> bool:
    match = re.match(r"^(.+)-(\d+)b_(A100|H100)_tp(\d+)$", str(config_id).strip())
    if match is None:
        return False
    model_family = str(match.group(1)).lower()
    model_size = int(match.group(2))
    if "deepseek-r1-distill" in model_family:
        return False
    if "gpt-oss" in model_family and model_size >= 20:
        return True
    return False


def select_trace_by_best_median_nrmse(
    rows: Sequence[Mapping[str, str]],
    *,
    config_id: str,
    rate: float,
) -> int:
    candidates: List[Tuple[int, float]] = []
    for row in rows:
        if str(row.get("config_id", "")) != str(config_id):
            continue
        if str(row.get("status", "")) != "evaluated":
            continue
        if not rate_matches(row.get("rate", ""), rate):
            continue
        trace_idx = int(row.get("trace_idx", "-1"))
        nrmse = float(row.get("nrmse_median", "nan"))
        if trace_idx < 0 or not np.isfinite(nrmse):
            continue
        candidates.append((trace_idx, nrmse))
    if not candidates:
        raise ValueError(f"No per-trace candidates for config={config_id}, rate={rate}")
    candidates.sort(key=lambda x: (x[1], x[0]))
    return int(candidates[0][0])


def select_trace_by_best_median_nrmse_subset(
    rows: Sequence[Mapping[str, str]],
    *,
    config_id: str,
    rate: float,
    allowed_trace_indices: Sequence[int],
) -> int:
    allowed = {int(x) for x in allowed_trace_indices}
    if not allowed:
        raise ValueError("allowed_trace_indices is empty.")
    candidates: List[Tuple[int, float]] = []
    for row in rows:
        if str(row.get("config_id", "")) != str(config_id):
            continue
        if str(row.get("status", "")) != "evaluated":
            continue
        if not rate_matches(row.get("rate", ""), rate):
            continue
        trace_idx = int(row.get("trace_idx", "-1"))
        nrmse = float(row.get("nrmse_median", "nan"))
        if trace_idx < 0 or not np.isfinite(nrmse):
            continue
        if int(trace_idx) not in allowed:
            continue
        candidates.append((trace_idx, nrmse))
    if not candidates:
        raise ValueError(
            f"No per-trace candidates for config={config_id}, rate={rate}, "
            "within allowed_trace_indices"
        )
    candidates.sort(key=lambda x: (x[1], x[0]))
    return int(candidates[0][0])


def select_seed_nearest_median_nrmse(
    rows: Sequence[Mapping[str, str]],
    *,
    config_id: str,
    trace_idx: int,
) -> int:
    candidates: List[Tuple[int, float]] = []
    for row in rows:
        if str(row.get("config_id", "")) != str(config_id):
            continue
        if int(row.get("trace_idx", "-1")) != int(trace_idx):
            continue
        if str(row.get("status", "")) != "ok":
            continue
        seed = int(row.get("seed", "-1"))
        nrmse = float(row.get("nrmse", "nan"))
        if seed < 0 or not np.isfinite(nrmse):
            continue
        candidates.append((seed, nrmse))
    if not candidates:
        raise ValueError(
            f"No per-seed candidates for config={config_id}, trace_idx={trace_idx}"
        )
    values = np.asarray([v for _, v in candidates], dtype=np.float64)
    med = float(np.median(values))
    candidates.sort(key=lambda x: (abs(x[1] - med), x[0]))
    return int(candidates[0][0])


def _gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    sigma = max(float(std), 1e-12)
    z = (x - float(mean)) / sigma
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)


def bic_sweep(
    power_values: np.ndarray,
    *,
    k_values: Sequence[int] = K_VALUES,
    random_state: int = 42,
    n_init: int = 5,
    max_iter: int = 200,
    reg_covar: float = 1e-6,
) -> Dict[str, Any]:
    y = np.asarray(power_values, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size < 10:
        raise ValueError("Insufficient finite points for BIC sweep.")
    x = y.reshape(-1, 1)
    bics: List[float] = []
    for k in k_values:
        gmm = GaussianMixture(
            n_components=int(k),
            covariance_type="full",
            random_state=int(random_state),
            n_init=int(max(1, n_init)),
            max_iter=int(max(10, max_iter)),
            reg_covar=float(max(reg_covar, 1e-12)),
        )
        gmm.fit(x)
        bics.append(float(gmm.bic(x)))
    best_i = int(np.argmin(np.asarray(bics, dtype=np.float64)))
    best_k = int(list(k_values)[best_i])
    return {
        "k_values": [int(v) for v in k_values],
        "bic_values": [float(v) for v in bics],
        "best_k": int(best_k),
        "n_points": int(y.size),
    }


def normalize_bic_values(bic_values: Sequence[float]) -> List[float]:
    arr = np.asarray(bic_values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("bic_values is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bic_values contains non-finite values.")
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    denom = max(max_v - min_v, 1e-12)
    return ((arr - min_v) / denom).astype(np.float64).tolist()


def _fit_gmm(power_values: np.ndarray, k: int) -> GaussianMixture:
    y = np.asarray(power_values, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size < int(k):
        raise ValueError(f"Need at least k points for GMM fit; got n={y.size}, k={k}")
    x = y.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=int(k),
        covariance_type="full",
        random_state=42,
        n_init=10,
        max_iter=300,
        reg_covar=1e-6,
    )
    gmm.fit(x)
    return gmm


def detect_first_power_spike(
    power_trace: np.ndarray,
    *,
    dt: float,
    idle_threshold: float = 200.0,
    active_threshold: float = 300.0,
    window_bins: int = 3,
) -> int:
    """
    Detect the bin index where GPU power first rises above idle.

    Args:
        power_trace: Power values in watts
        dt: Sampling interval
        idle_threshold: Power level considered idle (default 200W)
        active_threshold: Power level considered active inference (default 300W)
        window_bins: Number of consecutive bins that must exceed threshold

    Returns:
        Bin index of first sustained power spike (0 if not found)
    """
    arr = np.asarray(power_trace, dtype=np.float64).reshape(-1)
    if arr.size < window_bins:
        return 0

    # Find first window where all bins exceed active_threshold
    for i in range(arr.size - window_bins + 1):
        window = arr[i : i + window_bins]
        if np.all(window >= active_threshold):
            return int(i)

    # Fallback: find first single bin above active_threshold
    above = np.where(arr >= active_threshold)[0]
    return int(above[0]) if above.size > 0 else 0


def detect_first_at_activation(a_t: np.ndarray) -> int:
    """
    Detect the bin index where A_t (active requests) first becomes non-zero.

    Args:
        a_t: Active request count array

    Returns:
        Bin index of first A_t > 0 (0 if A_t is always zero)
    """
    arr = np.asarray(a_t, dtype=np.float64).reshape(-1)
    nonzero = np.where(arr > 1e-9)[0]
    return int(nonzero[0]) if nonzero.size > 0 else 0


def select_transition_dense_window(
    a_t: np.ndarray, window_bins: int
) -> Tuple[int, int]:
    arr = np.asarray(a_t, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n <= 1:
        return 0, n
    w = int(max(2, window_bins))
    if w >= n:
        return 0, n

    transitions = (np.abs(np.diff(arr)) > 1e-9).astype(np.int64)
    pref = np.concatenate([[0], np.cumsum(transitions)])

    best_start = 0
    best_count = -1
    best_std = -1.0
    for start in range(0, n - w + 1):
        end = start + w
        count = int(pref[end - 1] - pref[start])
        std = float(np.std(arr[start:end]))
        if count > best_count or (count == best_count and std > best_std):
            best_count = count
            best_std = std
            best_start = start
    return int(best_start), int(best_start + w)


def select_trace_by_lowest_mean_power(
    power_by_trace: Sequence[np.ndarray], candidate_indices: Sequence[int]
) -> int:
    candidates = [int(i) for i in candidate_indices]
    if not candidates:
        raise ValueError("No candidate indices for lowest-mean-power selection.")
    best_idx = -1
    best_mean = float("inf")
    for idx in candidates:
        if idx < 0 or idx >= len(power_by_trace):
            continue
        arr = np.asarray(power_by_trace[idx], dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mean_w = float(np.mean(arr))
        if mean_w < best_mean:
            best_mean = mean_w
            best_idx = idx
    if best_idx < 0:
        raise ValueError("No valid finite power trace in candidate indices.")
    return int(best_idx)


def _synthesize_request_timestamps(
    payload: Dict[str, object], n: int
) -> Optional[List[float]]:
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


def build_requests_from_stage0_json(
    request_json_path: str,
    *,
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
    alignment_offset_s: float = 0.0,
    require_recorded_timestamps: bool = False,
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
            raise ValueError(
                f"request json missing arrays: ['request_timestamps'] "
                f"(synthetic fallback disabled): {request_json_path}"
            )
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

    # CRITICAL: Normalize to [0, trace_duration] FIRST, then apply offset.
    # If we apply offset before normalization, the normalization will undo it
    # by shifting arrivals back to start at 0.
    if arrivals.size > 0:
        arr_min = float(np.min(arrivals))
        arr_max = float(np.max(arrivals))
        out_of_window = arr_min < -float(dt) or arr_max > float(
            trace_duration_s
        ) + float(dt)
        if out_of_window:
            # First try correcting by an integer-hour timezone offset inferred
            # from arrival timestamps relative to power window.
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

    # Apply alignment offset AFTER normalization (so it doesn't get undone)
    arrivals = arrivals + float(alignment_offset_s)

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


def _extract_norm_for_eval(norm_payload: Dict[str, object]) -> Dict[str, float]:
    required = (
        "active_mean",
        "active_std",
        "t_arrive_log_mean",
        "t_arrive_log_std",
        "power_mean",
        "power_std",
        "power_min",
        "power_max",
        "delta_A_mean",
        "delta_A_std",
    )
    missing = [k for k in required if k not in norm_payload]
    if missing:
        raise ValueError(f"Norm params missing keys: {missing}")

    out = {k: float(norm_payload[k]) for k in required}
    if out["active_std"] <= 0.0:
        raise ValueError("active_std must be positive")
    if out["t_arrive_log_std"] <= 0.0:
        raise ValueError("t_arrive_log_std must be positive")
    if out["power_std"] <= 0.0:
        raise ValueError("power_std must be positive")
    if out["delta_A_std"] <= 0.0:
        raise ValueError("delta_A_std must be positive")
    return out


def _resolve_throughput_entry(
    throughput_payload: Dict[str, object], config_id: str
) -> Dict[str, float]:
    cfgs = throughput_payload.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("Invalid throughput DB format")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"Throughput row missing for {config_id}")
    prefill = _to_float(row.get("prefill_rate_median_toks_per_s"))
    decode = _to_float(row.get("decode_rate_median_toks_per_s"))
    if prefill is None or prefill <= 0:
        raise ValueError(f"Invalid prefill throughput for {config_id}")
    if decode is None or decode <= 0:
        raise ValueError(f"Invalid decode throughput for {config_id}")
    return {"lambda_prefill": float(prefill), "lambda_decode": float(decode)}


def _resolve_device(device: str | None) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def _plot_bic_curve(
    path: Path,
    title: str,
    k_values: Sequence[int],
    bic_values: Sequence[float],
    best_k: int,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(
        list(k_values),
        list(bic_values),
        "o-",
        markersize=3,
        linewidth=1.25,
        color=COLOR_DARK,
    )
    y_min = float(np.min(np.asarray(bic_values, dtype=np.float64)))
    ax.axvline(
        float(best_k), linestyle="--", linewidth=0.75, color=COLOR_RED, alpha=0.8
    )
    ax.annotate(
        f"K*={int(best_k)}",
        xy=(float(best_k), y_min),
        xytext=(float(best_k) + 1.2, y_min),
        color=COLOR_RED,
    )
    ax.set_xlabel("Number of components K")
    ax.set_ylabel("BIC")
    ax.set_xlim(float(min(k_values)), float(max(k_values)))
    save_pdf(fig, path)


def _plot_bic_normalized_overlay(
    path: Path,
    *,
    k_values: Sequence[int],
    series: Sequence[Mapping[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    palette = [COLOR_DARK, COLOR_RED, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE]
    for i, row in enumerate(series):
        ax.plot(
            list(k_values),
            list(row["bic_norm_values"]),
            "o-",
            markersize=2.8,
            linewidth=1.5,
            color=palette[i % len(palette)],
            label=str(row["legend"]),
        )
    ax.axvline(
        10.0,
        linestyle="--",
        linewidth=1.0,
        color=COLOR_PURPLE,
        alpha=0.8,
    )
    ax.annotate(
        "K=10 plateau",
        xy=(10.0, 0.95),
        xytext=(10.0 + 0.25, 0.95),
        color=COLOR_PURPLE,
    )
    ax.set_xlabel("Number of components K")
    ax.set_ylabel("Normalized BIC")
    ax.set_xlim(float(min(k_values)), float(max(k_values)))
    ax.legend(
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    save_pdf(fig, path)


def _plot_at_overlay(
    path: Path,
    *,
    time_s: np.ndarray,
    power_w: np.ndarray,
    a_t: np.ndarray,
) -> None:
    fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
    ax1.plot(time_s, power_w, color=COLOR_DARK, linewidth=1.5, label="GPU Power")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("GPU Power (W)", color=COLOR_DARK)
    ax1.set_xlim(0, 100.0)

    ax2 = ax1.twinx()
    ax2.step(
        time_s,
        a_t,
        where="post",
        color=COLOR_ORANGE,
        linewidth=1.0,
        alpha=0.85,
        label="$A_t$",
    )
    ax2.spines["right"].set_color(COLOR_ORANGE)
    ax2.set_ylabel("Active Requests ($A_t$)", color=COLOR_ORANGE)
    ax2.tick_params(axis="y", colors=COLOR_ORANGE)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="best")
    # ax2.legend(loc="best")
    fig.tight_layout()
    save_pdf(fig, path)


def _plot_sim_overlay(
    path: Path, *, time_s: np.ndarray, measured: np.ndarray, simulated: np.ndarray
) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(time_s, measured, color=COLOR_DARK, label="Measured")
    ax.plot(time_s, simulated, color=COLOR_RED, alpha=0.8, label="Simulated")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GPU Power (W)")
    ax.set_ylim(0.0, 1250.0)
    ax.grid(True, alpha=0.25)
    if len(time_s) > 1:
        ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax.legend(loc="best")
    save_pdf(fig, path)


def _plot_gmm_structure(
    path: Path,
    *,
    title: str,
    power_values: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    y = np.asarray(power_values, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    ax.hist(y, bins=100, density=True, color=COLOR_LIGHT_GRAY, alpha=0.5)
    x = np.linspace(float(np.min(y)), float(np.max(y)), 500)
    k = int(len(weights))
    cmap = plt.cm.viridis
    for i in range(k):
        pdf = float(weights[i]) * _gaussian_pdf(x, float(means[i]), float(stds[i]))
        denom = max(k - 1, 1)
        ax.plot(x, pdf, color=cmap(float(i) / float(denom)), linewidth=1.0)
    ax.set_xlabel("GPU Power (W)")
    ax.set_ylabel("Density")
    save_pdf(fig, path)


def _load_model_from_artifacts(
    *,
    checkpoint_path: str,
    input_dim: int,
    k: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
) -> GRUClassifier:
    model = GRUClassifier(
        Dx=int(input_dim),
        K=int(k),
        H=int(hidden_dim),
        num_layers=int(max(1, num_layers)),
    ).to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
        and isinstance(state["model_state_dict"], dict)
    ):
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


class MethodsFigureGenerator:
    def __init__(
        self,
        *,
        out_dir: str,
        dry_run: bool,
        device: str = "auto",
        experimental_manifest_path: str = "results/experimental_continuous_v1/manifest.json",
        pair_manifest_csv_path: str = "results/stage0/pair_manifest.csv",
        run_manifest_path: str = "results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
        per_trace_csv_path: str = "results/continuous_v1_gmm_bigru/k10_f2/eval_metrics/per_trace_metrics.csv",
        per_seed_csv_path: str = "results/continuous_v1_gmm_bigru/k10_f2/eval_metrics/per_seed_metrics.csv",
        per_seed_ar1_csv_path: str = "results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/eval_metrics/per_seed_metrics.csv",
        ar1_params_dir: str = "results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/ar1_params",
        throughput_db_path: str = "model/config/throughput_database.json",
        validation_source_dir: str = "model/tests/validation_results",
        dense_config_id: str = DENSE_CONFIG_ID,
        deepseek_dense_config_id: str = DEEPSEEK_DENSE_CONFIG_ID,
        moe_proxy_config_id: str = MOE_PROXY_CONFIG_ID,
        moe_proxy_rate: float = MOE_PROXY_RATE,
        figure_mode: str = "all",
    ):
        self.out_dir = Path(out_dir)
        self.dry_run = bool(dry_run)
        self.device = _resolve_device(device)
        self.figure_mode = str(figure_mode).strip().lower()
        if self.figure_mode not in {"all", "simulated-moe"}:
            raise ValueError(
                f"Unsupported figure_mode={figure_mode}. Expected one of ['all', 'simulated-moe']."
            )

        self.experimental_manifest_path = Path(experimental_manifest_path)
        self.pair_manifest_csv_path = Path(pair_manifest_csv_path)
        self.run_manifest_path = Path(run_manifest_path)
        self.per_trace_csv_path = Path(per_trace_csv_path)
        self.per_seed_csv_path = Path(per_seed_csv_path)
        self.per_seed_ar1_csv_path = Path(per_seed_ar1_csv_path)
        self.ar1_params_dir = Path(ar1_params_dir)
        self.throughput_db_path = Path(throughput_db_path)
        self.validation_source_dir = Path(validation_source_dir)

        self.dense_config_id = str(dense_config_id).strip()
        self.deepseek_dense_config_id = str(deepseek_dense_config_id).strip()
        self.moe_proxy_config_id = str(moe_proxy_config_id).strip()
        self.moe_proxy_rate = float(moe_proxy_rate)

        self.experimental_manifest: Dict[str, Any] = {}
        self.run_manifest: Dict[str, Any] = {}
        self.throughput_payload: Dict[str, Any] = {}
        self.pair_map: Dict[str, str] = {}
        self.per_trace_rows: List[Dict[str, str]] = []
        self.per_seed_rows: List[Dict[str, str]] = []
        self.per_seed_ar1_rows: List[Dict[str, str]] = []

        self._dataset_cache: Dict[str, Dict[str, Any]] = {}
        self._model_cache: Dict[str, Dict[str, Any]] = {}
        self._pair_has_recorded_timestamps_cache: Dict[str, bool] = {}

        self._reload_inputs()

    def _reload_inputs(self) -> None:
        self.experimental_manifest = _load_json(self.experimental_manifest_path)
        self.run_manifest = _load_json(self.run_manifest_path)
        self.throughput_payload = _load_json(self.throughput_db_path)
        self.pair_map = _load_pair_manifest_map(str(self.pair_manifest_csv_path))
        self.per_trace_rows = _read_csv_rows(self.per_trace_csv_path)
        self.per_seed_rows = _read_csv_rows(self.per_seed_csv_path)
        self.per_seed_ar1_rows = _read_csv_rows(self.per_seed_ar1_csv_path)

        self._dataset_cache.clear()
        self._model_cache.clear()
        self._pair_has_recorded_timestamps_cache.clear()

    def _try_autoswitch_gptoss_a100_bundle(self) -> bool:
        candidates = {
            "experimental_manifest_path": Path(
                "results/experimental_continuous_v1_gptoss_a100/manifest.json"
            ),
            "pair_manifest_csv_path": Path(
                "results/stage0_sharegpt_gptoss_a100/pair_manifest.csv"
            ),
            "run_manifest_path": Path(
                "results/continuous_v1_gmm_bigru_gptoss_a100/kauto_max20_f2/run_manifest.json"
            ),
            "per_trace_csv_path": Path(
                "results/continuous_v1_gmm_bigru_gptoss_a100/kauto_max20_f2/eval_metrics/per_trace_metrics.csv"
            ),
            "per_seed_csv_path": Path(
                "results/continuous_v1_gmm_bigru_gptoss_a100/kauto_max20_f2/eval_metrics/per_seed_metrics.csv"
            ),
            "per_seed_ar1_csv_path": Path(
                "results/continuous_v1_gmm_bigru_gptoss_a100/kauto_max20_f2/eval_metrics/per_seed_metrics.csv"
            ),
            "ar1_params_dir": Path(
                "results/continuous_v1_gmm_bigru_gptoss_a100/kauto_max20_f2/ar1_params"
            ),
            "throughput_db_path": Path(
                "results/stage0_sharegpt_gptoss_a100/throughput_database.json"
            ),
        }
        if not all(path.exists() for path in candidates.values()):
            return False

        for attr, path in candidates.items():
            setattr(self, attr, path)
        self._reload_inputs()
        return True

    def _config_dataset_entry(self, config_id: str) -> Dict[str, Any]:
        configs = self.experimental_manifest.get("configs", {})
        if not isinstance(configs, dict) or config_id not in configs:
            raise ValueError(f"Config {config_id} not in experimental manifest.")
        row = configs[config_id]
        if not isinstance(row, dict):
            raise ValueError(f"Invalid config row in manifest for {config_id}")
        return row

    def _load_dataset(self, config_id: str) -> Dict[str, Any]:
        if config_id in self._dataset_cache:
            return self._dataset_cache[config_id]
        row = self._config_dataset_entry(config_id)
        dataset_path = str(row.get("dataset_npz", ""))
        resolved = _resolve_existing_path(
            dataset_path, str(self.experimental_manifest_path.parent)
        )
        if resolved is None:
            raise ValueError(f"Dataset path missing for {config_id}: {dataset_path}")
        with np.load(resolved, allow_pickle=True) as data:
            pair_key = np.asarray(data["pair_key"], dtype=object)
            power = np.asarray(data["power"], dtype=object)
            rate = (
                np.asarray(data["rate"], dtype=object)
                if "rate" in data
                else np.asarray([], dtype=object)
            )
            power_start = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
            dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)
        if dt_arr.size == 0:
            raise ValueError(f"Dataset dt missing for {config_id}")
        dt = float(dt_arr[0])
        payload = {
            "dataset_path": resolved,
            "pair_key": pair_key,
            "power": power,
            "rate": rate,
            "power_start_epoch_s": power_start,
            "dt": dt,
        }
        self._dataset_cache[config_id] = payload
        return payload

    def get_trace(self, config_id: str, trace_idx: int) -> TraceData:
        data = self._load_dataset(config_id)
        idx = int(trace_idx)
        n = int(
            min(
                len(data["pair_key"]),
                len(data["power"]),
                len(data["power_start_epoch_s"]),
            )
        )
        if idx < 0 or idx >= n:
            raise ValueError(f"Trace index {idx} out of range for {config_id} (n={n}).")
        rate_arr = data["rate"]
        rate_val = str(rate_arr[idx]) if idx < len(rate_arr) else ""
        return TraceData(
            config_id=str(config_id),
            trace_idx=idx,
            pair_key=str(data["pair_key"][idx]),
            rate=rate_val,
            power_start_epoch_s=float(data["power_start_epoch_s"][idx]),
            dt=float(data["dt"]),
            power=np.asarray(data["power"][idx], dtype=np.float64).reshape(-1),
        )

    def _collect_power_for_rate(self, config_id: str, rate: float) -> np.ndarray:
        data = self._load_dataset(config_id)
        all_power = []
        for i, p in enumerate(data["power"]):
            rate_arr = data["rate"]
            r = str(rate_arr[i]) if i < len(rate_arr) else ""
            if not rate_matches(r, rate):
                continue
            arr = np.asarray(p, dtype=np.float64).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                all_power.append(arr)
        if not all_power:
            raise ValueError(f"No traces found for config={config_id}, rate={rate}")
        return np.concatenate(all_power, axis=0)

    def _candidate_trace_indices_for_rate(
        self, config_id: str, rate: float
    ) -> List[int]:
        data = self._load_dataset(config_id)
        out: List[int] = []
        for i in range(len(data["power"])):
            rate_arr = data["rate"]
            r = str(rate_arr[i]) if i < len(rate_arr) else ""
            if rate_matches(r, rate):
                out.append(int(i))
        return out

    def _pair_has_recorded_request_timestamps(self, pair_key: str) -> bool:
        cached = self._pair_has_recorded_timestamps_cache.get(pair_key)
        if cached is not None:
            return bool(cached)
        json_path = self.pair_map.get(pair_key)
        if json_path is None:
            # Some datasets can include traces whose pair keys are absent from the
            # active pair manifest bundle. Treat those as not-recorded so callers
            # can continue filtering to valid candidates.
            self._pair_has_recorded_timestamps_cache[pair_key] = False
            return False
        payload = _load_json(json_path)
        raw = payload.get("request_timestamps")
        has_recorded = bool(isinstance(raw, list) and len(raw) > 0)
        self._pair_has_recorded_timestamps_cache[pair_key] = has_recorded
        return has_recorded

    def _candidate_trace_indices_for_rate_with_recorded_timestamps(
        self, config_id: str, rate: float
    ) -> List[int]:
        out: List[int] = []
        for idx in self._candidate_trace_indices_for_rate(config_id, rate):
            tr = self.get_trace(config_id, int(idx))
            if self._pair_has_recorded_request_timestamps(tr.pair_key):
                out.append(int(idx))
        return out

    def _select_trace_best_median_nrmse_recorded(
        self, config_id: str, rate: float
    ) -> int:
        candidates = self._candidate_trace_indices_for_rate_with_recorded_timestamps(
            config_id, float(rate)
        )
        if not candidates:
            raise ValueError(
                f"No traces with recorded request_timestamps for "
                f"config={config_id}, rate={rate}"
            )
        return int(
            select_trace_by_best_median_nrmse_subset(
                self.per_trace_rows,
                config_id=config_id,
                rate=float(rate),
                allowed_trace_indices=candidates,
            )
        )

    def _select_dense_trace_for_rate(
        self, rate: float, *, prefer_lowest_mean_power: bool
    ) -> Dict[str, Any]:
        candidates_recorded = (
            self._candidate_trace_indices_for_rate_with_recorded_timestamps(
                self.dense_config_id, float(rate)
            )
        )
        if not candidates_recorded:
            raise ValueError(
                f"No traces with recorded request_timestamps for "
                f"config={self.dense_config_id}, rate={rate}. "
                "Dense overlays require real arrivals."
            )

        if bool(prefer_lowest_mean_power):
            trace_idx = int(
                select_trace_by_lowest_mean_power(
                    self._load_dataset(self.dense_config_id)["power"],
                    candidates_recorded,
                )
            )
            selection_method = "lowest_mean_power_per_rate_recorded_timestamps_only"
        else:
            try:
                trace_idx = int(
                    self._select_trace_best_median_nrmse_recorded(
                        self.dense_config_id, float(rate)
                    )
                )
                selection_method = "best_median_nrmse_per_rate_recorded_timestamps_only"
            except ValueError:
                trace_idx = int(
                    select_trace_by_lowest_mean_power(
                        self._load_dataset(self.dense_config_id)["power"],
                        candidates_recorded,
                    )
                )
                selection_method = "fallback_lowest_mean_power_recorded_timestamps_only_no_per_trace_metric"

        return {
            "trace_idx": int(trace_idx),
            "selection_method": selection_method,
            "recorded_candidate_trace_indices": [int(x) for x in candidates_recorded],
        }

    def _resolve_run_artifacts(self, config_id: str) -> Dict[str, Any]:
        if config_id in self._model_cache:
            return self._model_cache[config_id]

        run_cfgs = self.run_manifest.get("configs", {})
        if not isinstance(run_cfgs, dict):
            raise ValueError("Invalid run manifest config section.")
        row = run_cfgs.get(config_id)
        if not isinstance(row, dict):
            raise ValueError(f"Config {config_id} not found in run manifest.")
        if str(row.get("status", "")) != "trained":
            raise ValueError(f"Config {config_id} status is not trained.")

        base = str(self.run_manifest_path.parent)
        ckpt = _resolve_existing_path(str(row.get("checkpoint_path", "")), base)
        norm_path = _resolve_existing_path(str(row.get("norm_params_path", "")), base)
        gmm_path = _resolve_existing_path(str(row.get("gmm_params_path", "")), base)
        if ckpt is None or norm_path is None or gmm_path is None:
            raise ValueError(f"Missing model artifacts for {config_id}")

        norm_payload = _load_json(norm_path)
        norm_cfg = _extract_norm_for_eval(norm_payload)
        gmm_payload = _load_json(gmm_path)
        gmm_cfg = load_gmm_params_json_dict(gmm_payload)

        k = int(row.get("k", gmm_cfg["k"]))
        if k != int(gmm_cfg["k"]):
            raise ValueError(
                f"k mismatch for {config_id}: row={k}, gmm={int(gmm_cfg['k'])}"
            )
        feature_set = str(
            row.get("feature_set", norm_payload.get("feature_set", "f2"))
        ).lower()
        if feature_set not in {"f2", "f3"}:
            raise ValueError(f"Invalid feature_set for {config_id}: {feature_set}")
        input_dim = int(row.get("input_dim", 2 if feature_set == "f2" else 3))
        hidden_dim = int(row.get("hidden_dim", norm_payload.get("hidden_dim", 64)))
        num_layers = int(row.get("num_layers", norm_payload.get("num_layers", 1)))

        model = _load_model_from_artifacts(
            checkpoint_path=ckpt,
            input_dim=input_dim,
            k=k,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=self.device,
        )
        throughput = _resolve_throughput_entry(self.throughput_payload, config_id)

        payload = {
            "checkpoint_path": ckpt,
            "norm_path": norm_path,
            "gmm_path": gmm_path,
            "norm_cfg": norm_cfg,
            "gmm_cfg": gmm_cfg,
            "feature_set": feature_set,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "k": k,
            "model": model,
            "throughput": throughput,
        }
        self._model_cache[config_id] = payload
        return payload

    def _load_requests_for_trace(
        self,
        tr: TraceData,
        alignment_offset_s: float = 0.0,
        require_recorded_timestamps: bool = False,
    ) -> List[Dict[str, float]]:
        json_path = self.pair_map.get(tr.pair_key)
        if json_path is None:
            raise ValueError(f"Pair key not found in pair manifest: {tr.pair_key}")
        return build_requests_from_stage0_json(
            json_path,
            power_start_epoch_s=float(tr.power_start_epoch_s),
            trace_duration_s=float(max(0, len(tr.power) - 1) * tr.dt),
            dt=float(tr.dt),
            alignment_offset_s=float(alignment_offset_s),
            require_recorded_timestamps=bool(require_recorded_timestamps),
        )

    def _compute_logits_for_trace(
        self, tr: TraceData, require_recorded_timestamps: bool = False
    ) -> Dict[str, Any]:
        art = self._resolve_run_artifacts(tr.config_id)
        requests = self._load_requests_for_trace(
            tr, require_recorded_timestamps=bool(require_recorded_timestamps)
        )
        t_horizon = int(max(0, len(tr.power) - 1))
        feat = build_rollout_features_from_requests(
            requests=requests,
            throughput=art["throughput"],
            norm=art["norm_cfg"],
            T=t_horizon,
            dt=float(tr.dt),
            feature_set=str(art["feature_set"]),
        )
        features_norm = np.asarray(feat["features_norm"], dtype=np.float32)
        if features_norm.ndim != 2 or features_norm.shape[1] != int(art["input_dim"]):
            raise ValueError(
                f"Feature shape mismatch for {tr.config_id}: {features_norm.shape} vs input_dim={art['input_dim']}"
            )
        with torch.no_grad():
            x = (
                torch.from_numpy(features_norm)
                .to(device=self.device, dtype=torch.float32)
                .unsqueeze(0)
            )
            logits = art["model"](x)[0].detach().cpu().numpy()
        return {
            "logits": np.asarray(logits, dtype=np.float64),
            "features": feat,
            "requests_count": len(requests),
            "norm_cfg": art["norm_cfg"],
            "gmm_cfg": art["gmm_cfg"],
        }

    def _simulate_dense_trace(self, tr: TraceData, seed: int) -> Dict[str, Any]:
        payload = self._compute_logits_for_trace(tr, require_recorded_timestamps=True)
        generated = generate_gmm_bigru_trace(
            logits=payload["logits"],
            gmm_params=payload["gmm_cfg"],
            seed=int(seed),
            decode_mode="stochastic",
            median_filter_window=1,
            clamp_range=(
                payload["norm_cfg"]["power_min"],
                payload["norm_cfg"]["power_max"],
            ),
        )
        pred = np.asarray(generated["power_w"], dtype=np.float64).reshape(-1)
        gt = np.asarray(tr.power[1:], dtype=np.float64).reshape(-1)
        expected_n = int(len(gt))
        if expected_n <= 0:
            raise ValueError(
                f"No ground-truth points for dense simulation: {tr.config_id} trace={tr.trace_idx}"
            )
        if int(len(pred)) != expected_n:
            raise ValueError(
                f"Length mismatch for dense simulation: {tr.config_id} trace={tr.trace_idx} "
                f"pred_len={len(pred)} gt_len={expected_n}. Check feature horizon/request alignment."
            )
        return {
            "time_s": np.arange(expected_n, dtype=np.float64) * float(tr.dt),
            "measured_w": gt,
            "simulated_w": pred,
            "requests_count": int(payload["requests_count"]),
        }

    def _simulate_moe_proxy_trace(self, tr: TraceData, seed: int) -> Dict[str, Any]:
        payload = self._compute_logits_for_trace(tr, require_recorded_timestamps=True)
        slug = _safe_slug(tr.config_id)
        ar1_path = self.ar1_params_dir / f"{slug}_ar1_params.json"
        if not ar1_path.exists():
            raise ValueError(f"AR(1) params missing: {ar1_path}")
        ar1_payload = _load_json(ar1_path)
        phi = np.asarray(ar1_payload.get("phi", []), dtype=np.float64).reshape(-1)
        sigma_innov = np.asarray(
            ar1_payload.get("sigma_innov", []), dtype=np.float64
        ).reshape(-1)
        sigma_marginal = np.asarray(
            ar1_payload.get("sigma_marginal", []), dtype=np.float64
        ).reshape(-1)
        if phi.size != int(payload["gmm_cfg"]["k"]):
            raise ValueError(f"AR(1) phi size mismatch for {tr.config_id}")
        generated = generate_gmm_bigru_trace_ar1_thresholded(
            logits=payload["logits"],
            gmm_params=payload["gmm_cfg"],
            phi=phi,
            sigma_innov=sigma_innov,
            sigma_marginal=sigma_marginal,
            p0=float(tr.power[0]),
            seed=int(seed),
            decode_mode="stochastic",
            median_filter_window=1,
            phi_threshold=float(ar1_payload.get("phi_threshold", 0.3)),
            clamp_range=(
                payload["norm_cfg"]["power_min"],
                payload["norm_cfg"]["power_max"],
            ),
        )
        pred = np.asarray(generated["power_w"], dtype=np.float64).reshape(-1)
        gt = np.asarray(tr.power[1:], dtype=np.float64).reshape(-1)
        expected_n = int(len(gt))
        if expected_n <= 0:
            raise ValueError(
                f"No ground-truth points for MoE simulation: {tr.config_id} trace={tr.trace_idx}"
            )
        if int(len(pred)) != expected_n:
            raise ValueError(
                f"Length mismatch for MoE simulation: {tr.config_id} trace={tr.trace_idx} "
                f"pred_len={len(pred)} gt_len={expected_n}. Check feature horizon/request alignment."
            )
        return {
            "time_s": np.arange(expected_n, dtype=np.float64) * float(tr.dt),
            "measured_w": gt,
            "simulated_w": pred,
            "requests_count": int(payload["requests_count"]),
            "ar1_params_path": str(ar1_path),
        }

    def _generate_bic_figures(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        overlay_series: List[Dict[str, Any]] = []
        k_values_ref: Optional[List[int]] = None
        for tag, spec in BIC_CONFIGS.items():
            cid = str(spec["config_id"])
            title = str(spec["title"])
            legend = str(spec.get("legend", title))
            power = self._collect_power_for_rate(cid, 1.0)
            sweep = bic_sweep(power, k_values=K_VALUES)
            bic_norm = normalize_bic_values(sweep["bic_values"])
            if k_values_ref is None:
                k_values_ref = [int(v) for v in sweep["k_values"]]
            elif [int(v) for v in sweep["k_values"]] != k_values_ref:
                raise ValueError("Inconsistent K grid across BIC sweeps.")
            file_name = f"{tag}.pdf"
            if not self.dry_run:
                _plot_bic_curve(
                    self.out_dir / file_name,
                    title=title,
                    k_values=sweep["k_values"],
                    bic_values=sweep["bic_values"],
                    best_k=int(sweep["best_k"]),
                )
            out[tag] = {
                "config_id": cid,
                "title": title,
                "legend": legend,
                "scope_rate": 1.0,
                "k_values": sweep["k_values"],
                "bic_values": sweep["bic_values"],
                "bic_norm_values": bic_norm,
                "best_k": int(sweep["best_k"]),
                "n_points": int(sweep["n_points"]),
                "file": str(self.out_dir / file_name),
            }
            overlay_series.append(
                {
                    "tag": tag,
                    "legend": legend,
                    "bic_norm_values": bic_norm,
                    "best_k": int(sweep["best_k"]),
                }
            )
        overlay_name = "bic_normalized_overlay.pdf"
        if k_values_ref is None:
            raise ValueError("No BIC curves were generated.")
        if not self.dry_run:
            _plot_bic_normalized_overlay(
                self.out_dir / overlay_name,
                k_values=k_values_ref,
                series=overlay_series,
            )
        out["normalized_overlay"] = {
            "k_values": [int(v) for v in k_values_ref],
            "series": overlay_series,
            "file": str(self.out_dir / overlay_name),
            "normalization": "per_curve_min_max",
            "plateau_reference_k": 10,
        }
        return out

    def _generate_at_overlay(self) -> Dict[str, Any]:
        selection = self._select_dense_trace_for_rate(
            float(AT_OVERLAY_RATE),
            prefer_lowest_mean_power=False,
        )
        trace_idx = int(selection["trace_idx"])
        tr = self.get_trace(self.dense_config_id, trace_idx)
        selection_method = str(selection["selection_method"])

        # STEP 1: Initial pass to detect offset
        requests_initial = self._load_requests_for_trace(
            tr, require_recorded_timestamps=True
        )
        t_horizon = int(max(0, len(tr.power) - 1))

        # Compute initial A_t without offset correction
        art = self._resolve_run_artifacts(tr.config_id)
        feat_initial = build_rollout_features_from_requests(
            requests=requests_initial,
            throughput=art["throughput"],
            norm=art["norm_cfg"],
            T=t_horizon,
            dt=float(tr.dt),
            feature_set=str(art["feature_set"]),
        )
        a_initial = np.asarray(feat_initial["A_raw"], dtype=np.float64).reshape(-1)

        # STEP 2: Detect offset
        # Use tr.power[1:] to match what we plot (important for alignment!)
        power_for_plot = np.asarray(tr.power[1:], dtype=np.float64).reshape(-1)
        power_spike_bin = detect_first_power_spike(
            power_for_plot,
            dt=float(tr.dt),
            idle_threshold=150.0,
            active_threshold=200.0,
            window_bins=3,
        )
        at_spike_bin = detect_first_at_activation(a_initial)
        offset_bins = power_spike_bin - at_spike_bin
        offset_seconds = float(offset_bins) * float(tr.dt)

        # STEP 3: Reload requests with offset correction
        requests_corrected = self._load_requests_for_trace(
            tr,
            alignment_offset_s=offset_seconds,
            require_recorded_timestamps=True,
        )

        # STEP 4: Compute corrected features (replaces original payload computation)
        feat_corrected = build_rollout_features_from_requests(
            requests=requests_corrected,
            throughput=art["throughput"],
            norm=art["norm_cfg"],
            T=t_horizon,
            dt=float(tr.dt),
            feature_set=str(art["feature_set"]),
        )

        # Build payload with corrected features
        payload = {
            "features": feat_corrected,
            "requests_count": len(requests_corrected),
        }

        a_raw = np.asarray(payload["features"]["A_raw"], dtype=np.float64).reshape(-1)
        gt = np.asarray(tr.power[1:], dtype=np.float64).reshape(-1)
        n = int(min(len(a_raw), len(gt)))
        if n <= 0:
            raise ValueError("No aligned points for A_t overlay.")
        a = a_raw[:n]
        p = gt[:n]

        if AT_OVERLAY_WINDOW_SECONDS is None:
            start, end = 0, n
        else:
            window_bins = int(round(float(AT_OVERLAY_WINDOW_SECONDS) / float(tr.dt)))
            start, end = select_transition_dense_window(a, window_bins)
        t = np.arange(end - start, dtype=np.float64) * float(tr.dt)
        p_win = p[start:end]
        a_win = a[start:end]

        out_path = self.out_dir / "at-overlay.pdf"
        if not self.dry_run:
            _plot_at_overlay(out_path, time_s=t, power_w=p_win, a_t=a_win)

        return {
            "config_id": tr.config_id,
            "trace_idx": int(tr.trace_idx),
            "pair_key": tr.pair_key,
            "rate": tr.rate,
            "selection_rate": float(AT_OVERLAY_RATE),
            "selection_method": selection_method,
            "request_timestamps_source": "recorded_only",
            "window_mode": "full_trace"
            if AT_OVERLAY_WINDOW_SECONDS is None
            else "transition_dense_window",
            "requested_window_seconds": None
            if AT_OVERLAY_WINDOW_SECONDS is None
            else float(AT_OVERLAY_WINDOW_SECONDS),
            "dt": float(tr.dt),
            "full_trace_duration_seconds": float(max(0, len(tr.power) - 1) * tr.dt),
            "window_seconds": float((end - start) * tr.dt),
            "window_start_idx": int(start),
            "window_end_idx": int(end),
            "requests_count": int(payload["requests_count"]),
            "alignment_correction": {
                "power_spike_bin": int(power_spike_bin),
                "at_spike_bin_before": int(at_spike_bin),
                "offset_bins": int(offset_bins),
                "offset_seconds": float(offset_seconds),
                "power_at_detected_spike_w": float(power_for_plot[power_spike_bin])
                if power_spike_bin < len(power_for_plot)
                else None,
                "detection_thresholds": {
                    "idle_threshold_w": 150.0,
                    "active_threshold_w": 250.0,
                    "window_bins": 3,
                },
            },
            "file": str(out_path),
        }

    def _generate_dense_overlays(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        file_map = {
            "sparse": "simulated_power_trace_sparse.pdf",
            "medium": "simulated_power_trace_medium.pdf",
            "high": "simulated_power_trace_high.pdf",
        }
        for tag, rate in DENSE_OVERLAY_RATES.items():
            selection = self._select_dense_trace_for_rate(
                float(rate),
                prefer_lowest_mean_power=(tag == "sparse"),
            )
            trace_idx = int(selection["trace_idx"])
            trace_selection_method = str(selection["selection_method"])

            try:
                seed = select_seed_nearest_median_nrmse(
                    self.per_seed_rows,
                    config_id=self.dense_config_id,
                    trace_idx=int(trace_idx),
                )
                seed_selection_method = "nearest_median_nrmse"
            except ValueError:
                seed = int(DEFAULT_SIM_SEED)
                seed_selection_method = "fallback_default_seed_no_eval_rows"
            tr = self.get_trace(self.dense_config_id, int(trace_idx))
            simulated = self._simulate_dense_trace(tr, seed=int(seed))
            measured = np.asarray(simulated["measured_w"], dtype=np.float64).reshape(-1)
            mean_power = float(np.mean(measured)) if measured.size > 0 else float("nan")
            frac_hi = (
                float(np.mean(measured > 400.0)) if measured.size > 0 else float("nan")
            )
            frac_idle = (
                float(np.mean(measured < 150.0)) if measured.size > 0 else float("nan")
            )
            out_path = self.out_dir / file_map[tag]
            if not self.dry_run:
                _plot_sim_overlay(
                    out_path,
                    time_s=np.asarray(simulated["time_s"], dtype=np.float64),
                    measured=np.asarray(simulated["measured_w"], dtype=np.float64),
                    simulated=np.asarray(simulated["simulated_w"], dtype=np.float64),
                )
            out[tag] = {
                "config_id": tr.config_id,
                "trace_idx": int(tr.trace_idx),
                "pair_key": tr.pair_key,
                "rate": tr.rate,
                "selection_rate": float(rate),
                "selection_method": trace_selection_method,
                "request_timestamps_source": "recorded_only",
                "seed": int(seed),
                "seed_selection_method": seed_selection_method,
                "dt": float(tr.dt),
                "num_points": int(len(simulated["time_s"])),
                "duration_seconds": float(len(simulated["time_s"]) * tr.dt),
                "requests_count": int(simulated["requests_count"]),
                "measured_mean_power_w": mean_power,
                "measured_frac_above_400w": frac_hi,
                "measured_frac_below_150w": frac_idle,
                "file": str(out_path),
            }
        return out

    def _generate_moe_overlay(self) -> Dict[str, Any]:
        def _candidate_rows(
            *,
            config_id: Optional[str],
            rate: Optional[float],
            require_pair_json: bool = True,
            require_moe: bool = False,
        ) -> List[Tuple[float, str, int, Dict[str, str]]]:
            out: List[Tuple[float, str, int, Dict[str, str]]] = []
            for row in self.per_trace_rows:
                if str(row.get("status", "")) != "evaluated":
                    continue
                cid = str(row.get("config_id", "")).strip()
                if cid == "":
                    continue
                if config_id is not None and cid != str(config_id):
                    continue
                if require_moe and (not is_moe_config_id(cid)):
                    continue
                if rate is not None and (
                    not rate_matches(row.get("rate", ""), float(rate))
                ):
                    continue
                try:
                    trace_idx_local = int(row.get("trace_idx", "-1"))
                except Exception:
                    continue
                if trace_idx_local < 0:
                    continue
                nrmse = _to_float(row.get("nrmse_median"))
                if nrmse is None:
                    continue
                pair_key = str(row.get("pair_key", "")).strip()
                if require_pair_json:
                    if pair_key not in self.pair_map:
                        continue
                    if not self._pair_has_recorded_request_timestamps(pair_key):
                        continue
                ar1_path = self.ar1_params_dir / f"{_safe_slug(cid)}_ar1_params.json"
                if not ar1_path.exists():
                    continue
                out.append((float(nrmse), cid, int(trace_idx_local), row))
            out.sort(key=lambda x: (x[0], x[1], x[2]))
            return out

        selection_method = "best_median_nrmse_per_rate"
        candidates = _candidate_rows(
            config_id=self.moe_proxy_config_id,
            rate=float(self.moe_proxy_rate),
            require_pair_json=True,
            require_moe=False,
        )
        if not candidates:
            selection_method = "fallback_best_median_nrmse_any_rate_pair_json"
            candidates = _candidate_rows(
                config_id=self.moe_proxy_config_id,
                rate=None,
                require_pair_json=True,
                require_moe=False,
            )
        if not candidates:
            selection_method = (
                "fallback_best_median_nrmse_any_moe_config_any_rate_pair_json"
            )
            candidates = _candidate_rows(
                config_id=None,
                rate=None,
                require_pair_json=True,
                require_moe=True,
            )
        if not candidates:
            if self._try_autoswitch_gptoss_a100_bundle():
                selection_method = "autoswitched_gptoss_a100_bundle_fallback_best_median_nrmse_any_moe_config_any_rate_pair_json"
                candidates = _candidate_rows(
                    config_id=None,
                    rate=None,
                    require_pair_json=True,
                    require_moe=True,
                )
            if not candidates:
                raise ValueError(
                    "No MoE per-trace candidates have both request JSON paths and AR(1) params. "
                    "Provide explicit --moe-config-id/--moe-rate and matching manifest paths."
                )

        _, selected_config_id, trace_idx, selected_row = candidates[0]
        try:
            seed = select_seed_nearest_median_nrmse(
                self.per_seed_ar1_rows,
                config_id=selected_config_id,
                trace_idx=int(trace_idx),
            )
            seed_selection_method = "nearest_median_nrmse"
        except ValueError:
            seed = int(DEFAULT_SIM_SEED)
            seed_selection_method = "fallback_default_seed_no_eval_rows"

        tr = self.get_trace(selected_config_id, int(trace_idx))
        simulated = self._simulate_moe_proxy_trace(tr, seed=int(seed))
        out_path = self.out_dir / "simulated_power_trace_moe.pdf"
        if not self.dry_run:
            _plot_sim_overlay(
                out_path,
                time_s=np.asarray(simulated["time_s"], dtype=np.float64),
                measured=np.asarray(simulated["measured_w"], dtype=np.float64),
                simulated=np.asarray(simulated["simulated_w"], dtype=np.float64),
            )
        return {
            "config_id": tr.config_id,
            "trace_idx": int(tr.trace_idx),
            "pair_key": tr.pair_key,
            "rate": tr.rate,
            "selection_rate": float(
                _to_float(selected_row.get("rate", self.moe_proxy_rate))
                if _to_float(selected_row.get("rate", self.moe_proxy_rate)) is not None
                else self.moe_proxy_rate
            ),
            "selection_method": selection_method,
            "seed": int(seed),
            "seed_selection_method": seed_selection_method,
            "dt": float(tr.dt),
            "num_points": int(len(simulated["time_s"])),
            "duration_seconds": float(len(simulated["time_s"]) * tr.dt),
            "requests_count": int(simulated["requests_count"]),
            "ar1_params_path": str(simulated["ar1_params_path"]),
            "requested_config_id": str(self.moe_proxy_config_id),
            "requested_rate": float(self.moe_proxy_rate),
            "selected_rate_from_per_trace": str(selected_row.get("rate", "")),
            "file": str(out_path),
        }

    def _ensure_validation_pdfs(self) -> Dict[str, Any]:
        copied: List[str] = []
        for name in VALIDATION_PDFS:
            src = self.validation_source_dir / name
            dst = self.out_dir / name
            if not src.exists():
                raise ValueError(f"Validation PDF missing: {src}")
            if not self.dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
            copied.append(str(dst))
        return {"files": copied}

    def _generate_gmm_structure_figures(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for tag, spec in GMM_STRUCTURE_CONFIGS.items():
            tr = self.get_trace(str(spec["config_id"]), int(spec["trace_idx"]))
            sweep = bic_sweep(tr.power, k_values=K_VALUES)
            k_star = int(sweep["best_k"])
            gmm = _fit_gmm(tr.power, k_star)
            means = np.asarray(gmm.means_, dtype=np.float64).reshape(-1)
            order = np.argsort(means)
            means_s = means[order]
            cov = np.asarray(gmm.covariances_, dtype=np.float64)
            if cov.ndim == 3:
                vars_s = cov.reshape(len(order), -1)[:, 0][order]
            elif cov.ndim == 2:
                vars_s = cov[:, 0][order]
            else:
                vars_s = cov.reshape(len(order))[order]
            vars_s = np.clip(vars_s, a_min=1e-12, a_max=None)
            stds_s = np.sqrt(vars_s)
            weights_s = np.asarray(gmm.weights_, dtype=np.float64).reshape(-1)[order]

            out_name = (
                "gmm_structure_dense.pdf" if tag == "dense" else "gmm_structure_moe.pdf"
            )
            out_path = self.out_dir / out_name
            if not self.dry_run:
                _plot_gmm_structure(
                    out_path,
                    title=str(spec["title"]),
                    power_values=tr.power,
                    weights=weights_s,
                    means=means_s,
                    stds=stds_s,
                )
            out[tag] = {
                "config_id": tr.config_id,
                "trace_idx": int(tr.trace_idx),
                "pair_key": tr.pair_key,
                "rate": tr.rate,
                "best_k": k_star,
                "k_values": sweep["k_values"],
                "bic_values": sweep["bic_values"],
                "weights": weights_s.tolist(),
                "means": means_s.tolist(),
                "stds": stds_s.tolist(),
                "file": str(out_path),
            }
        return out

    def _completion_check(self) -> None:
        required = [
            "bic_config1.pdf",
            "bic_config2.pdf",
            "bic_config3.pdf",
            "bic_config4.pdf",
            "bic_normalized_overlay.pdf",
            "at-overlay.pdf",
            "simulated_power_trace_sparse.pdf",
            "simulated_power_trace_medium.pdf",
            "simulated_power_trace_high.pdf",
            "simulated_power_trace_moe.pdf",
            "validation_deepseek-r1-distill_8b_h100_tp8_ttft.pdf",
            "validation_deepseek-r1-distill_8b_h100_tp8_decode.pdf",
            "gmm_structure_dense.pdf",
            "gmm_structure_moe.pdf",
        ]
        missing = [
            str(self.out_dir / name)
            for name in required
            if not (self.out_dir / name).exists()
        ]
        if missing:
            raise RuntimeError(f"Missing required outputs: {missing}")

    def generate(self) -> Dict[str, Any]:
        apply_publication_style()
        if not self.dry_run:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        manifest: Dict[str, Any] = {
            "schema_version": "methods-figures-v1",
            "generated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "assumptions": {
                "llama_31_alias_maps_to_llama_3_8b_ids": True,
                "deepseek_r1_distill_treated_as_dense": True,
                "moe_proxy": self.moe_proxy_config_id,
                "bic_scope_rate": 1.0,
                "reuse_validation_pdfs": True,
                "include_optional_moe_trace": True,
            },
            "inputs": {
                "experimental_manifest": str(self.experimental_manifest_path),
                "pair_manifest_csv": str(self.pair_manifest_csv_path),
                "run_manifest": str(self.run_manifest_path),
                "per_trace_metrics_csv": str(self.per_trace_csv_path),
                "per_seed_metrics_csv": str(self.per_seed_csv_path),
                "per_seed_ar1_metrics_csv": str(self.per_seed_ar1_csv_path),
                "ar1_params_dir": str(self.ar1_params_dir),
                "throughput_db": str(self.throughput_db_path),
            },
            "resolved_policy": {
                "dense_config_id": self.dense_config_id,
                "deepseek_dense_config_id": self.deepseek_dense_config_id,
                "dense_overlay_rates": DENSE_OVERLAY_RATES,
                "at_overlay_rate": float(AT_OVERLAY_RATE),
                "at_overlay_window_seconds": None
                if AT_OVERLAY_WINDOW_SECONDS is None
                else float(AT_OVERLAY_WINDOW_SECONDS),
                "moe_proxy_rate": float(self.moe_proxy_rate),
                "bic_configs": BIC_CONFIGS,
            },
        }

        if self.figure_mode == "simulated-moe":
            manifest["figures"] = {
                "simulated_moe": self._generate_moe_overlay(),
            }
            manifest["output_paths"] = {
                "out_dir": str(self.out_dir),
                "methods_figure_manifest": str(
                    self.out_dir / "methods_figure_manifest.json"
                ),
            }
            if self.dry_run:
                summary = {
                    "dry_run": True,
                    "figure_mode": self.figure_mode,
                    "moe_proxy_config_id": self.moe_proxy_config_id,
                    "moe_proxy_rate": float(self.moe_proxy_rate),
                    "simulated_moe": manifest["figures"]["simulated_moe"],
                }
                print(json.dumps(summary, indent=2, sort_keys=True))
                return manifest
            _write_json(self.out_dir / "methods_figure_manifest.json", manifest)
            return manifest

        # Diagnostics for deterministic selection helpers (recorded timestamps only).
        dense_rates_for_diag = [
            AT_OVERLAY_RATE,
            DENSE_OVERLAY_RATES["sparse"],
            DENSE_OVERLAY_RATES["medium"],
            DENSE_OVERLAY_RATES["high"],
        ]
        dense_selection_by_rate = {
            str(rate): self._select_dense_trace_for_rate(
                float(rate), prefer_lowest_mean_power=False
            )
            for rate in dense_rates_for_diag
        }
        dense_recorded_candidates_by_rate = {
            key: value["recorded_candidate_trace_indices"]
            for key, value in dense_selection_by_rate.items()
        }
        dense_trace_by_rate_best_nrmse = {
            key: int(value["trace_idx"])
            for key, value in dense_selection_by_rate.items()
        }
        dense_trace_selection_method_by_rate = {
            key: str(value["selection_method"])
            for key, value in dense_selection_by_rate.items()
        }
        sparse_trace_overlay = int(
            self._select_dense_trace_for_rate(
                float(DENSE_OVERLAY_RATES["sparse"]),
                prefer_lowest_mean_power=True,
            )["trace_idx"]
        )
        dense_overlay_trace_by_tag = {
            "sparse": sparse_trace_overlay,
            "medium": int(
                self._select_dense_trace_for_rate(
                    float(DENSE_OVERLAY_RATES["medium"]),
                    prefer_lowest_mean_power=False,
                )["trace_idx"]
            ),
            "high": int(
                self._select_dense_trace_for_rate(
                    float(DENSE_OVERLAY_RATES["high"]),
                    prefer_lowest_mean_power=False,
                )["trace_idx"]
            ),
        }
        moe_trace_idx = int(
            select_trace_by_best_median_nrmse(
                self.per_trace_rows,
                config_id=self.moe_proxy_config_id,
                rate=float(self.moe_proxy_rate),
            )
        )
        dense_overlay_seed_by_tag: Dict[str, int] = {}
        dense_overlay_seed_method_by_tag: Dict[str, str] = {}
        for tag, trace_idx in dense_overlay_trace_by_tag.items():
            try:
                dense_overlay_seed_by_tag[tag] = int(
                    select_seed_nearest_median_nrmse(
                        self.per_seed_rows,
                        config_id=self.dense_config_id,
                        trace_idx=int(trace_idx),
                    )
                )
                dense_overlay_seed_method_by_tag[tag] = "nearest_median_nrmse"
            except ValueError:
                dense_overlay_seed_by_tag[tag] = int(DEFAULT_SIM_SEED)
                dense_overlay_seed_method_by_tag[tag] = (
                    "fallback_default_seed_no_eval_rows"
                )
        dense_seed_by_rate_best_nrmse: Dict[str, int] = {}
        dense_seed_method_by_rate_best_nrmse: Dict[str, str] = {}
        for rate, trace_idx in dense_trace_by_rate_best_nrmse.items():
            try:
                dense_seed_by_rate_best_nrmse[str(rate)] = int(
                    select_seed_nearest_median_nrmse(
                        self.per_seed_rows,
                        config_id=self.dense_config_id,
                        trace_idx=int(trace_idx),
                    )
                )
                dense_seed_method_by_rate_best_nrmse[str(rate)] = "nearest_median_nrmse"
            except ValueError:
                dense_seed_by_rate_best_nrmse[str(rate)] = int(DEFAULT_SIM_SEED)
                dense_seed_method_by_rate_best_nrmse[str(rate)] = (
                    "fallback_default_seed_no_eval_rows"
                )

        try:
            moe_seed_rate_1 = int(
                select_seed_nearest_median_nrmse(
                    self.per_seed_ar1_rows,
                    config_id=self.moe_proxy_config_id,
                    trace_idx=int(moe_trace_idx),
                )
            )
            moe_seed_rate_1_method = "nearest_median_nrmse"
        except ValueError:
            moe_seed_rate_1 = int(DEFAULT_SIM_SEED)
            moe_seed_rate_1_method = "fallback_default_seed_no_eval_rows"
        helper_dense = {
            "dense_recorded_candidate_trace_indices_by_rate": dense_recorded_candidates_by_rate,
            "dense_trace_by_rate_best_median_nrmse": dense_trace_by_rate_best_nrmse,
            "dense_trace_selection_method_by_rate": dense_trace_selection_method_by_rate,
            "dense_seed_by_rate_best_median_nrmse": dense_seed_by_rate_best_nrmse,
            "dense_seed_method_by_rate_best_median_nrmse": dense_seed_method_by_rate_best_nrmse,
            "dense_overlay_trace_by_tag": dense_overlay_trace_by_tag,
            "dense_overlay_seed_by_tag": dense_overlay_seed_by_tag,
            "dense_overlay_seed_method_by_tag": dense_overlay_seed_method_by_tag,
            "moe_trace_idx_rate_1.0": int(moe_trace_idx),
            "moe_seed_rate_1.0": int(moe_seed_rate_1),
            "moe_seed_rate_1.0_method": moe_seed_rate_1_method,
        }
        manifest["helper_diagnostics"] = helper_dense

        manifest["figures"] = {
            "bic": self._generate_bic_figures(),
            "at_overlay": self._generate_at_overlay(),
            "simulated_dense": self._generate_dense_overlays(),
            "validation_reused": self._ensure_validation_pdfs(),
            "gmm_structure": self._generate_gmm_structure_figures(),
            "simulated_moe": self._generate_moe_overlay(),
        }

        manifest["output_paths"] = {
            "out_dir": str(self.out_dir),
            "methods_figure_manifest": str(
                self.out_dir / "methods_figure_manifest.json"
            ),
        }

        if self.dry_run:
            summary = {
                "dry_run": True,
                "bic_best_k": {
                    tag: int(row["best_k"])
                    for tag, row in manifest["figures"]["bic"].items()
                },
                "dense_overlay_rates": DENSE_OVERLAY_RATES,
                "at_overlay_rate": float(AT_OVERLAY_RATE),
                "moe_proxy_rate": float(self.moe_proxy_rate),
                "helper_diagnostics": helper_dense,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return manifest

        _write_json(self.out_dir / "methods_figure_manifest.json", manifest)
        self._completion_check()
        return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic methods figures for revised paper sections."
    )
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--figure-mode", choices=["all", "simulated-moe"], default="all"
    )
    parser.add_argument(
        "--experimental-manifest",
        default="results/experimental_continuous_v1/manifest.json",
    )
    parser.add_argument(
        "--pair-manifest-csv",
        default="results/stage0/pair_manifest.csv",
    )
    parser.add_argument(
        "--run-manifest",
        default="results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    )
    parser.add_argument(
        "--per-trace-csv",
        default="results/continuous_v1_gmm_bigru/k10_f2/eval_metrics/per_trace_metrics.csv",
    )
    parser.add_argument(
        "--per-seed-csv",
        default="results/continuous_v1_gmm_bigru/k10_f2/eval_metrics/per_seed_metrics.csv",
    )
    parser.add_argument(
        "--per-seed-ar1-csv",
        default="results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/eval_metrics/per_seed_metrics.csv",
    )
    parser.add_argument(
        "--ar1-params-dir",
        default="results/continuous_v1_gmm_bigru/k10_f2_ar1_thresh/ar1_params",
    )
    parser.add_argument(
        "--throughput-db", default="model/config/throughput_database.json"
    )
    parser.add_argument(
        "--validation-source-dir", default="model/tests/validation_results"
    )
    parser.add_argument("--dense-config-id", default=DENSE_CONFIG_ID)
    parser.add_argument("--deepseek-dense-config-id", default=DEEPSEEK_DENSE_CONFIG_ID)
    parser.add_argument("--moe-config-id", default=MOE_PROXY_CONFIG_ID)
    parser.add_argument("--moe-rate", type=float, default=MOE_PROXY_RATE)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    generator = MethodsFigureGenerator(
        out_dir=args.out_dir,
        dry_run=bool(args.dry_run),
        device=args.device,
        experimental_manifest_path=args.experimental_manifest,
        pair_manifest_csv_path=args.pair_manifest_csv,
        run_manifest_path=args.run_manifest,
        per_trace_csv_path=args.per_trace_csv,
        per_seed_csv_path=args.per_seed_csv,
        per_seed_ar1_csv_path=args.per_seed_ar1_csv,
        ar1_params_dir=args.ar1_params_dir,
        throughput_db_path=args.throughput_db,
        validation_source_dir=args.validation_source_dir,
        dense_config_id=args.dense_config_id,
        deepseek_dense_config_id=args.deepseek_dense_config_id,
        moe_proxy_config_id=args.moe_config_id,
        moe_proxy_rate=float(args.moe_rate),
        figure_mode=args.figure_mode,
    )
    run = generator.generate()
    if args.dry_run:
        print("[generate_methods_figures] Dry run complete")
    else:
        print("[generate_methods_figures] Done")
        print(f"  out_dir: {run['output_paths']['out_dir']}")
        print(f"  manifest: {run['output_paths']['methods_figure_manifest']}")


if __name__ == "__main__":
    main()
