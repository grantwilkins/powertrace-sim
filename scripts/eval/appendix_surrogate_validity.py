#!/usr/bin/env python3
"""
Appendix A1 surrogate-validity pipeline (A_t only).

Produces measured-vs-surrogate sanity checks for:
- A_t time-series per selected config (separate files)
- A_t histogram per selected config (separate files)
- Lambda-vs-mean(A_t) scatter and saturation diagnostics

Queue analysis is intentionally omitted.
This script is lightweight and avoids torch/sklearn imports.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from model.utils.io import ensure_dir, load_json, safe_slug, write_json

CONFIG_70B_TP4_RE = re.compile(r"^.+-70b_(A100|H100)_tp4$")
CONFIG_70B_ALL_TP_RE = re.compile(r"^.+-70b_(A100|H100)_tp\d+$")

STYLE = {
    "a_measured": {
        "label": "Measured",
        "color": "#111111",
        "linewidth": 1.6,
        "alpha": 0.9,
    },
    "a_surrogate": {
        "label": "Surrogate",
        "color": "#1f77b4",
        "linewidth": 1.4,
        "alpha": 0.9,
    },
    "scatter_measured": {
        "label": "Measured",
        "color": "#111111",
        "marker": "o",
        "alpha": 0.85,
    },
    "scatter_surrogate": {
        "label": "Surrogate",
        "color": "#1f77b4",
        "marker": "s",
        "alpha": 0.85,
    },
}


@dataclass(frozen=True)
class TraceRow:
    config_id: str
    trace_index: int
    pair_key: str
    rate_label: str
    dt: float
    n_timesteps: int
    n_requests: int
    lambda_emp: float
    mean_a_measured: float
    p95_a_measured: float
    p99_a_measured: float
    mean_a_surrogate: float
    p95_a_surrogate: float
    p99_a_surrogate: float
    corr_a: float
    mae_a: float
    rmse_a: float
    status: str
    reason: str


@dataclass(frozen=True)
class ConfigSummary:
    config_id: str
    n_test_traces: int
    n_eligible_traces: int
    median_lambda_emp: float
    median_corr_a: float
    median_rmse_a: float
    median_mae_a: float
    median_mean_a_measured: float
    median_mean_a_surrogate: float
    measured_slope_low: float
    measured_slope_high: float
    measured_saturation_flag: bool
    surrogate_slope_low: float
    surrogate_slope_high: float
    surrogate_saturation_flag: bool


@dataclass(frozen=True)
class SelectedConfig:
    bucket: str
    corr_threshold_used: float
    config_id: str


def _write_csv(
    path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]
) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _nanpercentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, float(q)))


def _nanmedian(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _nanmean(values: Iterable[float] | np.ndarray) -> float:
    arr = np.asarray(
        list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64
    ).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


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
            resolved = _resolve_existing_path(json_path_raw, base_dir)
            if resolved is not None:
                out[key] = resolved
    return out


def _list_candidate_configs(
    run_manifest: Mapping[str, object],
    experimental_manifest: Mapping[str, object],
    throughput_db: Mapping[str, object],
    config_pool: str,
) -> List[str]:
    run_cfgs = run_manifest.get("configs", {})
    exp_cfgs = experimental_manifest.get("configs", {})
    thr_cfgs = throughput_db.get("configs", {})
    if (
        not isinstance(run_cfgs, dict)
        or not isinstance(exp_cfgs, dict)
        or not isinstance(thr_cfgs, dict)
    ):
        raise ValueError("Invalid manifest format for configs section")

    base_ids: List[str] = []
    for cid, row in run_cfgs.items():
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")) != "trained":
            continue
        if cid not in exp_cfgs or cid not in thr_cfgs:
            continue
        base_ids.append(str(cid))

    mode = str(config_pool).strip().lower()
    if mode == "all_trained":
        return sorted(base_ids)
    if mode == "70b_tp4":
        return sorted(
            [cid for cid in base_ids if CONFIG_70B_TP4_RE.match(cid) is not None]
        )
    if mode == "70b_all_tp":
        return sorted(
            [cid for cid in base_ids if CONFIG_70B_ALL_TP_RE.match(cid) is not None]
        )
    raise ValueError(
        "config_pool must be one of {'all_trained','70b_tp4','70b_all_tp'}"
    )


def _resolve_throughput_entry(
    throughput_db: Mapping[str, object], config_id: str
) -> Dict[str, float]:
    cfgs = throughput_db.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("Invalid throughput DB format")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"throughput row missing for config {config_id}")
    prefill = _safe_float(row.get("prefill_rate_median_toks_per_s"))
    decode = _safe_float(row.get("decode_rate_median_toks_per_s"))
    if prefill is None or prefill <= 0.0:
        raise ValueError(f"invalid prefill throughput for {config_id}")
    if decode is None or decode <= 0.0:
        raise ValueError(f"invalid decode throughput for {config_id}")
    return {"lambda_prefill": float(prefill), "lambda_decode": float(decode)}


def _resolve_dataset_split_paths(
    experimental_manifest_path: str,
    experimental_manifest: Mapping[str, object],
    config_id: str,
) -> Tuple[str, str]:
    cfgs = experimental_manifest.get("configs", {})
    if not isinstance(cfgs, dict):
        raise ValueError("Invalid experimental manifest format")
    row = cfgs.get(config_id)
    if not isinstance(row, dict):
        raise ValueError(f"config_id '{config_id}' not found in experimental manifest")
    base = str(Path(experimental_manifest_path).resolve().parent)
    dpath = _resolve_existing_path(str(row.get("dataset_npz", "")), base)
    spath = _resolve_existing_path(str(row.get("split_json", "")), base)
    if dpath is None:
        raise ValueError(f"Dataset path not found for {config_id}")
    if spath is None:
        raise ValueError(f"Split path not found for {config_id}")
    return dpath, spath


def _build_requests_from_stage0_json(
    request_json_path: str,
    *,
    power_start_epoch_s: float,
    trace_duration_s: float,
    dt: float,
) -> Tuple[List[Dict[str, float]], str]:
    payload = load_json(request_json_path)
    input_lens_raw = payload.get("input_lens")
    output_lens_raw = payload.get("output_lens")
    ts_raw = payload.get("request_timestamps")

    if not isinstance(input_lens_raw, list) or not isinstance(output_lens_raw, list):
        return [], "missing_required_input_output_arrays"
    if not isinstance(ts_raw, list) or len(ts_raw) == 0:
        return [], "missing_recorded_request_timestamps"

    n = int(min(len(input_lens_raw), len(output_lens_raw), len(ts_raw)))
    if n <= 0:
        return [], "empty_request_arrays"

    arrivals = np.asarray(ts_raw[:n], dtype=np.float64) - float(power_start_epoch_s)
    if arrivals.size > 0 and (
        float(np.min(arrivals)) < -float(dt)
        or float(np.max(arrivals)) > float(trace_duration_s) + float(dt)
    ):
        arrivals = arrivals - float(np.min(arrivals))

    rows: List[Dict[str, float]] = []
    for i in range(n):
        arrival = _safe_float(arrivals[i])
        nin = _safe_float(input_lens_raw[i])
        nout = _safe_float(output_lens_raw[i])
        if arrival is None or nin is None or nout is None:
            continue
        rows.append(
            {
                "arrival_time": float(arrival),
                "input_tokens": float(max(0.0, nin)),
                "output_tokens": float(max(0.0, nout)),
            }
        )

    if len(rows) == 0:
        return [], "no_valid_requests_after_filtering"

    rows.sort(key=lambda x: float(x["arrival_time"]))
    return rows, ""


def _interval_count_series(
    *,
    starts: np.ndarray,
    ends: np.ndarray,
    dt: float,
    T: int,
) -> np.ndarray:
    if int(T) <= 0:
        return np.zeros((0,), dtype=np.float64)
    diff = np.zeros((int(T) + 1,), dtype=np.float64)
    starts_arr = np.asarray(starts, dtype=np.float64).reshape(-1)
    ends_arr = np.asarray(ends, dtype=np.float64).reshape(-1)
    if starts_arr.size != ends_arr.size:
        raise ValueError("starts and ends size mismatch")

    for s, e in zip(starts_arr, ends_arr):
        if not np.isfinite(s) or not np.isfinite(e):
            continue
        left = int(math.ceil(float(s) / float(dt)))
        right = int(math.ceil(float(e) / float(dt)))
        if right <= 0 or left >= int(T):
            continue
        l = max(0, left)
        r = min(int(T), right)
        if r <= l:
            continue
        diff[l] += 1.0
        diff[r] -= 1.0
    return np.clip(np.cumsum(diff[:-1]), a_min=0.0, a_max=None)


def _reconstruct_surrogate_A_t(
    requests: Sequence[Mapping[str, float]],
    *,
    lambda_prefill: float,
    lambda_decode: float,
    dt: float,
    T: int,
) -> np.ndarray:
    starts = []
    ends = []
    for req in requests:
        arrival = float(req["arrival_time"])
        nin = float(req["input_tokens"])
        nout = float(req["output_tokens"])
        service = (nin / float(lambda_prefill)) + (nout / float(lambda_decode))
        starts.append(arrival)
        ends.append(arrival + service)
    return _interval_count_series(
        starts=np.asarray(starts, dtype=np.float64),
        ends=np.asarray(ends, dtype=np.float64),
        dt=float(dt),
        T=int(T),
    )


def _extract_measured_A_t(active_requests: np.ndarray, T: int) -> np.ndarray:
    active = np.asarray(active_requests, dtype=np.float64).reshape(-1)
    if active.size <= 1 or int(T) <= 0:
        return np.zeros((0,), dtype=np.float64)
    return active[1 : 1 + int(T)].astype(np.float64)


def _compute_saturation_slopes(
    lambda_vals: np.ndarray, mean_vals: np.ndarray
) -> Dict[str, object]:
    x = np.asarray(lambda_vals, dtype=np.float64).reshape(-1)
    y = np.asarray(mean_vals, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 3:
        return {
            "slope_low": float("nan"),
            "slope_high": float("nan"),
            "saturation_flag": False,
        }

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    k = int(max(2, math.ceil(n / 3)))
    k = min(k, n)

    x_low = x[:k]
    y_low = y[:k]
    x_high = x[n - k :]
    y_high = y[n - k :]

    def _fit_slope(xx: np.ndarray, yy: np.ndarray) -> float:
        if xx.size < 2:
            return float("nan")
        if float(np.max(xx) - np.min(xx)) <= 1e-12:
            return float("nan")
        coeff = np.polyfit(xx, yy, deg=1)
        return float(coeff[0])

    slope_low = _fit_slope(x_low, y_low)
    slope_high = _fit_slope(x_high, y_high)
    flag = bool(
        np.isfinite(slope_low) and np.isfinite(slope_high) and (slope_high < slope_low)
    )
    return {"slope_low": slope_low, "slope_high": slope_high, "saturation_flag": flag}


def _corr_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(min(xx.size, yy.size))
    if n < 2:
        return float("nan")
    xx = xx[:n]
    yy = yy[:n]
    if float(np.std(xx)) <= 1e-12 or float(np.std(yy)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def _make_skipped_trace_row(
    *,
    config_id: str,
    trace_index: int,
    pair_key: str,
    rate_label: str,
    dt: float,
    reason: str,
) -> TraceRow:
    return TraceRow(
        config_id=config_id,
        trace_index=int(trace_index),
        pair_key=pair_key,
        rate_label=rate_label,
        dt=float(dt),
        n_timesteps=0,
        n_requests=0,
        lambda_emp=float("nan"),
        mean_a_measured=float("nan"),
        p95_a_measured=float("nan"),
        p99_a_measured=float("nan"),
        mean_a_surrogate=float("nan"),
        p95_a_surrogate=float("nan"),
        p99_a_surrogate=float("nan"),
        corr_a=float("nan"),
        mae_a=float("nan"),
        rmse_a=float("nan"),
        status="skipped",
        reason=str(reason),
    )


def _build_trace_row(
    *,
    config_id: str,
    trace_index: int,
    pair_key: str,
    rate_label: str,
    dt: float,
    T: int,
    n_requests: int,
    A_meas: np.ndarray,
    A_sur: np.ndarray,
) -> TraceRow:
    n = int(min(len(A_meas), len(A_sur), int(T)))
    if n <= 0:
        return _make_skipped_trace_row(
            config_id=config_id,
            trace_index=int(trace_index),
            pair_key=pair_key,
            rate_label=rate_label,
            dt=float(dt),
            reason="empty_aligned_horizon",
        )

    A_m = np.asarray(A_meas[:n], dtype=np.float64)
    A_s = np.asarray(A_sur[:n], dtype=np.float64)
    err = A_m - A_s
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    duration_s = float(n) * float(dt)
    lambda_emp = float(n_requests / duration_s) if duration_s > 0.0 else float("nan")

    return TraceRow(
        config_id=config_id,
        trace_index=int(trace_index),
        pair_key=pair_key,
        rate_label=rate_label,
        dt=float(dt),
        n_timesteps=int(n),
        n_requests=int(n_requests),
        lambda_emp=float(lambda_emp),
        mean_a_measured=_nanmean(A_m),
        p95_a_measured=_nanpercentile(A_m, 95.0),
        p99_a_measured=_nanpercentile(A_m, 99.0),
        mean_a_surrogate=_nanmean(A_s),
        p95_a_surrogate=_nanpercentile(A_s, 95.0),
        p99_a_surrogate=_nanpercentile(A_s, 99.0),
        corr_a=_corr_or_nan(A_m, A_s),
        mae_a=float(mae),
        rmse_a=float(rmse),
        status="evaluated",
        reason="",
    )


def _trace_row_to_dict(row: TraceRow) -> Dict[str, object]:
    return {
        "config_id": row.config_id,
        "trace_index": int(row.trace_index),
        "pair_key": row.pair_key,
        "rate_label": row.rate_label,
        "dt": float(row.dt),
        "n_timesteps": int(row.n_timesteps),
        "n_requests": int(row.n_requests),
        "lambda_emp": float(row.lambda_emp),
        "mean_a_measured": float(row.mean_a_measured),
        "p95_a_measured": float(row.p95_a_measured),
        "p99_a_measured": float(row.p99_a_measured),
        "mean_a_surrogate": float(row.mean_a_surrogate),
        "p95_a_surrogate": float(row.p95_a_surrogate),
        "p99_a_surrogate": float(row.p99_a_surrogate),
        "corr_a": float(row.corr_a),
        "mae_a": float(row.mae_a),
        "rmse_a": float(row.rmse_a),
        "status": row.status,
        "reason": row.reason,
    }


def _config_summary_to_dict(row: ConfigSummary) -> Dict[str, object]:
    return {
        "config_id": row.config_id,
        "n_test_traces": int(row.n_test_traces),
        "n_eligible_traces": int(row.n_eligible_traces),
        "median_lambda_emp": float(row.median_lambda_emp),
        "median_corr_a": float(row.median_corr_a),
        "median_rmse_a": float(row.median_rmse_a),
        "median_mae_a": float(row.median_mae_a),
        "median_mean_a_measured": float(row.median_mean_a_measured),
        "median_mean_a_surrogate": float(row.median_mean_a_surrogate),
        "measured_slope_low": float(row.measured_slope_low),
        "measured_slope_high": float(row.measured_slope_high),
        "measured_saturation_flag": bool(row.measured_saturation_flag),
        "surrogate_slope_low": float(row.surrogate_slope_low),
        "surrogate_slope_high": float(row.surrogate_slope_high),
        "surrogate_saturation_flag": bool(row.surrogate_saturation_flag),
    }


def _summarize_config(
    config_id: str, trace_rows: Sequence[TraceRow], n_test_traces: int
) -> ConfigSummary:
    eval_rows = [r for r in trace_rows if r.status == "evaluated"]

    lambdas = [r.lambda_emp for r in eval_rows]
    corrs = [r.corr_a for r in eval_rows]
    rmses = [r.rmse_a for r in eval_rows]
    maes = [r.mae_a for r in eval_rows]
    mean_a_meas = [r.mean_a_measured for r in eval_rows]
    mean_a_sur = [r.mean_a_surrogate for r in eval_rows]

    sat_measured = _compute_saturation_slopes(
        np.asarray(lambdas, dtype=np.float64),
        np.asarray(mean_a_meas, dtype=np.float64),
    )
    sat_surrogate = _compute_saturation_slopes(
        np.asarray(lambdas, dtype=np.float64),
        np.asarray(mean_a_sur, dtype=np.float64),
    )

    return ConfigSummary(
        config_id=config_id,
        n_test_traces=int(n_test_traces),
        n_eligible_traces=int(len(eval_rows)),
        median_lambda_emp=_nanmedian(lambdas),
        median_corr_a=_nanmedian(corrs),
        median_rmse_a=_nanmedian(rmses),
        median_mae_a=_nanmedian(maes),
        median_mean_a_measured=_nanmedian(mean_a_meas),
        median_mean_a_surrogate=_nanmedian(mean_a_sur),
        measured_slope_low=float(sat_measured["slope_low"]),
        measured_slope_high=float(sat_measured["slope_high"]),
        measured_saturation_flag=bool(sat_measured["saturation_flag"]),
        surrogate_slope_low=float(sat_surrogate["slope_low"]),
        surrogate_slope_high=float(sat_surrogate["slope_high"]),
        surrogate_saturation_flag=bool(sat_surrogate["saturation_flag"]),
    )


def _sort_config_candidates(configs: Sequence[ConfigSummary]) -> List[ConfigSummary]:
    def key(row: ConfigSummary) -> Tuple[float, float, str]:
        corr = row.median_corr_a
        rmse = row.median_rmse_a
        corr_score = -corr if np.isfinite(corr) else float("inf")
        rmse_score = rmse if np.isfinite(rmse) else float("inf")
        return (corr_score, rmse_score, str(row.config_id))

    return sorted(configs, key=key)


def _bucket_configs_by_lambda(
    configs: Sequence[ConfigSummary],
) -> Dict[str, List[ConfigSummary]]:
    rows = [r for r in configs if np.isfinite(r.median_lambda_emp)]
    if len(rows) == 0:
        return {"low": [], "mid": [], "high": []}

    lambdas = np.asarray([r.median_lambda_emp for r in rows], dtype=np.float64)
    q1, q2 = np.quantile(lambdas, [1.0 / 3.0, 2.0 / 3.0])

    out = {"low": [], "mid": [], "high": []}
    for row in rows:
        lam = float(row.median_lambda_emp)
        if lam <= float(q1):
            out["low"].append(row)
        elif lam <= float(q2):
            out["mid"].append(row)
        else:
            out["high"].append(row)
    return out


def _select_representative_configs(
    summaries: Sequence[ConfigSummary],
    *,
    num_representative_configs: int,
    min_eligible_traces: int,
    stable_corr_threshold: float,
) -> Tuple[List[SelectedConfig], Dict[str, object]]:
    if int(num_representative_configs) != 3:
        raise ValueError("num_representative_configs is currently fixed to 3")

    base = [
        s for s in summaries if int(s.n_eligible_traces) >= int(min_eligible_traces)
    ]

    buckets_order = ["low", "mid", "high"]
    selected: List[SelectedConfig] = []
    selected_ids: set[str] = set()
    notes: Dict[str, object] = {
        "selection_relaxation": {},
        "fallback_used": {},
    }

    for bucket in buckets_order:
        chosen: Optional[SelectedConfig] = None
        threshold = float(stable_corr_threshold)
        while threshold >= 0.5 - 1e-12:
            candidate_set = [
                s
                for s in base
                if np.isfinite(s.median_corr_a)
                and float(s.median_corr_a) >= float(threshold)
            ]
            if len(candidate_set) == 0:
                threshold -= 0.05
                continue

            bucketed = _bucket_configs_by_lambda(candidate_set)
            avail = [r for r in bucketed[bucket] if r.config_id not in selected_ids]
            ranked = _sort_config_candidates(avail)
            if len(ranked) > 0:
                chosen_row = ranked[0]
                chosen = SelectedConfig(
                    bucket=bucket,
                    corr_threshold_used=float(round(threshold, 2)),
                    config_id=str(chosen_row.config_id),
                )
                break
            threshold -= 0.05

        if chosen is None:
            remain = [r for r in base if r.config_id not in selected_ids]
            ranked = _sort_config_candidates(remain)
            if len(ranked) > 0:
                chosen = SelectedConfig(
                    bucket=bucket,
                    corr_threshold_used=float("nan"),
                    config_id=str(ranked[0].config_id),
                )
                notes["fallback_used"][bucket] = True
            else:
                notes["fallback_used"][bucket] = True
                continue
        else:
            notes["fallback_used"][bucket] = False

        selected.append(chosen)
        selected_ids.add(chosen.config_id)
        notes["selection_relaxation"][bucket] = {
            "corr_threshold_used": (
                float(chosen.corr_threshold_used)
                if np.isfinite(chosen.corr_threshold_used)
                else None
            )
        }

    if len(selected) < 3:
        remaining = [r for r in base if r.config_id not in selected_ids]
        ranked = _sort_config_candidates(remaining)
        for row in ranked:
            if len(selected) >= 3:
                break
            bucket_name = f"extra_{len(selected) + 1}"
            selected.append(
                SelectedConfig(
                    bucket=bucket_name,
                    corr_threshold_used=float("nan"),
                    config_id=str(row.config_id),
                )
            )
            selected_ids.add(row.config_id)

    return selected[:3], notes


def _choose_overlay_trace(
    rows: Sequence[TraceRow], summary: ConfigSummary
) -> Optional[TraceRow]:
    eval_rows = [r for r in rows if r.status == "evaluated"]
    if len(eval_rows) == 0:
        return None
    lam_med = float(summary.median_lambda_emp)

    def key(r: TraceRow) -> Tuple[float, float, int]:
        lam = float(r.lambda_emp)
        lam_delta = (
            abs(lam - lam_med)
            if np.isfinite(lam) and np.isfinite(lam_med)
            else float("inf")
        )
        rmse = float(r.rmse_a) if np.isfinite(r.rmse_a) else float("inf")
        return (lam_delta, rmse, int(r.trace_index))

    return sorted(eval_rows, key=key)[0]


def _build_per_config_figure_paths(
    out_figure_overlays: str,
    selected_rows: Sequence[Tuple[str, TraceRow]],
) -> List[Dict[str, object]]:
    out_path = Path(out_figure_overlays)
    out_dir = out_path.parent
    stem = out_path.stem

    rows: List[Dict[str, object]] = []
    for config_id, trace_row in selected_rows:
        slug = safe_slug(config_id).strip("-_") or "unknown"
        trace_idx = int(trace_row.trace_index)
        ts_file = out_dir / f"{stem}_{slug}_trace{trace_idx}_at_timeseries.pdf"
        hist_file = out_dir / f"{stem}_{slug}_trace{trace_idx}_at_histogram.pdf"
        rows.append(
            {
                "config_id": config_id,
                "trace_index": trace_idx,
                "timeseries_file": str(ts_file),
                "histogram_file": str(hist_file),
            }
        )
    return rows


def _plot_at_time_series(
    *,
    out_path: str,
    A_meas: np.ndarray,
    A_sur: np.ndarray,
    dt: float,
    time_window_s: float,
) -> None:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.9)

    n = int(min(len(A_meas), len(A_sur)))
    if n <= 0:
        return
    win_bins = int(max(1, math.floor(float(time_window_s) / float(dt))))
    n_win = int(min(n, win_bins))
    t = np.arange(n_win, dtype=np.float64) * float(dt)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.step(
        t,
        np.asarray(A_meas[:n_win], dtype=np.float64),
        where="post",
        color=STYLE["a_measured"]["color"],
        linewidth=STYLE["a_measured"]["linewidth"],
        alpha=STYLE["a_measured"]["alpha"],
        label=STYLE["a_measured"]["label"],
    )
    ax.step(
        t,
        np.asarray(A_sur[:n_win], dtype=np.float64),
        where="post",
        color=STYLE["a_surrogate"]["color"],
        linewidth=STYLE["a_surrogate"]["linewidth"],
        alpha=STYLE["a_surrogate"]["alpha"],
        label=STYLE["a_surrogate"]["label"],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("A_t")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ensure_dir(os.path.dirname(out_path) or ".")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_at_histogram(
    *,
    out_path: str,
    A_meas: np.ndarray,
    A_sur: np.ndarray,
) -> None:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.9)

    a_all = np.concatenate(
        [np.asarray(A_meas, dtype=np.float64), np.asarray(A_sur, dtype=np.float64)]
    )
    a_all = a_all[np.isfinite(a_all)]
    if a_all.size > 0:
        max_a = int(np.ceil(float(np.max(a_all))))
        bins = np.arange(-0.5, max_a + 1.5, 1.0) if max_a <= 80 else 40
    else:
        bins = 20

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(
        np.asarray(A_meas, dtype=np.float64),
        bins=bins,
        density=True,
        alpha=0.45,
        color=STYLE["a_measured"]["color"],
        label=STYLE["a_measured"]["label"],
    )
    ax.hist(
        np.asarray(A_sur, dtype=np.float64),
        bins=bins,
        density=True,
        alpha=0.45,
        color=STYLE["a_surrogate"]["color"],
        label=STYLE["a_surrogate"]["label"],
    )
    ax.set_xlabel("A_t")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ensure_dir(os.path.dirname(out_path) or ".")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 2:
        return float("nan"), float("nan")
    if float(np.max(xx) - np.min(xx)) <= 1e-12:
        return float("nan"), float("nan")
    coef = np.polyfit(xx, yy, deg=1)
    return float(coef[0]), float(coef[1])


def _plot_lambda_vs_mean_at(
    *,
    out_path: str,
    selected_config_ids: Sequence[str],
    trace_rows_by_config: Mapping[str, Sequence[TraceRow]],
    summary_by_config: Mapping[str, ConfigSummary],
) -> None:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.9)

    n_cfg = int(len(selected_config_ids))
    if n_cfg <= 0:
        raise ValueError("No selected configs to plot for lambda-vs-mean(A_t)")

    fig, axes = plt.subplots(1, n_cfg, figsize=(5.4 * n_cfg, 4.2), squeeze=False)
    axes_row = axes[0]

    for i, cid in enumerate(selected_config_ids):
        ax = axes_row[i]
        rows = [r for r in trace_rows_by_config.get(cid, []) if r.status == "evaluated"]
        lam = np.asarray([r.lambda_emp for r in rows], dtype=np.float64)
        y_meas = np.asarray([r.mean_a_measured for r in rows], dtype=np.float64)
        y_sur = np.asarray([r.mean_a_surrogate for r in rows], dtype=np.float64)

        ax.scatter(
            lam,
            y_meas,
            color=STYLE["scatter_measured"]["color"],
            marker=STYLE["scatter_measured"]["marker"],
            alpha=STYLE["scatter_measured"]["alpha"],
            label=STYLE["scatter_measured"]["label"],
        )
        ax.scatter(
            lam,
            y_sur,
            color=STYLE["scatter_surrogate"]["color"],
            marker=STYLE["scatter_surrogate"]["marker"],
            alpha=STYLE["scatter_surrogate"]["alpha"],
            label=STYLE["scatter_surrogate"]["label"],
        )

        m1, b1 = _fit_line(lam, y_meas)
        m2, b2 = _fit_line(lam, y_sur)
        if np.isfinite(m1) and np.isfinite(b1):
            xline = np.linspace(float(np.nanmin(lam)), float(np.nanmax(lam)), 80)
            ax.plot(
                xline,
                (m1 * xline) + b1,
                color=STYLE["scatter_measured"]["color"],
                linewidth=1.3,
                alpha=0.7,
            )
        if np.isfinite(m2) and np.isfinite(b2):
            xline = np.linspace(float(np.nanmin(lam)), float(np.nanmax(lam)), 80)
            ax.plot(
                xline,
                (m2 * xline) + b2,
                color=STYLE["scatter_surrogate"]["color"],
                linewidth=1.3,
                alpha=0.7,
            )

        summary = summary_by_config[cid]
        meas_text = (
            f"M: low={summary.measured_slope_low:.2f}, high={summary.measured_slope_high:.2f}, "
            f"sat={summary.measured_saturation_flag}"
        )
        sur_text = (
            f"S: low={summary.surrogate_slope_low:.2f}, high={summary.surrogate_slope_high:.2f}, "
            f"sat={summary.surrogate_saturation_flag}"
        )
        ax.text(
            0.02,
            0.98,
            meas_text + "\n" + sur_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#dddddd"},
        )

        ax.set_title(cid)
        ax.set_xlabel("Empirical arrival rate λ (req/s)")
        ax.set_ylabel("Mean A_t")
        ax.grid(True, alpha=0.25)

    handles, labels = axes_row[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _self_checks() -> Dict[str, object]:
    dt = 1.0
    T = 10
    req = [{"arrival_time": 2.2, "input_tokens": 2.0, "output_tokens": 3.0}]
    A = _reconstruct_surrogate_A_t(
        req, lambda_prefill=1.0, lambda_decode=1.0, dt=dt, T=T
    )
    assert int(np.sum(A)) == 5, "single-request A_t interval sanity failed"

    req2 = [
        {"arrival_time": 1.1, "input_tokens": 1.0, "output_tokens": 4.0},
        {"arrival_time": 2.3, "input_tokens": 1.0, "output_tokens": 4.0},
    ]
    A2 = _reconstruct_surrogate_A_t(
        req2, lambda_prefill=1.0, lambda_decode=1.0, dt=1.0, T=12
    )
    assert float(np.max(A2)) >= 2.0, "overlap sanity failed for A_t"

    return {
        "single_request_active_sum": int(np.sum(A)),
        "two_request_active_max": float(np.max(A2)),
    }


def _aggregate_surrogate_quality(rows: Sequence[TraceRow]) -> Dict[str, object]:
    eval_rows = [r for r in rows if r.status == "evaluated"]
    if len(eval_rows) == 0:
        return {
            "n_evaluated": 0,
            "mean_corr_a": float("nan"),
            "median_corr_a": float("nan"),
            "mean_mae_a": float("nan"),
            "median_mae_a": float("nan"),
            "mean_rmse_a": float("nan"),
            "median_rmse_a": float("nan"),
            "mean_abs_mean_a_error": float("nan"),
            "median_abs_mean_a_error": float("nan"),
        }

    corr = np.asarray([r.corr_a for r in eval_rows], dtype=np.float64)
    mae = np.asarray([r.mae_a for r in eval_rows], dtype=np.float64)
    rmse = np.asarray([r.rmse_a for r in eval_rows], dtype=np.float64)
    abs_mean_err = np.asarray(
        [abs(float(r.mean_a_measured) - float(r.mean_a_surrogate)) for r in eval_rows],
        dtype=np.float64,
    )

    return {
        "n_evaluated": int(len(eval_rows)),
        "mean_corr_a": _nanmean(corr),
        "median_corr_a": _nanmedian(corr),
        "mean_mae_a": _nanmean(mae),
        "median_mae_a": _nanmedian(mae),
        "mean_rmse_a": _nanmean(rmse),
        "median_rmse_a": _nanmedian(rmse),
        "mean_abs_mean_a_error": _nanmean(abs_mean_err),
        "median_abs_mean_a_error": _nanmedian(abs_mean_err),
    }


def run_appendix_surrogate_validity(
    *,
    run_manifest: str,
    experimental_manifest: str,
    pair_manifest_csv: str,
    throughput_db: str,
    config_pool: str,
    num_representative_configs: int,
    min_eligible_traces: int,
    stable_corr_threshold: float,
    time_window_s: float,
    out_csv: str,
    out_summary_csv: str,
    out_figure_overlays: str,
    out_figure_scatter: str,
    out_manifest_json: str,
    dry_run: bool,
) -> Dict[str, object]:
    if int(num_representative_configs) != 3:
        raise ValueError(
            "num_representative_configs must be 3 for low/mid/high selection"
        )
    if int(min_eligible_traces) <= 0:
        raise ValueError("min_eligible_traces must be >= 1")
    if float(stable_corr_threshold) < 0.0 or float(stable_corr_threshold) > 1.0:
        raise ValueError("stable_corr_threshold must be in [0,1]")
    if float(time_window_s) <= 0.0:
        raise ValueError("time_window_s must be > 0")

    checks = _self_checks()

    run_payload = load_json(run_manifest)
    exp_payload = load_json(experimental_manifest)
    throughput_payload = load_json(throughput_db)
    pair_map = _load_pair_manifest_map(pair_manifest_csv)

    target_config_ids = _list_candidate_configs(
        run_manifest=run_payload,
        experimental_manifest=exp_payload,
        throughput_db=throughput_payload,
        config_pool=config_pool,
    )
    if len(target_config_ids) == 0:
        raise ValueError(
            "No target configs found after filtering manifests/config_pool"
        )

    trace_rows: List[TraceRow] = []
    summary_rows: List[ConfigSummary] = []

    trace_arrays: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    trace_rows_by_config: Dict[str, List[TraceRow]] = {}

    for config_id in target_config_ids:
        throughput = _resolve_throughput_entry(throughput_payload, config_id)
        dataset_path, split_path = _resolve_dataset_split_paths(
            experimental_manifest_path=experimental_manifest,
            experimental_manifest=exp_payload,
            config_id=config_id,
        )
        split_payload = load_json(split_path)
        test_indices = [int(x) for x in split_payload.get("test_indices", [])]

        with np.load(dataset_path, allow_pickle=True) as data:
            pair_key_arr = np.asarray(data["pair_key"], dtype=object)
            power_arr = np.asarray(data["power"], dtype=object)
            power_start_arr = np.asarray(data["power_start_epoch_s"], dtype=np.float64)
            active_arr = np.asarray(data["active_requests"], dtype=object)
            rate_arr = (
                np.asarray(data["rate"], dtype=object)
                if "rate" in data
                else np.asarray([], dtype=object)
            )
            dt_arr = np.asarray(data["dt"], dtype=np.float64).reshape(-1)

        if dt_arr.size == 0:
            summary_rows.append(
                ConfigSummary(
                    config_id=config_id,
                    n_test_traces=0,
                    n_eligible_traces=0,
                    median_lambda_emp=float("nan"),
                    median_corr_a=float("nan"),
                    median_rmse_a=float("nan"),
                    median_mae_a=float("nan"),
                    median_mean_a_measured=float("nan"),
                    median_mean_a_surrogate=float("nan"),
                    measured_slope_low=float("nan"),
                    measured_slope_high=float("nan"),
                    measured_saturation_flag=False,
                    surrogate_slope_low=float("nan"),
                    surrogate_slope_high=float("nan"),
                    surrogate_saturation_flag=False,
                )
            )
            trace_rows_by_config[config_id] = []
            continue

        dt = float(dt_arr[0])
        if (not np.isfinite(dt)) or dt <= 0.0:
            raise ValueError(f"Invalid dt for {config_id}: {dt}")

        n_total = int(
            min(
                len(pair_key_arr), len(power_arr), len(power_start_arr), len(active_arr)
            )
        )
        config_rows: List[TraceRow] = []

        for idx in test_indices:
            if idx < 0 or idx >= n_total:
                config_rows.append(
                    _make_skipped_trace_row(
                        config_id=config_id,
                        trace_index=int(idx),
                        pair_key="",
                        rate_label="",
                        dt=float(dt),
                        reason="test_index_out_of_range",
                    )
                )
                continue

            pair_key = str(pair_key_arr[idx])
            rate_label = str(rate_arr[idx]) if idx < len(rate_arr) else ""
            request_json = pair_map.get(pair_key)
            if request_json is None:
                config_rows.append(
                    _make_skipped_trace_row(
                        config_id=config_id,
                        trace_index=int(idx),
                        pair_key=pair_key,
                        rate_label=rate_label,
                        dt=float(dt),
                        reason="missing_pair_manifest_json_path",
                    )
                )
                continue

            power = np.asarray(power_arr[idx], dtype=np.float64).reshape(-1)
            active = np.asarray(active_arr[idx], dtype=np.float64).reshape(-1)
            if power.size < 2 or active.size < 2:
                config_rows.append(
                    _make_skipped_trace_row(
                        config_id=config_id,
                        trace_index=int(idx),
                        pair_key=pair_key,
                        rate_label=rate_label,
                        dt=float(dt),
                        reason="empty_power_or_active_trace",
                    )
                )
                continue

            if not np.all(np.isfinite(power)) or not np.all(np.isfinite(active)):
                config_rows.append(
                    _make_skipped_trace_row(
                        config_id=config_id,
                        trace_index=int(idx),
                        pair_key=pair_key,
                        rate_label=rate_label,
                        dt=float(dt),
                        reason="non_finite_power_or_active_trace",
                    )
                )
                continue

            T = int(max(0, len(power) - 1))
            if T <= 0:
                config_rows.append(
                    _make_skipped_trace_row(
                        config_id=config_id,
                        trace_index=int(idx),
                        pair_key=pair_key,
                        rate_label=rate_label,
                        dt=float(dt),
                        reason="zero_horizon",
                    )
                )
                continue

            requests, req_reason = _build_requests_from_stage0_json(
                request_json,
                power_start_epoch_s=float(power_start_arr[idx]),
                trace_duration_s=float(len(power) * dt),
                dt=float(dt),
            )
            if len(requests) == 0:
                config_rows.append(
                    _make_skipped_trace_row(
                        config_id=config_id,
                        trace_index=int(idx),
                        pair_key=pair_key,
                        rate_label=rate_label,
                        dt=float(dt),
                        reason=req_reason,
                    )
                )
                continue

            A_sur = _reconstruct_surrogate_A_t(
                requests,
                lambda_prefill=float(throughput["lambda_prefill"]),
                lambda_decode=float(throughput["lambda_decode"]),
                dt=float(dt),
                T=int(T),
            )
            A_meas = _extract_measured_A_t(active, T)

            row = _build_trace_row(
                config_id=config_id,
                trace_index=int(idx),
                pair_key=pair_key,
                rate_label=rate_label,
                dt=float(dt),
                T=int(T),
                n_requests=int(len(requests)),
                A_meas=A_meas,
                A_sur=A_sur,
            )
            config_rows.append(row)
            if row.status == "evaluated":
                trace_arrays[(config_id, int(idx))] = (
                    np.asarray(A_meas[: row.n_timesteps], dtype=np.float64),
                    np.asarray(A_sur[: row.n_timesteps], dtype=np.float64),
                )

        trace_rows.extend(config_rows)
        trace_rows_by_config[config_id] = config_rows
        summary_rows.append(
            _summarize_config(config_id, config_rows, n_test_traces=len(test_indices))
        )

    selected_cfgs, selection_notes = _select_representative_configs(
        summaries=summary_rows,
        num_representative_configs=int(num_representative_configs),
        min_eligible_traces=int(min_eligible_traces),
        stable_corr_threshold=float(stable_corr_threshold),
    )

    selected_ids = [s.config_id for s in selected_cfgs]
    summary_by_config = {s.config_id: s for s in summary_rows}

    selected_rows: List[Tuple[str, TraceRow]] = []
    for sc in selected_cfgs:
        cfg_rows = trace_rows_by_config.get(sc.config_id, [])
        summary = summary_by_config.get(sc.config_id)
        if summary is None:
            continue
        chosen = _choose_overlay_trace(cfg_rows, summary)
        if chosen is None:
            continue
        selected_rows.append((sc.config_id, chosen))

    per_config_figures = _build_per_config_figure_paths(
        out_figure_overlays=out_figure_overlays,
        selected_rows=selected_rows,
    )

    if not bool(dry_run):
        for row in per_config_figures:
            cfg = str(row["config_id"])
            trace_idx = int(row["trace_index"])
            arrs = trace_arrays.get((cfg, trace_idx))
            if arrs is None:
                continue
            _plot_at_time_series(
                out_path=str(row["timeseries_file"]),
                A_meas=arrs[0],
                A_sur=arrs[1],
                dt=float(
                    next(
                        r.dt
                        for r in trace_rows_by_config[cfg]
                        if r.trace_index == trace_idx
                    )
                ),
                time_window_s=float(time_window_s),
            )
            _plot_at_histogram(
                out_path=str(row["histogram_file"]),
                A_meas=arrs[0],
                A_sur=arrs[1],
            )

        if len(selected_ids) > 0:
            _plot_lambda_vs_mean_at(
                out_path=out_figure_scatter,
                selected_config_ids=selected_ids,
                trace_rows_by_config=trace_rows_by_config,
                summary_by_config=summary_by_config,
            )

        trace_fieldnames = [
            "config_id",
            "trace_index",
            "pair_key",
            "rate_label",
            "dt",
            "n_timesteps",
            "n_requests",
            "lambda_emp",
            "mean_a_measured",
            "p95_a_measured",
            "p99_a_measured",
            "mean_a_surrogate",
            "p95_a_surrogate",
            "p99_a_surrogate",
            "corr_a",
            "mae_a",
            "rmse_a",
            "status",
            "reason",
        ]
        _write_csv(
            out_csv, [_trace_row_to_dict(r) for r in trace_rows], trace_fieldnames
        )

        summary_fieldnames = [
            "config_id",
            "n_test_traces",
            "n_eligible_traces",
            "median_lambda_emp",
            "median_corr_a",
            "median_rmse_a",
            "median_mae_a",
            "median_mean_a_measured",
            "median_mean_a_surrogate",
            "measured_slope_low",
            "measured_slope_high",
            "measured_saturation_flag",
            "surrogate_slope_low",
            "surrogate_slope_high",
            "surrogate_saturation_flag",
        ]
        _write_csv(
            out_summary_csv,
            [_config_summary_to_dict(s) for s in summary_rows],
            summary_fieldnames,
        )

    aggregate_all = _aggregate_surrogate_quality(trace_rows)
    aggregate_selected = _aggregate_surrogate_quality(
        [r for r in trace_rows if r.config_id in set(selected_ids)]
    )

    manifest = {
        "schema_version": "appendix-a1-surrogate-validity-v2",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "run_manifest": str(Path(run_manifest).resolve()),
            "experimental_manifest": str(Path(experimental_manifest).resolve()),
            "pair_manifest_csv": str(Path(pair_manifest_csv).resolve()),
            "throughput_db": str(Path(throughput_db).resolve()),
            "config_pool": str(config_pool),
        },
        "selection": {
            "num_representative_configs": int(num_representative_configs),
            "min_eligible_traces": int(min_eligible_traces),
            "stable_corr_threshold": float(stable_corr_threshold),
            "selected_configs": [
                {
                    "bucket": s.bucket,
                    "corr_threshold_used": (
                        float(s.corr_threshold_used)
                        if np.isfinite(s.corr_threshold_used)
                        else None
                    ),
                    "config_id": s.config_id,
                }
                for s in selected_cfgs
            ],
            "notes": selection_notes,
        },
        "assumptions": {
            "measured_A_t_source": "experimental dataset npz active_requests",
            "surrogate_A_t": "arrival + throughput-based service interval accumulation",
            "queue_analysis": "omitted",
            "arrival_rate_definition": "lambda_emp = n_requests / (T*dt) from aligned recorded arrivals",
            "lightweight_runtime": "no torch/sklearn imports",
        },
        "surrogate_quality": {
            "all_evaluated_traces": aggregate_all,
            "selected_configs_only": aggregate_selected,
        },
        "internal_checks": checks,
        "stats": {
            "num_target_configs": int(len(target_config_ids)),
            "num_trace_rows": int(len(trace_rows)),
            "num_summary_rows": int(len(summary_rows)),
            "num_selected_configs": int(len(selected_cfgs)),
            "num_per_config_figures": int(len(per_config_figures)),
        },
        "output_paths": {
            "trace_metrics_csv": str(out_csv),
            "config_summary_csv": str(out_summary_csv),
            "per_config_at_figures": per_config_figures,
            "scatter_figure_pdf": str(out_figure_scatter),
            "manifest_json": str(out_manifest_json),
        },
        "dry_run": bool(dry_run),
    }
    write_json(out_manifest_json, manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Appendix A1 surrogate-validity pipeline: A_t sanity checks and saturation diagnostics."
    )
    parser.add_argument(
        "--run-manifest",
        default="results/continuous_v1_gmm_bigru/k10_f2/run_manifest.json",
    )
    parser.add_argument(
        "--experimental-manifest",
        default="results/experimental_continuous_v1/manifest.json",
    )
    parser.add_argument(
        "--pair-manifest-csv", default="results/stage0/pair_manifest.csv"
    )
    parser.add_argument(
        "--throughput-db", default="model/config/throughput_database.json"
    )
    parser.add_argument(
        "--config-pool",
        default="all_trained",
        choices=["all_trained", "70b_tp4", "70b_all_tp"],
    )
    parser.add_argument("--num-representative-configs", type=int, default=3)
    parser.add_argument("--min-eligible-traces", type=int, default=2)
    parser.add_argument("--stable-corr-threshold", type=float, default=0.80)
    parser.add_argument("--time-window-s", type=float, default=600.0)
    parser.add_argument(
        "--out-csv", default="results/eval_paper/appendix_a1_trace_metrics.csv"
    )
    parser.add_argument(
        "--out-summary-csv", default="results/eval_paper/appendix_a1_config_summary.csv"
    )
    parser.add_argument(
        "--out-figure-overlays",
        default="figures/appendix_a1_at",
        help="Base path used to derive per-config A_t time-series/histogram filenames.",
    )
    parser.add_argument(
        "--out-figure-scatter", default="figures/appendix_a1_lambda_vs_mean_at.pdf"
    )
    parser.add_argument(
        "--out-manifest-json", default="results/eval_paper/appendix_a1_manifest.json"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run = run_appendix_surrogate_validity(
        run_manifest=str(args.run_manifest),
        experimental_manifest=str(args.experimental_manifest),
        pair_manifest_csv=str(args.pair_manifest_csv),
        throughput_db=str(args.throughput_db),
        config_pool=str(args.config_pool),
        num_representative_configs=int(args.num_representative_configs),
        min_eligible_traces=int(args.min_eligible_traces),
        stable_corr_threshold=float(args.stable_corr_threshold),
        time_window_s=float(args.time_window_s),
        out_csv=str(args.out_csv),
        out_summary_csv=str(args.out_summary_csv),
        out_figure_overlays=str(args.out_figure_overlays),
        out_figure_scatter=str(args.out_figure_scatter),
        out_manifest_json=str(args.out_manifest_json),
        dry_run=bool(args.dry_run),
    )

    if bool(args.dry_run):
        print("[appendix_surrogate_validity] Dry run complete")
    else:
        print("[appendix_surrogate_validity] Done")

    quality = run.get("surrogate_quality", {})
    quality_all = quality.get("all_evaluated_traces", {})
    print(f"  trace_csv   : {run['output_paths']['trace_metrics_csv']}")
    print(f"  summary_csv : {run['output_paths']['config_summary_csv']}")
    print(f"  scatter_pdf : {run['output_paths']['scatter_figure_pdf']}")
    print(f"  manifest    : {run['output_paths']['manifest_json']}")
    print(
        "  surrogate_quality_all: "
        f"n={quality_all.get('n_evaluated')}, "
        f"mean_corr={quality_all.get('mean_corr_a')}, "
        f"mean_mae={quality_all.get('mean_mae_a')}, "
        f"mean_rmse={quality_all.get('mean_rmse_a')}"
    )


if __name__ == "__main__":
    main()
