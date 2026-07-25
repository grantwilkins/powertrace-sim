"""Measurement and metric helpers for disaggregated power evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation_core import normalized_soft_dtw_diagnostics, trace_metrics
from model.power import predict_power
from model.simulation import prepare_simulation
from model.timing.ledger import NATIVE_DT_S, emit_bins
from model.training_data.power_parsing import parse_power_csv_per_gpu

TEMPORAL_FIELDS = (
    "acf_mae",
    "acf_r2",
    "soft_dtw_divergence",
    "soft_dtw_diagonal_divergence",
    "soft_dtw_band_effect",
    "soft_dtw_band_effect_fraction",
    "nrmse_range",
)


def cell_key(path: Path) -> tuple[float, int]:
    _, rate, _, repeat = path.name.split("-")
    return float(rate.replace("p", ".")), int(repeat)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_sha256(run_root: Path) -> tuple[str, int]:
    files = [run_root / "run_metadata.json", run_root / "proxy_events.jsonl"]
    for cell in sorted(run_root.glob("rate-*")):
        files.extend(
            cell / name
            for name in (
                "requests.json",
                "power.csv",
                "engine_prefill.csv",
                "engine_decode.csv",
                "start_epoch_s",
                "end_epoch_s",
                "workload_start_epoch_s",
                "workload_end_epoch_s",
            )
            if (cell / name).is_file()
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(run_root)).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest(), len(files)


def load_events(path: Path) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        output.setdefault(row["request_id"], {})[row["event"]] = (
            int(row["wall_ns"]) / 1e9
        )
    return output


def load_power(
    cell: Path, metadata: dict
) -> tuple[np.ndarray, np.ndarray]:
    parsed = parse_power_csv_per_gpu(
        str(cell / "power.csv"),
        gpus_per_node=2,
        strict_topology=True,
        local_utc_offset_s=float(metadata["clock"]["local_utc_offset_s"]),
    )
    if parsed is None:
        raise ValueError(f"{cell}/power.csv is not a two-GPU raw trace")
    order = [
        parsed["device_ids"].index(metadata["roles"][role]["gpu_uuid"])
        for role in ("prefill", "decode")
    ]
    return (
        np.asarray(parsed["timestamps"], dtype=float),
        np.asarray(parsed["power_per_gpu"], dtype=float)[:, order],
    )


def idle_calibration(
    run_root: Path, metadata: dict, calibration_cell: str
) -> dict[str, object]:
    cell = run_root / calibration_cell
    request = json.loads((cell / "requests.json").read_text())
    start = float((cell / "start_epoch_s").read_text())
    timestamps, power = load_power(cell, metadata)
    window = (start + 2.0, min(request["request_timestamps"]) - 1.0)
    selected = (timestamps >= window[0]) & (timestamps < window[1])
    if selected.sum() < 8:
        raise ValueError("idle calibration window has too few settled samples")
    return {
        "cell": calibration_cell,
        "window_start_epoch_s": window[0],
        "window_end_epoch_s": window[1],
        "target_idle_w_per_gpu": float(np.median(power[selected])),
        "device_medians_w": np.median(power[selected], axis=0).tolist(),
        "timestamp_samples": int(selected.sum()),
        "pooled_gpu_samples": int(power[selected].size),
    }


def measured_grid(
    timestamps: np.ndarray,
    power: np.ndarray,
    start: float,
    count: int,
) -> np.ndarray:
    bins = np.floor((timestamps - start) / NATIVE_DT_S).astype(int)
    output = []
    for values in power.T:
        keep = (bins >= 0) & (bins < count) & np.isfinite(values)
        sums = np.bincount(bins[keep], weights=values[keep], minlength=count)
        samples = np.bincount(bins[keep], minlength=count)
        output.append(
            np.divide(
                sums,
                samples,
                out=np.full(count, np.nan),
                where=samples > 0,
            )
        )
    return np.column_stack(output)


def _coverage(values: np.ndarray) -> dict[str, object]:
    factor = int(round(1.0 / NATIVE_DT_S))
    n = values.size // factor * factor
    observed_1s = np.isfinite(values[:n]).reshape(-1, factor).any(axis=1)
    missing = ~np.isfinite(values)
    edges = np.r_[0, np.flatnonzero(missing[1:] != missing[:-1]) + 1, missing.size]
    maximum_gap_bins = max(
        (hi - lo for lo, hi in zip(edges[:-1], edges[1:]) if missing[lo]),
        default=0,
    )
    missing_1s = float(np.mean(~observed_1s)) if observed_1s.size else 1.0
    return {
        "observed_native_fraction": float(np.mean(~missing)),
        "missing_one_second_fraction": missing_1s,
        "maximum_power_gap_s": float(maximum_gap_bins * NATIVE_DT_S),
        "temporal_supported": bool(
            observed_1s.size >= 62
            and missing_1s <= 0.05
            and maximum_gap_bins * NATIVE_DT_S <= 2.0
        ),
    }


def align_power(
    measured: np.ndarray, predictions: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray], int, dict[str, object]]:
    valid = np.flatnonzero(np.isfinite(measured))
    if valid.size == 0:
        raise ValueError("power trace does not overlap the campaign window")
    lo, hi = int(valid[0]), int(valid[-1] + 1)
    measured = measured[lo:hi].copy()
    coverage = _coverage(measured)
    missing = ~np.isfinite(measured)
    measured[missing] = np.interp(
        np.flatnonzero(missing),
        np.flatnonzero(~missing),
        measured[~missing],
    )
    return (
        measured,
        {name: np.asarray(values)[lo:hi] for name, values in predictions.items()},
        int(missing.sum()),
        coverage,
    )


def power_metrics(
    measured: np.ndarray,
    predicted: np.ndarray,
    temporal_supported: bool,
) -> dict[str, float]:
    signed = (
        100.0
        * (float(predicted.mean()) - float(measured.mean()))
        / float(measured.mean())
    )
    temporal = (
        trace_metrics(measured, predicted, native_dt=NATIVE_DT_S)
        if temporal_supported
        else {}
    )
    return {
        "energy_error_pct": abs(signed),
        "mean_bias_pct": signed,
        "measured_mean_w": float(measured.mean()),
        "predicted_mean_w": float(predicted.mean()),
        **{
            field: float(temporal.get(field, float("nan")))
            for field in TEMPORAL_FIELDS
        },
    }


def role_trace_metrics(
    measured: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if (
        measured.ndim != 1
        or predicted.shape != measured.shape
        or measured.size < 2
        or not np.isfinite(measured).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("role traces must be aligned finite vectors")
    measured_std = float(measured.std())
    predicted_std = float(predicted.std())
    correlation = (
        float(np.corrcoef(measured, predicted)[0, 1])
        if measured_std > 0.0 and predicted_std > 0.0
        else float("nan")
    )
    measured_p95 = float(np.percentile(measured, 95))
    predicted_p95 = float(np.percentile(predicted, 95))
    diagonal = normalized_soft_dtw_diagnostics(measured, predicted, band=0)
    one_sample_band = normalized_soft_dtw_diagnostics(
        measured, predicted, band=1
    )
    constant = normalized_soft_dtw_diagnostics(
        measured, np.full_like(measured, measured.mean()), band=0
    )
    return {
        "correlation": correlation,
        "rmse_w": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "std_ratio": predicted_std / measured_std if measured_std > 0.0 else float("nan"),
        "mean_bias_w": float(predicted.mean() - measured.mean()),
        "measured_p95_w": measured_p95,
        "predicted_p95_w": predicted_p95,
        "p95_error_pct": 100.0 * abs(predicted_p95 - measured_p95) / measured_p95,
        "soft_dtw_diagonal_divergence": diagonal["soft_dtw_divergence"],
        "soft_dtw_one_sample_divergence": one_sample_band[
            "soft_dtw_divergence"
        ],
        "soft_dtw_one_sample_band_effect_fraction": one_sample_band[
            "soft_dtw_band_effect_fraction"
        ],
        "constant_mean_soft_dtw_diagonal_divergence": constant[
            "soft_dtw_divergence"
        ],
    }


def apply_phase_calibration(
    predicted: np.ndarray, *,
    source_idle_w: float,
    target_idle_w: float,
    dynamic_scale: float,
) -> np.ndarray:
    values = np.asarray(predicted, dtype=float)
    scalars = np.asarray(
        [source_idle_w, target_idle_w, dynamic_scale], dtype=float
    )
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("phase power must be a finite vector")
    if not np.isfinite(scalars).all() or min(scalars) <= 0.0:
        raise ValueError("phase calibration scalars must be positive and finite")
    return target_idle_w + dynamic_scale * (values - source_idle_w)


def fit_diagonal_soft_dtw_scale(
    measured: np.ndarray,
    predicted: np.ndarray,
    *,
    measured_idle_w: float,
    predicted_idle_w: float,
    bounds: tuple[float, float] = (0.5, 1.5),
) -> float:
    """Fit one dynamic gain to diagonal soft-DTW with fixed idle baselines."""
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if (
        measured.ndim != 1
        or predicted.shape != measured.shape
        or measured.size == 0
        or not np.isfinite(measured).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("phase calibration traces must be aligned finite vectors")
    lower, upper = map(float, bounds)
    if not 0.0 < lower <= upper:
        raise ValueError("dynamic-scale bounds must be positive and ordered")
    dynamic = predicted - float(predicted_idle_w)
    target = measured - float(measured_idle_w)
    denominator = float(dynamic @ dynamic)
    if denominator <= 0.0:
        raise ValueError("predicted phase trace has no dynamic power")
    optimum = float(dynamic @ target) / denominator
    return float(np.clip(optimum, lower, upper))


def _phase_metrics(
    measured: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    percent = 100.0 * (predicted - measured) / measured
    return {
        "medabs_pct": float(np.median(np.abs(percent))),
        "p90abs_pct": float(np.percentile(np.abs(percent), 90)),
        "median_signed_pct": float(np.median(percent)),
    }


def timing_metrics(
    request: dict,
    predicted: list[dict[str, object]],
    event_table: dict[str, dict[str, float]],
) -> dict[str, float]:
    events = [event_table[request_id] for request_id in request["request_ids"]]
    measured = {
        "prefill": np.asarray(
            [row["prefill_completed"] - row["prefill_sent"] for row in events]
        ),
        "decode_ttft": np.asarray(
            [row["decode_first_byte"] - row["decode_sent"] for row in events]
        ),
        "decode_duration": np.asarray(
            [row["decode_completed"] - row["decode_first_byte"] for row in events]
        ),
        "e2e": np.asarray(
            [row["decode_completed"] - row["proxy_received"] for row in events]
        ),
    }
    modeled = {
        "prefill": np.asarray([row["prefill_s"] for row in predicted]),
        "decode_ttft": np.asarray([row["decode_ttft_s"] for row in predicted]),
        "decode_duration": np.asarray(
            [row["decode_duration_s"] for row in predicted]
        ),
        "e2e": np.asarray([row["e2e_s"] for row in predicted]),
    }
    result = {
        f"{phase}_{metric}": value
        for phase in measured
        for metric, value in _phase_metrics(
            measured[phase], modeled[phase]
        ).items()
    }
    for phase in measured:
        result[f"{phase}_measured_median_ms"] = float(
            1000.0 * np.median(measured[phase])
        )
        result[f"{phase}_predicted_median_ms"] = float(
            1000.0 * np.median(modeled[phase])
        )
    request_epoch = np.asarray(request["request_timestamps"], dtype=float)
    client_ttft = np.asarray(request["ttfts"], dtype=float)
    proxy_received = np.asarray([row["proxy_received"] for row in events])
    first_byte = np.asarray([row["decode_first_byte"] for row in events])
    result.update({
        "client_to_proxy_median_ms": float(
            1000.0 * np.median(proxy_received - request_epoch)
        ),
        "prefill_dispatch_median_ms": float(1000.0 * np.median([
            row["prefill_sent"] - row["proxy_received"] for row in events
        ])),
        "handoff_median_ms": float(1000.0 * np.median([
            row["decode_sent"] - row["prefill_completed"] for row in events
        ])),
        "client_ttft_event_abs_median_ms": float(
            1000.0 * np.median(np.abs(
                client_ttft - (first_byte - request_epoch)
            ))
        ),
    })
    return result


def _counter_delta(path: Path, name: str) -> float:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = np.asarray([float(row[name]) for row in rows])
    return float(values[-1] - values[0])


def telemetry(cell: Path, power_timestamps: np.ndarray) -> dict[str, float]:
    prefill = cell / "engine_prefill.csv"
    decode = cell / "engine_decode.csv"
    count = _counter_delta(decode, "nixl_xfer_time_seconds_count")
    byte_count = _counter_delta(decode, "nixl_bytes_transferred_count")
    cache_queries = _counter_delta(prefill, "prefix_cache_queries_total")
    cache_hits = _counter_delta(prefill, "prefix_cache_hits_total")
    return {
        "prefill_prefix_cache_hit_fraction": cache_hits / cache_queries,
        "nixl_transfers": count,
        "nixl_mean_xfer_ms": (
            1000.0 * _counter_delta(decode, "nixl_xfer_time_seconds_sum") / count
        ),
        "nixl_mean_post_ms": (
            1000.0
            * _counter_delta(decode, "nixl_post_time_seconds_sum")
            / _counter_delta(decode, "nixl_post_time_seconds_count")
        ),
        "nixl_mean_transfer_mb": (
            _counter_delta(decode, "nixl_bytes_transferred_sum")
            / byte_count
            / 1e6
        ),
        "nixl_failed_transfers": _counter_delta(
            decode, "nixl_num_failed_transfers"
        ),
        "nixl_failed_notifications": _counter_delta(
            decode, "nixl_num_failed_notifications"
        ),
        "decode_preemptions": _counter_delta(decode, "num_preemptions_total"),
        "prefill_engine_mean_ms": (
            1000.0
            * _counter_delta(prefill, "request_prefill_time_seconds_sum")
            / count
        ),
        "decode_engine_mean_ms": (
            1000.0
            * _counter_delta(decode, "request_decode_time_seconds_sum")
            / count
        ),
        "maximum_sample_spacing_s": float(np.max(np.diff(power_timestamps))),
    }


def collocated_tp2(
    requests: list[dict[str, object]],
    horizon_s: float,
    release: dict,
) -> np.ndarray:
    prepared = prepare_simulation(
        requests,
        deployment={
            "preset": "gpt-oss-20b-a100-tp1",
            "overrides": {"tp": 2, "max_num_batched_tokens": 8192},
        },
        artifact=release,
        allow_unsupported=True,
    )
    ledger = emit_bins(
        prepared.trace,
        prepared.timed,
        arch=prepared.arch,
        tp=2,
        dt=NATIVE_DT_S,
        horizon_s=horizon_s,
    )
    return np.asarray(
        predict_power(
            ledger,
            arch=prepared.arch,
            model="gpt-oss-20b",
            hardware="A100",
            tp=2,
            artifact=release,
            dt_s=NATIVE_DT_S,
        )["node_gpu_power_w"]
    )


def summarize(
    rows: list[dict], candidates: tuple[str, ...]
) -> dict[str, dict]:
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    output = {}
    for candidate in candidates:
        selected = [row for row in evaluation if row["candidate"] == candidate]
        output[candidate] = {
            "cells": len(selected),
            "temporal_cells": sum(row["temporal_supported"] for row in selected),
            "energy_error_pct_median": float(
                np.median([row["energy_error_pct"] for row in selected])
            ),
            "energy_error_pct_worst": float(
                np.max([row["energy_error_pct"] for row in selected])
            ),
            "nrmse_range_median": float(
                np.nanmedian([row["nrmse_range"] for row in selected])
            ),
            "acf_r2_median": float(
                np.nanmedian([row["acf_r2"] for row in selected])
            ),
        }
    return output
