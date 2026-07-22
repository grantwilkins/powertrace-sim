"""Shared request-to-power primitives for retrospective transfer panels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model.power.predictor import dense_design, moe_design
from model.power.response import apply_response
from model.timing.iteration import HARDWARE_PROFILES, launch_overhead_s, transformer_bw_scale
from model.timing.ledger import emit_bins
from model.timing.scheduler import EngineConfig, simulate_requests
from model.training_data.run_record import load_bundle_run

DT_S = 0.25


def _engine(manifest: dict) -> EngineConfig:
    server = manifest["server"]
    return EngineConfig(
        max_num_seqs=int(server["max_num_seqs"]),
        chunk_budget_tokens=int(server["max_num_batched_tokens"]),
        gpu_memory_utilization=float(server.get("gpu_memory_utilization", 0.9)),
    )


def _schedule(record) -> tuple[list[tuple], np.ndarray, float]:
    cached_raw = record.request_table.get("cached_prompt_tokens")
    if cached_raw is None:
        cached = np.zeros(record.input_lens.size, dtype=int)
    else:
        indices = np.asarray(record.provenance["request_projection_indices"], dtype=int)
        cached = np.asarray(cached_raw, dtype=object)[indices].astype(int)
    executed = np.asarray(record.input_lens, int) - cached
    if np.any(cached < 0) or np.any(executed < 1):
        raise ValueError("cached prompts must leave at least one executed token")
    origin = float(np.min(record.request_timestamps))
    arrivals = np.asarray(record.request_timestamps, float) - origin
    order = np.argsort(arrivals, kind="stable")
    requests = [
        (float(arrivals[i]), int(executed[i]), int(record.output_lens[i]), int(cached[i]))
        for i in order
    ]
    return requests, order, origin


def _timing_summary(rows: list[dict]) -> dict:
    report = {"requests": len(rows)}
    for phase in ("ttft_s", "decode_s", "e2e_s"):
        measured = np.asarray([row[f"measured_{phase}"] for row in rows])
        predicted = np.asarray([row[f"predicted_{phase}"] for row in rows])
        valid = np.isfinite(measured) & np.isfinite(predicted) & (measured > 0.0)
        error = 100.0 * (predicted[valid] - measured[valid]) / measured[valid]
        if error.size == 0:
            raise ValueError(f"no valid measured {phase} values")
        report[f"{phase}_medabs_pct"] = float(np.median(np.abs(error)))
        report[f"{phase}_p90abs_pct"] = float(np.percentile(np.abs(error), 90))
        report[f"{phase}_median_signed_pct"] = float(np.median(error))
    return report


def _measured_power(record, origin: float, count: int) -> np.ndarray:
    bins = np.floor((np.asarray(record.power_timestamps) - origin) / DT_S).astype(int)
    values = record.tp_sum_power()
    keep = (bins >= 0) & (bins < count) & np.isfinite(values)
    sums = np.bincount(bins[keep], weights=values[keep], minlength=count)
    counts = np.bincount(bins[keep], minlength=count)
    return np.divide(sums, counts, out=np.full(count, np.nan), where=counts > 0)


def _interpolate(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    missing = ~np.isfinite(values)
    if missing.all():
        raise ValueError("trace contains no measured power")
    output = values.copy()
    observed = np.flatnonzero(~missing)
    output[missing] = np.interp(np.flatnonzero(missing), observed, values[observed])
    return output, int(missing.sum())


def request_ledger(bundle: Path, timing_fit: dict):
    manifest = json.loads((bundle / "manifest.json").read_text())
    record = load_bundle_run(bundle)
    params = timing_fit[record.hardware]
    requests, order, origin = _schedule(record)
    trace = []
    timed = simulate_requests(
        requests, arch=record.arch, hardware=record.hardware, tp=record.tp,
        eff_flops=float(params["eff_flops"]), eff_bw=float(params["eff_bw"]),
        transformer_bw_scale=transformer_bw_scale(record.arch, params, record.hardware),
        t_launch_s=launch_overhead_s(
            record.arch, base_s=float(params["base_overhead_s"]),
            per_message_s=float(params["per_message_s"][str(record.tp)]),
        ),
        t_sample_s=float(params.get("per_token_sample_s", 0.0)),
        engine=_engine(manifest), iteration_trace=trace,
    )
    completion = (
        np.asarray(record.request_timestamps) - origin
        + np.asarray(record.ttfts) + np.asarray(record.decode_times)
    )
    bins = emit_bins(
        trace, timed, arch=record.arch, tp=record.tp, dt=DT_S,
        horizon_s=float(np.max(completion)),
    )
    ledger = {
        key: np.asarray(value) for key, value in bins.items()
        if isinstance(value, np.ndarray)
    }
    count = int(bins["n"])
    ledger.update({
        "tp": np.full(count, record.tp, dtype=float),
        "run_id": np.zeros(count, dtype=int),
    })
    measured_raw = _measured_power(record, origin, count)
    valid = np.flatnonzero(np.isfinite(measured_raw))
    lo, hi = int(valid[0]), int(valid[-1] + 1)
    measured, gaps = _interpolate(measured_raw[lo:hi])
    first = float(params["first_token_overhead_s"])
    timing_rows = []
    for result, index in zip(timed, order):
        measured_ttft = float(record.ttfts[index])
        measured_decode = float(record.decode_times[index])
        timing_rows.append({
            "measured_ttft_s": measured_ttft,
            "predicted_ttft_s": result["ttft_s"] + first,
            "measured_decode_s": measured_decode,
            "predicted_decode_s": result["decode_duration_s"],
            "measured_e2e_s": measured_ttft + measured_decode,
            "predicted_e2e_s": result["e2e_s"] + first,
        })
    return manifest, record, ledger, measured, lo, hi, gaps, _timing_summary(timing_rows)


def target_idle(record, manifest: dict) -> tuple[float, list[float], int]:
    window = manifest["probe"]["idle_window"]
    if float(window["end_epoch"]) > float(np.min(record.request_timestamps)):
        raise ValueError("idle calibration overlaps request arrivals")
    selected = (
        (record.power_timestamps >= float(window["start_epoch"]))
        & (record.power_timestamps < float(window["end_epoch"]))
    )
    per_device = np.median(record.power_per_gpu[selected], axis=0)
    return float(np.median(per_device)), per_device.tolist(), int(selected.sum())


def dense_response(record, ledger: dict, fit: dict) -> np.ndarray:
    raw = dense_design(
        ledger, arch=record.arch, hardware=record.hardware, tp=record.tp
    )
    return apply_response(
        raw, dt_s=DT_S, hardware=record.hardware, delay_s=float(fit["delay_s"])
    )


def phase_compute(ledger: dict, hardware: str, tp: int) -> np.ndarray:
    profile = HARDWARE_PROFILES[hardware]
    return (
        np.asarray(ledger["prefill_gemm_flops_rate"])
        + np.asarray(ledger["prefill_attn_flops_rate"])
    ) / (tp * profile["peak_flops_s"])


def source_moe_design(ledger: dict, fit: dict, *, hardware: str, tp: int) -> np.ndarray:
    return moe_design(
        ledger, hardware=hardware, tp=tp, feature_names=list(fit["feature_names"])
    )


def hardware_energy_ratios(power_fit: dict) -> dict[str, float]:
    dense = {
        hardware: dict(zip(fit["feature_names"], fit["coefficients"]))
        for hardware, fit in power_fit["dense"].items()
    }
    return {
        "logical_memory_util_lag_250ms": (
            dense["H100"]["duty_sqrt_memory_util"] / HARDWARE_PROFILES["H100"]["hbm_bytes_s"]
        ) / (dense["A100"]["duty_sqrt_memory_util"] / HARDWARE_PROFILES["A100"]["hbm_bytes_s"]),
        "duty_sqrt_exact_compute_util": (
            dense["H100"]["compute_util"] / HARDWARE_PROFILES["H100"]["peak_flops_s"]
        ) / (dense["A100"]["compute_util"] / HARDWARE_PROFILES["A100"]["peak_flops_s"]),
    }


def predict_moe(design, coefficients, *, tp: int, hardware: str) -> np.ndarray:
    raw = np.asarray(design) @ np.asarray(coefficients) * tp
    return apply_response(
        raw[:, None], dt_s=DT_S, hardware=hardware, delay_s=0.0
    )[:, 0]
