"""Score controlled-arrival and agentic bundles with source-only artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feature-test"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "power-test"))

from evaluation_core import trace_metrics  # noqa: E402
from clean_dense_surface import filter_design, raw_design  # noqa: E402
from fit_clean_power_pipelines import (  # noqa: E402
    DENSE_FEATURES,
    moe_compute_coordinate,
    moe_feature_basis,
    per_gpu,
)
from fit_power_surface import interpolate_nan  # noqa: E402
from iteration_time import launch_overhead_s, transformer_bw_scale  # noqa: E402
from moe_surface_core import surface_design as moe_design  # noqa: E402
from power_surface import predict, surface_design  # noqa: E402
from response_chain import apply_chain  # noqa: E402
from scheduler_sim import EngineConfig, simulate_requests  # noqa: E402
from simulated_ledger import emit_bins  # noqa: E402

MAX_TEMPORAL_MISSING_1S_FRACTION = 0.05
MAX_TEMPORAL_POWER_GAP_S = 2.0
from model.training_data.run_record import load_bundle_run  # noqa: E402


def engine_from_manifest(manifest: dict) -> EngineConfig:
    server = manifest.get("server") or {}
    required = ("max_num_seqs", "max_num_batched_tokens")
    missing = [key for key in required if key not in server]
    if missing:
        raise ValueError(f"Manifest server is missing engine bindings: {missing}")
    return EngineConfig(
        max_num_seqs=int(server["max_num_seqs"]),
        chunk_budget_tokens=int(server["max_num_batched_tokens"]),
        gpu_memory_utilization=float(server.get("gpu_memory_utilization", 0.9)),
    )


def _projected_request_column(record, name: str, default: float) -> np.ndarray:
    values = record.request_table.get(name)
    if values is None:
        return np.full(record.input_lens.size, default, dtype=float)
    indices = np.asarray(record.provenance["request_projection_indices"], dtype=int)
    return np.asarray(values, dtype=object)[indices].astype(float)


def request_schedule(record) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    cached = _projected_request_column(record, "cached_prompt_tokens", 0.0)
    executed = np.asarray(record.input_lens, int) - cached.astype(int)
    if np.any(cached < 0) or np.any(executed < 1):
        raise ValueError("Cached prompt accounting must leave at least one executed token")
    arrivals_epoch = np.asarray(record.request_timestamps, float)
    origin = float(np.min(arrivals_epoch))
    arrivals = arrivals_epoch - origin
    order = np.argsort(arrivals, kind="stable")
    requests = [
        (
            float(arrivals[i]), int(executed[i]), int(record.output_lens[i]),
            int(cached[i]),
        )
        for i in order
    ]
    return requests, order, np.asarray([origin])


def measured_horizon_s(record, origin: float) -> float:
    completion = (
        np.asarray(record.request_timestamps, float) - origin
        + np.asarray(record.ttfts, float)
        + np.asarray(record.decode_times, float)
    )
    return float(np.max(completion))


def _timing_summary(rows: list[dict]) -> dict:
    out = {"requests": len(rows)}
    for phase in ("ttft_s", "decode_s", "e2e_s"):
        measured = np.asarray([row[f"measured_{phase}"] for row in rows])
        predicted = np.asarray([row[f"predicted_{phase}"] for row in rows])
        valid = np.isfinite(measured) & np.isfinite(predicted) & (measured > 0.0)
        pct = 100.0 * (predicted[valid] - measured[valid]) / measured[valid]
        if pct.size == 0:
            out[f"{phase}_medabs_pct"] = float("nan")
            out[f"{phase}_p90abs_pct"] = float("nan")
            out[f"{phase}_median_signed_pct"] = float("nan")
            continue
        out[f"{phase}_medabs_pct"] = float(np.median(np.abs(pct)))
        out[f"{phase}_p90abs_pct"] = float(np.percentile(np.abs(pct), 90))
        out[f"{phase}_median_signed_pct"] = float(np.median(pct))
    return out


def _session_summaries(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["session_id"]:
            grouped[row["session_id"]].append(row)
    return [
        {"session_id": session, **_timing_summary(values)}
        for session, values in sorted(grouped.items())
        if len(values) > 1
    ]


def _measured_power_on_grid(record, origin: float, n: int, dt: float) -> np.ndarray:
    relative = np.asarray(record.power_timestamps, float) - origin
    bins = np.floor(relative / dt).astype(int)
    power = record.tp_sum_power()
    keep = (bins >= 0) & (bins < n) & np.isfinite(power)
    sums = np.bincount(bins[keep], weights=power[keep], minlength=n)
    counts = np.bincount(bins[keep], minlength=n)
    return np.divide(
        sums, counts, out=np.full(n, np.nan), where=counts > 0
    )


def power_coverage(measured: np.ndarray, dt: float) -> dict:
    """Coverage diagnostics before any interpolation of power telemetry."""
    measured = np.asarray(measured, float).reshape(-1)
    factor = int(round(1.0 / dt))
    if factor < 1 or not np.isclose(factor * dt, 1.0):
        raise ValueError("Power timestep must divide one second")
    n = measured.size // factor * factor
    observed_1s = (
        np.isfinite(measured[:n]).reshape(-1, factor).any(1)
        if n else np.asarray([], bool)
    )
    missing = ~np.isfinite(measured)
    edges = np.r_[0, np.flatnonzero(missing[1:] != missing[:-1]) + 1,
                  missing.size]
    maximum_gap_bins = max(
        (hi - lo for lo, hi in zip(edges[:-1], edges[1:]) if missing[lo]),
        default=0,
    )
    missing_fraction = (
        float(np.mean(~observed_1s)) if observed_1s.size else 1.0
    )
    return {
        "observed_native_fraction": float(np.mean(~missing)),
        "missing_one_second_fraction": missing_fraction,
        "maximum_power_gap_s": float(maximum_gap_bins * dt),
        "temporal_supported": bool(
            observed_1s.size >= 62
            and missing_fraction <= MAX_TEMPORAL_MISSING_1S_FRACTION
            and maximum_gap_bins * dt <= MAX_TEMPORAL_POWER_GAP_S
        ),
    }


def _ledger_design(record, bins: dict) -> dict:
    n = bins["n"]
    d = {
        key: np.asarray(value)
        for key, value in bins.items()
        if isinstance(value, np.ndarray)
    }
    d.update({
        "tp": np.full(n, record.tp, dtype=float),
        "n_active": np.full(n, float(record.arch["n_active"])),
        "w_bytes": np.full(n, float(record.arch["w_bytes"])),
        "fp8": np.full(n, float(record.arch.get("fp8", 0))),
        "fp8_flop_frac": np.full(
            n, float(record.arch.get(
                "fp8_flop_frac", 1.0 if record.arch.get("fp8", 0) else 0.0
            )),
        ),
        "run_id": np.zeros(n, dtype=np.int32),
    })
    return d


def _clean_prediction(record, d: dict, power_fit: dict, dt: float):
    family = str(record.arch["family"])
    if family.startswith("dense"):
        fit = power_fit["dense"][record.hardware]
        if fit["feature_names"] != list(DENSE_FEATURES):
            raise ValueError("Clean dense artifact feature contract mismatch")
        design = filter_design(
            raw_design(d, record.hardware), d["run_id"], dt,
            record.hardware, float(fit["delay_s"]),
        )
        prediction = design @ np.asarray(fit["coefficients"], float)
        return prediction * record.tp, "dense", True, None

    model_key = next(
        (key for key in power_fit["moe"]["per_model"]
         if key in record.model.lower()),
        None,
    )
    if model_key is None:
        reason = "No architecture-specific MoE coefficients for this checkpoint"
        return None, "moe", False, reason
    fit = power_fit["moe"]["per_model"][model_key]
    design_pg, _ = per_gpu(moe_design(d), np.zeros(d["run_id"].size), d["tp"])
    design_pg = np.insert(
        design_pg, 3, moe_compute_coordinate(d), axis=1
    )
    design_pg, names = moe_feature_basis(design_pg, d["tp"])
    if fit["feature_names"] != names:
        raise ValueError("Clean MoE artifact feature contract mismatch")
    prediction = design_pg @ np.asarray(fit["coefficients"], float)
    return prediction * record.tp, f"moe:{model_key}", True, None


def _power_summary(
    record, bins: dict, origin: float, power_fit: dict, dt: float
) -> dict:
    d = _ledger_design(record, bins)
    clean = power_fit.get("schema_version") == "clean-separated-power-surfaces-v4"
    if clean:
        predicted, surface, supported, reason = _clean_prediction(
            record, d, power_fit, dt
        )
    else:
        supported, reason, surface = True, None, "legacy-dense"
        hardware_fit = power_fit["per_hardware"][record.hardware]
        design, names = surface_design(d, record.hardware)
        coefficients = np.asarray([
            hardware_fit["coefficients"][name] for name in names
        ])
        raw = predict(design, coefficients, d["tp"], record.hardware)
        predicted = apply_chain(
            raw, dt, record.hardware, float(hardware_fit["delay_s"])
        )
    n = bins["n"]
    measured_raw = _measured_power_on_grid(record, origin, n, dt)
    valid = np.flatnonzero(np.isfinite(measured_raw))
    if valid.size == 0:
        raise ValueError("Bundle power does not overlap the simulated schedule")
    lo, hi = int(valid[0]), int(valid[-1] + 1)
    coverage = power_coverage(measured_raw[lo:hi], dt)
    measured, gaps = interpolate_nan(measured_raw[lo:hi])
    if not supported:
        return {
            "surface": surface,
            "surface_supported": False,
            "support_reason": reason,
            "duration_s": float((hi - lo) * dt),
            "power_bins": int(hi - lo),
            "interpolated_bins": int(gaps),
        }
    predicted = np.asarray(predicted)[lo:hi]
    busy = np.asarray(d["busy"], float)[lo:hi] > 0.0
    active = np.flatnonzero(busy)
    predicted_busy_end_s = (
        float((active[-1] + 1) * dt) if active.size else 0.0
    )
    signed = 100.0 * (float(np.mean(predicted)) - float(np.mean(measured))) \
        / float(np.mean(measured))
    temporal = (
        trace_metrics(measured, predicted, native_dt=dt)
        if coverage["temporal_supported"] else {}
    )
    return {
        "surface": surface,
        "surface_supported": True,
        "support_reason": None,
        "duration_s": float((hi - lo) * dt),
        "power_bins": int(hi - lo),
        "interpolated_bins": int(gaps),
        **coverage,
        "temporal_support_reason": (
            None if coverage["temporal_supported"] else
            "Power telemetry lacks 62 seconds with at least 95% one-second "
            "coverage and no gap longer than 2 seconds"
        ),
        "energy_error_pct": abs(signed),
        "mean_bias_pct": signed,
        "measured_mean_w": float(np.mean(measured)),
        "predicted_mean_w": float(np.mean(predicted)),
        "predicted_busy_fraction": float(np.mean(busy)),
        "predicted_busy_end_s": predicted_busy_end_s,
        "predicted_idle_tail_s": float((busy.size * dt) - predicted_busy_end_s),
        "acf_mae": float(temporal.get("acf_mae", float("nan"))),
        "acf_r2": float(temporal.get("acf_r2", float("nan"))),
        "soft_dtw_divergence": float(
            temporal.get("soft_dtw_divergence", float("nan"))
        ),
        "soft_dtw_diagonal_divergence": float(
            temporal.get("soft_dtw_diagonal_divergence", float("nan"))
        ),
        "soft_dtw_band_effect": float(
            temporal.get("soft_dtw_band_effect", float("nan"))
        ),
        "soft_dtw_band_effect_fraction": float(
            temporal.get("soft_dtw_band_effect_fraction", float("nan"))
        ),
        "nrmse_range": float(temporal.get("nrmse_range", float("nan"))),
    }


def evaluate_bundle(
    run_dir: Path, timing_fit: dict, power_fit: dict, *, dt: float
) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    record = load_bundle_run(run_dir)
    params = timing_fit[record.hardware]
    engine = engine_from_manifest(manifest)
    requests, order, origin_array = request_schedule(record)
    trace = []
    simulated = simulate_requests(
        requests, arch=record.arch, hardware=record.hardware, tp=record.tp,
        eff_flops=float(params["eff_flops"]),
        eff_bw=float(params["eff_bw"]),
        transformer_bw_scale=transformer_bw_scale(
            record.arch, params, record.hardware
        ),
        t_launch_s=launch_overhead_s(
            record.arch, base_s=float(params["base_overhead_s"]),
            per_message_s=float(params["per_message_s"][str(record.tp)]),
        ),
        t_sample_s=float(params.get("per_token_sample_s", 0.0)),
        engine=engine, iteration_trace=trace,
    )
    session_values = (
        np.asarray(record.request_table["session_ids"], dtype=object)[
            np.asarray(record.provenance["request_projection_indices"], dtype=int)
        ]
        if "session_ids" in record.request_table
        else np.full(record.input_lens.size, "", dtype=object)
    )
    rows = []
    first_token = float(params["first_token_overhead_s"])
    for sim, index in zip(simulated, order):
        measured_ttft = float(record.ttfts[index])
        measured_decode = float(record.decode_times[index])
        rows.append({
            "session_id": str(session_values[index]),
            "measured_ttft_s": measured_ttft,
            "predicted_ttft_s": sim["ttft_s"] + first_token,
            "measured_decode_s": measured_decode,
            "predicted_decode_s": sim["decode_duration_s"],
            "measured_e2e_s": measured_ttft + measured_decode,
            "predicted_e2e_s": sim["e2e_s"] + first_token,
        })
    origin = float(origin_array[0])
    bins = emit_bins(
        trace, simulated, arch=record.arch, tp=record.tp, dt=dt,
        horizon_s=measured_horizon_s(record, origin),
    )
    campaign = run_dir.parent.name
    return {
        "campaign": campaign,
        "run_id": manifest["run_id"],
        "hardware": record.hardware,
        "model": record.model,
        "probe_type": (manifest.get("probe") or {}).get("type"),
        "prefix_cache": bool((manifest.get("probe") or {}).get("prefix_cache", False)),
        "engine": {
            "max_num_seqs": engine.max_num_seqs,
            "max_num_batched_tokens": engine.chunk_budget_tokens,
            "kv_cache_dtype": (manifest.get("server") or {}).get("kv_cache_dtype"),
            "vllm": (manifest.get("versions") or {}).get("vllm"),
        },
        "timing": _timing_summary(rows),
        "session_timing": _session_summaries(rows),
        "power": _power_summary(
            record, bins, origin, power_fit, dt
        ),
    }


def development_run_dirs(root: Path) -> list[Path]:
    output = []
    for path in sorted(root.glob("*/*/manifest.json")):
        manifest = json.loads(path.read_text())
        probe = manifest.get("probe") or {}
        role = manifest.get("validation_role") or probe.get("validation_role")
        campaign = path.parents[1].name
        if role == "sealed" or "smoke" in campaign:
            continue
        if role == "development" or probe.get("type") == "trace_replay":
            output.append(path.parent)
    return output


def replay_pair_identity(off_dir: Path, on_dir: Path) -> dict:
    off = json.loads((off_dir / "requests.json").read_text())
    on = json.loads((on_dir / "requests.json").read_text())
    off_rows = {
        (session, turn): i
        for i, (session, turn) in enumerate(zip(off["session_ids"], off["turn_idx"]))
    }
    on_rows = {
        (session, turn): i
        for i, (session, turn) in enumerate(zip(on["session_ids"], on["turn_idx"]))
    }
    shared = sorted(set(off_rows) & set(on_rows))
    fields = (
        "input_lens", "output_lens", "prefix_tokens", "new_input_tokens",
        "planned_output_tokens", "prompt_sha256", "output_sha256",
    )
    mismatches = {
        field: sum(
            off[field][off_rows[key]] != on[field][on_rows[key]]
            for key in shared
        )
        for field in fields
    }
    return {
        "cache_pair_valid": (
            len(shared) == len(off_rows) == len(on_rows)
            and not any(mismatches.values())
        ),
        "cache_off_rows": len(off_rows),
        "cache_on_rows": len(on_rows),
        "shared_rows": len(shared),
        "mismatches": mismatches,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-fit", required=True)
    parser.add_argument("--power-fit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dt", type=float, default=0.25)
    args = parser.parse_args(argv)
    timing_fit = json.loads(Path(args.timing_fit).read_text())
    power_fit = json.loads(Path(args.power_fit).read_text())
    run_dirs = development_run_dirs(Path("data/runs"))
    runs = [
        evaluate_bundle(run_dir, timing_fit, power_fit, dt=args.dt)
        for run_dir in run_dirs
    ]
    off = next(path for path in run_dirs if "cache_off" in str(path))
    on = next(path for path in run_dirs if "cache_on" in str(path))
    report = {
        "schema_version": "expansion-development-score-v2",
        "evidence_role": "retrospective development",
        "timing_fit": args.timing_fit,
        "power_fit": args.power_fit,
        "runs": runs,
        "agentic_cache_pair_identity": replay_pair_identity(off, on),
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"scored {len(runs)} development bundles -> {args.out}")


if __name__ == "__main__":
    main()
