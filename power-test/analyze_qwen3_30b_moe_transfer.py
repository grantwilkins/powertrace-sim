"""Audit Qwen3-30B-A3B H100 transfer from the frozen source power laws."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path[:0] = [
    str(ROOT), str(ROOT / "timing-test"), str(ROOT / "feature-test"), str(BASE)
]

import evaluate_expansion as expansion  # noqa: E402
from clean_dense_surface import filter_design, raw_design  # noqa: E402
from evaluation_core import trace_metrics  # noqa: E402
from fit_clean_power_pipelines import (  # noqa: E402
    MOE_FEATURES,
    moe_compute_coordinate,
    per_gpu,
)
from fit_power_surface import interpolate_nan  # noqa: E402
from iteration_time import launch_overhead_s, transformer_bw_scale  # noqa: E402
from model.training_data.run_record import load_bundle_run  # noqa: E402
from moe_surface_core import surface_design as moe_design  # noqa: E402
from power_surface import HARDWARE  # noqa: E402
from response_chain import apply_chain  # noqa: E402
from scheduler_sim import simulate_requests  # noqa: E402
from simulated_ledger import emit_bins  # noqa: E402

DT_S = 0.25
BUNDLE = next((ROOT / "data/runs/sealed_qwen3-30b-a3b_h100").glob("*/manifest.json")).parent
TIMING_FIT = ROOT / "timing-test/fitted_efficiencies.json"
POWER_FIT = BASE / "clean_power_surfaces.json"
PLOT = BASE / "qwen3_30b_h100_moe_transfer.png"
REPORT = BASE / "qwen3_30b_h100_moe_transfer.json"
TRACE_CSV = BASE / "qwen3_30b_h100_moe_transfer_1s.csv"


def target_idle_w_per_gpu(record, manifest: dict) -> tuple[float, list[float], int]:
    """Return the pre-declared pre-request idle median, never loaded power."""
    window = manifest["probe"]["idle_window"]
    if float(window["end_epoch"]) > float(np.min(record.request_timestamps)):
        raise ValueError("Idle calibration window overlaps request arrivals")
    selected = (
        (record.power_timestamps >= float(window["start_epoch"]))
        & (record.power_timestamps < float(window["end_epoch"]))
    )
    if not np.any(selected):
        raise ValueError("Idle calibration window has no power samples")
    per_device = np.median(record.power_per_gpu[selected], axis=0)
    return (
        float(np.median(record.power_per_gpu[selected])),
        per_device.tolist(),
        int(selected.sum()),
    )


def design_for_source_fit(ledger: dict, fit: dict) -> tuple[np.ndarray, list[str]]:
    """Build columns from the source artifact contract, not target TP support."""
    design, _ = per_gpu(
        moe_design(ledger), np.zeros(ledger["run_id"].size), ledger["tp"]
    )
    design = np.insert(design, 3, moe_compute_coordinate(ledger), axis=1)
    names = list(MOE_FEATURES)
    if "multi_gpu_floor" not in fit["feature_names"]:
        index = names.index("multi_gpu_floor")
        design = np.delete(design, index, axis=1)
        names.pop(index)
    if names != fit["feature_names"]:
        raise ValueError("Source MoE feature contract mismatch")
    return design, names


def replace_idle(coefficients, names: list[str], idle_w_per_gpu: float) -> np.ndarray:
    output = np.asarray(coefficients, float).copy()
    output[names.index("idle")] = idle_w_per_gpu
    return output


def hardware_energy_ratios(power_fit: dict) -> dict[str, float]:
    dense = {
        hardware: dict(zip(fit["feature_names"], fit["coefficients"]))
        for hardware, fit in power_fit["dense"].items()
    }
    return {
        "logical_memory_util_lag_250ms": (
            dense["H100"]["duty_sqrt_memory_util"]
            / HARDWARE["H100"]["hbm_bandwidth_bytes_s"]
        ) / (
            dense["A100"]["duty_sqrt_memory_util"]
            / HARDWARE["A100"]["hbm_bandwidth_bytes_s"]
        ),
        "duty_sqrt_exact_compute_util": (
            dense["H100"]["compute_util"]
            / HARDWARE["H100"]["compute_peak_flops_s"]
        ) / (
            dense["A100"]["compute_util"]
            / HARDWARE["A100"]["compute_peak_flops_s"]
        ),
    }


def physics_adapt(coefficients, names, idle_w_per_gpu, ratios) -> np.ndarray:
    output = replace_idle(coefficients, names, idle_w_per_gpu)
    for name, ratio in ratios.items():
        output[names.index(name)] *= ratio
    return output


def predict_moe_node(design, coefficients, tp: int, *, h100_meter=True):
    prediction = design @ np.asarray(coefficients, float) * tp
    return apply_chain(prediction, DT_S, "H100", 0.0) if h100_meter else prediction


def one_second_mean(values: np.ndarray) -> np.ndarray:
    factor = int(round(1.0 / DT_S))
    n = values.size // factor * factor
    return np.asarray(values[:n], float).reshape(-1, factor).mean(axis=1)


def _simulation(record, manifest, timing_fit):
    params = timing_fit[record.hardware]
    requests, order, origin_array = expansion.request_schedule(record)
    trace = []
    simulated = simulate_requests(
        requests,
        arch=record.arch,
        hardware=record.hardware,
        tp=record.tp,
        eff_flops=float(params["eff_flops"]),
        eff_bw=float(params["eff_bw"]),
        transformer_bw_scale=transformer_bw_scale(record.arch, params, record.hardware),
        t_launch_s=launch_overhead_s(
            record.arch,
            base_s=float(params["base_overhead_s"]),
            per_message_s=float(params["per_message_s"][str(record.tp)]),
        ),
        t_sample_s=float(params.get("per_token_sample_s", 0.0)),
        engine=expansion.engine_from_manifest(manifest),
        iteration_trace=trace,
    )
    origin = float(origin_array[0])
    bins = emit_bins(
        trace,
        simulated,
        arch=record.arch,
        tp=record.tp,
        dt=DT_S,
        horizon_s=expansion.measured_horizon_s(record, origin),
    )
    rows = []
    first = float(params["first_token_overhead_s"])
    for result, index in zip(simulated, order):
        measured_ttft = float(record.ttfts[index])
        measured_decode = float(record.decode_times[index])
        rows.append({
            "measured_ttft_s": measured_ttft,
            "predicted_ttft_s": result["ttft_s"] + first,
            "measured_decode_s": measured_decode,
            "predicted_decode_s": result["decode_duration_s"],
            "measured_e2e_s": measured_ttft + measured_decode,
            "predicted_e2e_s": result["e2e_s"] + first,
        })
    return bins, origin, expansion._timing_summary(rows)


def _counter_diagnostic(record, bins: dict) -> dict:
    engine = record.engine_table
    running = np.asarray(engine["num_requests_running"], float)
    waiting = np.asarray(engine["num_requests_waiting"], float)
    active = np.flatnonzero((running + waiting) > 0.0)
    lo, hi = max(int(active[0]) - 1, 0), min(int(active[-1]) + 1, running.size - 1)
    measured_iterations = float(
        np.asarray(engine["iteration_tokens_total_count"], float)[hi]
        - np.asarray(engine["iteration_tokens_total_count"], float)[lo]
    )
    simulated_iterations = float(np.sum(bins["engine_iterations_rate"]) * DT_S)
    return {
        "measured_engine_iterations": measured_iterations,
        "simulated_engine_iterations": simulated_iterations,
        "simulated_iteration_error_pct": 100.0
        * (simulated_iterations - measured_iterations)
        / measured_iterations,
        "measured_mean_running_requests_when_active": float(
            np.mean(running[running > 0.0])
        ),
        "simulated_busy_weighted_decode_batch": float(
            np.average(bins["batch"], weights=bins["busy"])
        ),
    }


def _architecture_comparison(record) -> dict:
    fields = ("n_active", "w_bytes", "n_experts", "top_k", "n_layers", "d_model")
    output = {"qwen3-30b-a3b": {key: record.arch[key] for key in fields}}
    patterns = {
        "gpt-oss-20b": "data/runs/a100_iteration_gpt-oss-20b/*/manifest.json",
        "gpt-oss-120b": "data/runs/a100_hardcells_gpt-oss-120b/*/manifest.json",
    }
    for model, pattern in patterns.items():
        path = next(ROOT.glob(pattern))
        arch = json.loads(path.read_text())["arch"]
        output[model] = {key: arch[key] for key in fields}
    return output


def _candidate_record(measured, prediction) -> dict:
    metrics = trace_metrics(measured, prediction, native_dt=DT_S)
    return {
        key: float(metrics[key])
        for key in (
            "energy_error_pct", "acf_mae", "acf_r2",
            "soft_dtw_divergence", "nrmse_range",
        )
    } | {
        "measured_mean_w": float(np.mean(measured)),
        "predicted_mean_w": float(np.mean(prediction)),
        "mean_bias_pct": 100.0
        * (float(np.mean(prediction)) - float(np.mean(measured)))
        / float(np.mean(measured)),
    }


def _plot(measured, predictions, metrics):
    measured_1s = one_second_mean(measured)
    time_s = np.arange(measured_1s.size)
    colors = ("#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7")
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=True)
    for axis, (name, prediction), color in zip(axes.flat, predictions.items(), colors):
        predicted_1s = one_second_mean(prediction)
        axis.plot(time_s, measured_1s, color="black", linewidth=1.25, label="Measured")
        axis.plot(time_s, predicted_1s, color=color, linewidth=1.15, label=name)
        row = metrics[name]
        axis.set_title(
            f"{name}\nenergy {row['energy_error_pct']:.1f}% | "
            f"ACF R² {row['acf_r2']:.3f} | NRMSE {row['nrmse_range']:.3f}"
        )
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Seconds from first request")
    for axis in axes[:, 0]:
        axis.set_ylabel("TP2 node power (W)")
    handles = [
        plt.Line2D([], [], color="black", linewidth=1.5, label="Measured"),
        plt.Line2D([], [], color="#666666", linewidth=1.5, label="Candidate"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=2, frameon=False,
    )
    fig.suptitle(
        "Qwen3-30B-A3B H100 MoE transfer candidates", y=0.998, fontsize=15
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)


def main() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    record = load_bundle_run(BUNDLE)
    timing_fit = json.loads(TIMING_FIT.read_text())
    power_fit = json.loads(POWER_FIT.read_text())
    bins, origin, timing = _simulation(record, manifest, timing_fit)
    ledger = expansion._ledger_design(record, bins)
    measured_raw = expansion._measured_power_on_grid(record, origin, bins["n"], DT_S)
    valid = np.flatnonzero(np.isfinite(measured_raw))
    lo, hi = int(valid[0]), int(valid[-1] + 1)
    measured, interpolated = interpolate_nan(measured_raw[lo:hi])
    idle, idle_per_device, idle_samples = target_idle_w_per_gpu(record, manifest)
    ratios = hardware_energy_ratios(power_fit)

    predictions = {}
    candidate_details = {}
    source_designs = {}
    for model, fit in power_fit["moe"]["per_model"].items():
        design, names = design_for_source_fit(ledger, fit)
        source_designs[model] = (design, names)
        base = np.asarray(fit["coefficients"], float)
        for suffix, coefficients in (
            ("zero-shot", base),
            ("updated idle", replace_idle(base, names, idle)),
        ):
            label = f"{model} {suffix}"
            predictions[label] = predict_moe_node(
                design, coefficients, record.tp
            )[lo:hi]
            candidate_details[label] = {
                "source_hardware": "A100",
                "source_model": model,
                "coefficient_changes": [] if suffix == "zero-shot" else ["idle"],
                "h100_meter_response_applied": True,
            }

    fit20 = power_fit["moe"]["per_model"]["gpt-oss-20b"]
    design20, names20 = source_designs["gpt-oss-20b"]
    adapted = physics_adapt(fit20["coefficients"], names20, idle, ratios)
    label = "gpt-oss-20b H100 physics"
    predictions[label] = predict_moe_node(design20, adapted, record.tp)[lo:hi]
    candidate_details[label] = {
        "source_hardware": "A100",
        "source_model": "gpt-oss-20b",
        "coefficient_changes": ["idle", *ratios],
        "hardware_energy_per_work_ratios": ratios,
        "h100_meter_response_applied": True,
    }

    dense_fit = power_fit["dense"]["H100"]
    dense_design = filter_design(
        raw_design(ledger, "H100"), ledger["run_id"], DT_S,
        "H100", float(dense_fit["delay_s"]),
    )
    label = "dense H100 architecture baseline"
    predictions[label] = (
        dense_design @ np.asarray(dense_fit["coefficients"], float) * record.tp
    )[lo:hi]
    candidate_details[label] = {
        "source_hardware": "H100",
        "source_model": "dense source models",
        "coefficient_changes": [],
        "semantic_support": False,
        "support_reason": "Dense law assumes full-weight iteration reads for an MoE checkpoint",
    }

    metrics = {
        name: _candidate_record(measured, pred)
        for name, pred in predictions.items()
    }
    _plot(measured, predictions, metrics)
    columns = {
        "time_s": np.arange(one_second_mean(measured).size),
        "measured_w": one_second_mean(measured),
    }
    columns.update({name: one_second_mean(values) for name, values in predictions.items()})
    with TRACE_CSV.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))

    report = {
        "schema_version": "qwen3-30b-h100-moe-transfer-audit-v1",
        "evidence_role": "sealed target opened once for diagnostic transfer audit",
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "run_id": manifest["run_id"],
        "dt_s": DT_S,
        "common_evaluation_bins": int(measured.size),
        "interpolated_power_bins": int(interpolated),
        "target_idle": {
            "pooled_median_w_per_gpu": idle,
            "device_medians_w": idle_per_device,
            "native_samples": idle_samples,
            "window": manifest["probe"]["idle_window"],
        },
        "timing": timing,
        "engine_diagnostic": _counter_diagnostic(record, bins),
        "architecture": _architecture_comparison(record),
        "candidates": {
            name: {**candidate_details[name], **metrics[name]}
            for name in predictions
        },
        "diagnosis": [
            "Target idle is much higher than both A100 MoE source intercepts; idle-only calibration removes most GPT-OSS-20B energy bias.",
            "The frozen timing law underpredicts Qwen end-to-end latency and overpredicts engine iteration count, limiting event alignment independently of the power coefficients.",
            "Qwen matches GPT-OSS-120B in resident weight bytes and expert count but is closer to GPT-OSS-20B in active parameters; neither source identity captures both axes.",
            "The dense H100 trace is a diagnostic architecture baseline, not a supported MoE prediction, because it treats the full checkpoint as read each iteration.",
            "The H100-physics candidate scales only memory and compute energy per unit work using frozen dense A100/H100 laws; iteration-rate and batch residuals remain A100-specific.",
        ],
        "outputs": {
            "plot": str(PLOT.relative_to(ROOT)),
            "trace_csv": str(TRACE_CSV.relative_to(ROOT)),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {PLOT}, {REPORT}, and {TRACE_CSV}")


if __name__ == "__main__":
    main()
