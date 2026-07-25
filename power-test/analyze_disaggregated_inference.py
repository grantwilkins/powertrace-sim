"""Evaluate role-aware PowerTrace transfer on the disaggregated GPT-OSS run."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "feature-test")]

from disaggregated_analysis_core import (  # noqa: E402
    apply_phase_calibration,
    align_power,
    campaign_sha256,
    cell_key,
    collocated_tp2,
    fit_diagonal_soft_dtw_scale,
    idle_calibration,
    load_events,
    load_power,
    measured_grid,
    power_metrics,
    role_trace_metrics,
    sha256,
    telemetry,
    timing_metrics,
)
from disaggregated_native import sample_native_power  # noqa: E402
from disaggregated_plot import plot_phase_traces  # noqa: E402
from disaggregated_reporting import build_report, write_csv  # noqa: E402
from model.disaggregated import (  # noqa: E402
    apply_shared_idle_calibration,
    simulate_disaggregated,
)
from model.release import DEFAULT_ARTIFACT, load_artifact  # noqa: E402
from model.timing.ledger import NATIVE_DT_S  # noqa: E402
from profiling.disaggregated_prefill.campaign import (  # noqa: E402
    validate_events,
    validate_result,
)

DATA_ROOT = ROOT / "data/disagg"
OUT_DIR = ROOT / "results/disaggregated"
REPORT = OUT_DIR / "gpt_oss_20b_a100_pd_report.json"
METRICS = OUT_DIR / "gpt_oss_20b_a100_pd_per_cell.csv"
TRACES = OUT_DIR / "gpt_oss_20b_a100_pd_representative_250ms.csv"
FIGURE = OUT_DIR / "gpt_oss_20b_a100_pd_power_comparison.png"
CALIBRATION_CELL = "rate-0p25-repeat-0"
DYNAMIC_CALIBRATION_CELL = "rate-2-repeat-1"
FIGURE_CELL = "rate-4-repeat-2"
REPRESENTATIVE_REPEAT = 2


def _run_root(data_root: Path = DATA_ROOT) -> Path:
    roots = sorted(data_root.glob("gpt-oss-20b-a100-pd-*"))
    roots = [path for path in roots if (path / "run_metadata.json").is_file()]
    roots = [path for path in roots if "confirmatory" not in path.name]
    if len(roots) != 1:
        raise ValueError(f"expected one pilot run under {data_root}, found {roots}")
    return roots[0]


def _simulate_cell(cell: Path, release: dict):
    request = json.loads((cell / "requests.json").read_text())
    start = float((cell / "start_epoch_s").read_text())
    end = float((cell / "end_epoch_s").read_text())
    schedule = [
        {
            "arrival_time": timestamp - start,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        for timestamp, input_tokens, output_tokens in zip(
            request["request_timestamps"],
            request["input_lens"],
            request["output_lens"],
        )
    ]
    result = simulate_disaggregated(
        schedule,
        deployment={
            "preset": "gpt-oss-20b-a100-tp1",
            "overrides": {"max_num_batched_tokens": 8192},
        },
        artifact=release,
        horizon_s=end - start,
        allow_unsupported=True,
    )
    return request, start, end, schedule, result


def _phase_split(cell: str) -> str:
    if cell == CALIBRATION_CELL:
        return "idle_calibration"
    if cell == DYNAMIC_CALIBRATION_CELL:
        return "soft_dtw_calibration"
    if cell in ("rate-0p25-repeat-2", "rate-4-repeat-2"):
        return "heldout_rate"
    if cell == "rate-2-repeat-2":
        return "same_rate_repeatability"
    return "diagnostic"


def _fit_phase_calibration(
    run_root: Path, metadata: dict, release: dict, role_idles: dict[str, float]
) -> tuple[dict[str, dict[str, object]], float]:
    cell = run_root / DYNAMIC_CALIBRATION_CELL
    _, start, _, _, result = _simulate_cell(cell, release)
    timestamps, power = load_power(cell, metadata)
    output = {}
    for index, role in enumerate(("prefill", "decode")):
        source = np.asarray(
            result.roles[role].power["node_gpu_power_w"], dtype=float
        )
        _, observed, aligned = sample_native_power(
            timestamps,
            power[:, index],
            {"source": source},
            start_epoch_s=start,
            dt_s=NATIVE_DT_S,
        )
        source = aligned["source"]
        fitted = fit_diagonal_soft_dtw_scale(
            observed,
            source,
            measured_idle_w=role_idles[role],
            predicted_idle_w=result.source_idle_w_per_gpu,
        )
        baseline = apply_phase_calibration(
            source, source_idle_w=result.source_idle_w_per_gpu,
            target_idle_w=role_idles[role], dynamic_scale=1.0,
        )
        candidate = apply_phase_calibration(
            source, source_idle_w=result.source_idle_w_per_gpu,
            target_idle_w=role_idles[role], dynamic_scale=fitted,
        )
        baseline_metrics = role_trace_metrics(observed, baseline)
        candidate_metrics = role_trace_metrics(observed, candidate)
        guards = {
            "diagonal_soft_dtw": (
                candidate_metrics["soft_dtw_diagonal_divergence"]
                <= baseline_metrics["soft_dtw_diagonal_divergence"]
            ),
            "absolute_mean_bias": (
                abs(candidate_metrics["mean_bias_w"])
                <= abs(baseline_metrics["mean_bias_w"])
            ),
            "p95_error": (
                candidate_metrics["p95_error_pct"]
                <= baseline_metrics["p95_error_pct"]
            ),
            "std_ratio": (
                abs(candidate_metrics["std_ratio"] - 1.0)
                <= abs(baseline_metrics["std_ratio"] - 1.0)
            ),
        }
        accepted = all(guards.values())
        output[role] = {
            "fitted_scale": fitted,
            "accepted_scale": fitted if accepted else 1.0,
            "accepted": accepted,
            "acceptance_guards": guards,
            "baseline_metrics": baseline_metrics,
            "fitted_metrics": candidate_metrics,
        }
    return output, result.source_idle_w_per_gpu


def main() -> None:
    run_root = _run_root()
    metadata = json.loads((run_root / "run_metadata.json").read_text())
    if (
        metadata["model"] != "openai/gpt-oss-20b"
        or metadata["hardware"] != "A100-80GB"
        or metadata["deployment"] != "disaggregated_prefill"
    ):
        raise ValueError("campaign metadata is not the declared GPT-OSS PD deployment")
    release = load_artifact()
    event_table = load_events(run_root / "proxy_events.jsonl")
    calibration = idle_calibration(run_root, metadata, CALIBRATION_CELL)
    target_idle = float(calibration["target_idle_w_per_gpu"])
    role_idles = dict(zip(
        ("prefill", "decode"), calibration["device_medians_w"]
    ))
    phase_calibration, source_idle = _fit_phase_calibration(
        run_root, metadata, release, role_idles
    )
    cells = sorted(run_root.glob("rate-*"), key=cell_key)
    rows, trace_rows, cell_diagnostics = [], [], []
    cell_role_traces = {}
    candidates = (
        "role_aware_zero_shot",
        "role_aware_idle_calibrated",
        "phase_soft_dtw_calibrated",
        "colocated_tp2_idle_calibrated",
    )

    for cell in cells:
        rate, repeat = cell_key(cell)
        request, start, end, schedule, role_aware = _simulate_cell(cell, release)
        request_ids = validate_result(cell / "requests.json", int(request["completed"]))
        validate_events(run_root / "proxy_events.jsonl", 0, request_ids)
        horizon = end - start
        calibrated = apply_shared_idle_calibration(role_aware, target_idle)
        phase_roles = {
            role: apply_phase_calibration(
                np.asarray(
                    role_aware.roles[role].power["node_gpu_power_w"], dtype=float
                ),
                source_idle_w=source_idle,
                target_idle_w=role_idles[role],
                dynamic_scale=float(
                    phase_calibration[role]["accepted_scale"]
                ),
            )
            for role in ("prefill", "decode")
        }
        collocated = collocated_tp2(schedule, horizon, release)
        collocated += 2.0 * (target_idle - role_aware.source_idle_w_per_gpu)
        predictions = {
            "role_aware_zero_shot": role_aware.node_gpu_power_w,
            "role_aware_idle_calibrated": calibrated["node_gpu_power_w"],
            "phase_soft_dtw_calibrated": sum(phase_roles.values()),
            "colocated_tp2_idle_calibrated": collocated,
        }
        power_timestamps, role_power = load_power(cell, metadata)
        measured_roles = measured_grid(
            power_timestamps, role_power, start, role_aware.node_gpu_power_w.size
        )
        measured, aligned, gaps, coverage = align_power(
            measured_roles.sum(axis=1), predictions
        )
        role_errors = {}
        representative_roles = {}
        representative_time_s = None
        for role_index, role in enumerate(("prefill", "decode")):
            source_role = np.asarray(
                role_aware.roles[role].power["node_gpu_power_w"], dtype=float
            )
            native_time_s, measured_role, predicted_role = sample_native_power(
                power_timestamps,
                role_power[:, role_index],
                {
                    "frozen": apply_phase_calibration(
                        source_role,
                        source_idle_w=source_idle,
                        target_idle_w=role_idles[role],
                        dynamic_scale=1.0,
                    ),
                    "accepted": phase_roles[role],
                },
                start_epoch_s=start,
                dt_s=NATIVE_DT_S,
            )
            signed = 100.0 * (
                float(predicted_role["accepted"].mean())
                - float(measured_role.mean())
            ) / float(measured_role.mean())
            role_errors[f"{role}_energy_error_pct"] = abs(signed)
            role_errors[f"{role}_mean_bias_pct"] = signed
            role_errors[f"{role}_measured_mean_w"] = float(measured_role.mean())
            role_errors[f"{role}_predicted_mean_w"] = float(
                predicted_role["accepted"].mean()
            )
            role_errors.update({
                f"{role}_trace_{name}": value
                for name, value in role_trace_metrics(
                    measured_role, predicted_role["accepted"]
                ).items()
            })
            role_errors.update({
                f"{role}_frozen_trace_{name}": value
                for name, value in role_trace_metrics(
                    measured_role, predicted_role["frozen"]
                ).items()
            })
            cell_role_traces[(rate, repeat, role)] = measured_role
            if repeat == REPRESENTATIVE_REPEAT:
                representative_time_s = native_time_s
                representative_roles[f"measured_{role}_w"] = measured_role
                representative_roles[f"predicted_{role}_w"] = predicted_role[
                    "accepted"
                ]
                representative_roles[f"frozen_{role}_w"] = predicted_role[
                    "frozen"
                ]

        timing = timing_metrics(request, role_aware.requests, event_table)
        telemetry_row = telemetry(cell, power_timestamps)
        if (
            telemetry_row["nixl_failed_transfers"]
            or telemetry_row["nixl_failed_notifications"]
        ):
            raise ValueError(f"{cell} contains failed NIXL operations")
        if telemetry_row["decode_preemptions"]:
            raise ValueError(f"{cell} contains decoder preemptions")
        split = (
            "calibration"
            if cell.name in (CALIBRATION_CELL, DYNAMIC_CALIBRATION_CELL)
            else "evaluation"
        )
        for candidate in candidates:
            metric = power_metrics(
                measured, aligned[candidate], bool(coverage["temporal_supported"])
            )
            rows.append({
                "cell": cell.name,
                "rate": rate,
                "repeat": repeat,
                "split": split,
                "phase_split": _phase_split(cell.name),
                "candidate": candidate,
                "requests": int(request["completed"]),
                "duration_s": float(measured.size * NATIVE_DT_S),
                "interpolated_power_bins": gaps,
                **coverage,
                **metric,
                **role_errors,
                **timing,
                **telemetry_row,
            })

        if repeat == REPRESENTATIVE_REPEAT:
            native_time_s, measured_node, native_predictions = sample_native_power(
                power_timestamps,
                role_power.sum(axis=1),
                predictions,
                start_epoch_s=start,
                dt_s=NATIVE_DT_S,
            )
            if not np.array_equal(native_time_s, representative_time_s):
                raise ValueError(f"{cell} role and node sample times differ")
            trace_series = {
                "measured_node_w": measured_node,
                **{
                    f"{candidate}_w": native_predictions[candidate]
                    for candidate in candidates
                },
                **representative_roles,
            }
            if len({values.size for values in trace_series.values()}) != 1:
                raise ValueError(f"{cell} role and node trace lengths differ")
            for time_s, values in zip(
                native_time_s, zip(*trace_series.values())
            ):
                trace_rows.append({
                    "cell": cell.name,
                    "rate": rate,
                    "time_s": float(time_s),
                    **dict(zip(trace_series, values)),
                })
        cell_diagnostics.append({
            "cell": cell.name,
            "rate": rate,
            "repeat": repeat,
            "input_lengths_sha256": hashlib.sha256(
                np.asarray(request["input_lens"], dtype=np.int64).tobytes()
            ).hexdigest(),
            "measured_prefill_mean_w": role_errors["prefill_measured_mean_w"],
            **telemetry_row,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS, rows)
    write_csv(TRACES, trace_rows)
    plot_phase_traces(
        trace_rows, rows, phase_calibration, FIGURE, cell=FIGURE_CELL
    )
    campaign_hash, campaign_files = campaign_sha256(run_root)
    report = build_report(
        campaign={
            "path": str(run_root.relative_to(ROOT)),
            "sha256": campaign_hash,
            "hashed_files": campaign_files,
            "cells": len(cells),
            "requests": sum(
                int(json.loads((cell / "requests.json").read_text())["completed"])
                for cell in cells
            ),
        },
        release={
            "path": str(DEFAULT_ARTIFACT.relative_to(ROOT)),
            "sha256": sha256(DEFAULT_ARTIFACT),
            "release_status": release["release_status"],
        },
        candidates=candidates,
        rows=rows,
        calibration=calibration,
        source_idle_w=source_idle,
        phase_calibration=phase_calibration,
        cell_diagnostics=cell_diagnostics,
        cell_role_traces=cell_role_traces,
        outputs={
            "per_cell_metrics": str(METRICS.relative_to(ROOT)),
            "representative_traces": str(TRACES.relative_to(ROOT)),
            "figure": str(FIGURE.relative_to(ROOT)),
        },
    )
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["power"]["summaries"]["phase_soft_dtw_calibrated"]
    print(
        f"wrote {REPORT}, {METRICS}, {TRACES}, and {FIGURE}; "
        f"held-out median energy error "
        f"{summary['energy_error_pct_median']:.2f}%"
    )


if __name__ == "__main__":
    main()
