"""Post-hoc calibration-only timing diagnosis for disaggregated confirmation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "power-test"), str(ROOT / "feature-test")]

import analyze_disaggregated_confirmation as base  # noqa: E402
from disaggregated_confirmation import (  # noqa: E402
    average_trace_rows,
    fit_nonnegative_gain,
    fit_positive_time_scale,
)
from disaggregated_plot import plot_confirmation_traces  # noqa: E402
from disaggregated_reporting import write_csv  # noqa: E402
from model.release import DEFAULT_ARTIFACT, load_artifact  # noqa: E402

REPORT = (
    base.OUT_DIR
    / "gpt_oss_20b_a100_pd_confirmation_timing_calibrated_report.json"
)
METRICS = (
    base.OUT_DIR
    / "gpt_oss_20b_a100_pd_confirmation_timing_calibrated_per_cell.csv"
)
TRACES = (
    base.OUT_DIR
    / "gpt_oss_20b_a100_pd_confirmation_timing_calibrated_250ms.csv"
)
FIGURE = (
    base.OUT_DIR
    / "gpt_oss_20b_a100_pd_confirmation_timing_calibrated_power.png"
)
TRACES_1S = (
    base.OUT_DIR
    / "gpt_oss_20b_a100_pd_confirmation_timing_calibrated_1s.csv"
)
FIGURE_1S = (
    base.OUT_DIR
    / "gpt_oss_20b_a100_pd_confirmation_timing_calibrated_power_1s.png"
)


def _fit_timing_scales(
    run_root: Path, release: dict, calibration_cell: str,
) -> tuple[dict[str, float], dict[str, object]]:
    request, _, _, result = base._simulate_cell(
        run_root / calibration_cell, release
    )
    events = base.load_events(run_root / "proxy_events.jsonl")
    measured_prefill = np.asarray([
        events[request_id]["prefill_completed"]
        - events[request_id]["prefill_sent"]
        for request_id in request["request_ids"]
    ])
    predicted_prefill = np.asarray([
        row["prefill_s"] for row in result.requests
    ])
    measured_decode = np.asarray([
        events[request_id]["decode_completed"]
        - events[request_id]["decode_first_byte"]
        for request_id in request["request_ids"]
    ])
    predicted_decode = np.asarray([
        row["decode_duration_s"] for row in result.requests
    ])
    first_token_overhead = float(
        release["timing"]["A100"]["first_token_overhead_s"]
    )
    scales = {
        "prefill": fit_positive_time_scale(
            measured_prefill[:1],
            predicted_prefill[:1],
            fixed_overhead_s=first_token_overhead,
        ),
        "decode": fit_positive_time_scale(
            measured_decode, predicted_decode
        ),
    }
    return scales, {
        "cell": calibration_cell,
        "prefill_fit_scope": "first uncontended request",
        "decode_fit_scope": "request-level decode service durations",
        "first_request_measured_prefill_ms": float(
            1000.0 * measured_prefill[0]
        ),
        "first_request_source_prefill_ms": float(
            1000.0 * predicted_prefill[0]
        ),
        "decode_requests": int(measured_decode.size),
    }


def _fit_power_calibration(
    run_root: Path,
    metadata: dict,
    release: dict,
    calibration_cell: str,
    timing_scales: dict[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    idles, idle_evidence = base._idle_levels(
        run_root, metadata, calibration_cell
    )
    _, start, _, result = base._simulate_cell(
        run_root / calibration_cell, release, timing_scales
    )
    samples = base._role_samples(
        run_root / calibration_cell, metadata, result, start
    )
    calibration = {}
    for role in base.ROLES:
        calibration[role] = {
            "target_idle_w": idles[role],
            "model_gain": fit_nonnegative_gain(
                samples[role]["measured"],
                samples[role]["source"] - result.source_idle_w_per_gpu,
                measured_idle_w=idles[role],
            ),
            "duty_gain_w": fit_nonnegative_gain(
                samples[role]["measured"],
                samples[role]["busy"],
                measured_idle_w=idles[role],
            ),
        }
    return calibration, {
        "idle": idle_evidence,
        "source_idle_w": result.source_idle_w_per_gpu,
    }


def _evaluate(
    run_root: Path,
    metadata: dict,
    release: dict,
    timing_scales: dict[str, float],
    calibration: dict[str, dict[str, float]],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    specs = base._cell_specs(metadata)
    thresholds = metadata["analysis_protocol"]["heldout_role_acceptance"]
    event_table = base.load_events(run_root / "proxy_events.jsonl")
    rows, trace_rows, timings = [], [], []
    replay_samples = {}
    for cell in sorted((run_root / name for name in specs), key=base.cell_key):
        spec = specs[cell.name]
        request, start, _, result = base._simulate_cell(
            cell, release, timing_scales
        )
        samples = base._role_samples(cell, metadata, result, start)
        rate, repeat = base.cell_key(cell)
        for role in base.ROLES:
            row, traces = base._evaluate_role(
                cell.name,
                spec["split"],
                role,
                samples[role],
                calibration[role],
                result.source_idle_w_per_gpu,
                thresholds,
            )
            rows.append(row)
            replay_samples[(rate, repeat, role)] = samples[role]
            if cell.name == base.FIGURE_CELL:
                if not trace_rows:
                    trace_rows = [
                        {"cell": cell.name, "time_s": float(time_s)}
                        for time_s in samples[role]["time_s"]
                    ]
                for output, measured, predicted, duty in zip(
                    trace_rows,
                    traces["measured"],
                    traces["predicted"],
                    traces["duty_null"],
                ):
                    output[f"measured_{role}_w"] = float(measured)
                    output[f"predicted_{role}_w"] = float(predicted)
                    output[f"duty_null_{role}_w"] = float(duty)
        timings.append({
            "cell": cell.name,
            "split": spec["split"],
            **base.timing_metrics(request, result.requests, event_table),
        })
    replay_threshold = metadata["analysis_protocol"][
        "measured_replay_acceptance"
    ]["correlation_min"]
    return (
        rows,
        trace_rows,
        timings,
        base._replay_evidence(replay_samples, replay_threshold),
    )


def main() -> None:
    run_root = base._run_root()
    metadata = json.loads((run_root / "run_metadata.json").read_text())
    release = load_artifact()
    calibration_cell = metadata["analysis_protocol"]["calibration_cell"]
    timing_scales, timing_evidence = _fit_timing_scales(
        run_root, release, calibration_cell
    )
    calibration, power_evidence = _fit_power_calibration(
        run_root, metadata, release, calibration_cell, timing_scales
    )
    rows, trace_rows, timings, replay = _evaluate(
        run_root, metadata, release, timing_scales, calibration
    )
    heldout = [row for row in rows if row["split"] == "evaluation"]
    summary = {}
    for role in base.ROLES:
        selected = [row for row in heldout if row["role"] == role]
        summary[role] = {
            "cells": len(selected),
            "accepted_cells": sum(row["accept_accepted"] for row in selected),
            "accepted": all(row["accept_accepted"] for row in selected),
            "correlation_median": base._median(selected, "correlation"),
            "std_ratio_median": base._median(selected, "std_ratio"),
            "p95_error_pct_median": base._median(
                selected, "p95_error_pct"
            ),
            "energy_error_pct_median": base._median(
                selected, "energy_error_pct"
            ),
            "model_to_duty_null_loss_ratio_median": base._median(
                selected, "model_to_duty_null_loss_ratio"
            ),
        }
    report = {
        "schema_version": (
            "powertrace-disaggregated-confirmation-timing-diagnostic-v1"
        ),
        "evidence_role": (
            "post-hoc calibration-cell-only timing diagnostic; "
            "not the preregistered confirmation result"
        ),
        "campaign": str(run_root.relative_to(ROOT)),
        "release": {
            "path": str(DEFAULT_ARTIFACT.relative_to(ROOT)),
            "sha256": base.sha256(DEFAULT_ARTIFACT),
        },
        "model_adjustment": {
            "base_timing_coefficients_changed": False,
            "base_power_coefficients_changed": False,
            "target_scalars": 6,
            "role_timing_scales": timing_scales,
            "role_power_parameters": calibration,
            "fit_cell": calibration_cell,
        },
        "timing_calibration_evidence": timing_evidence,
        "power_calibration_evidence": power_evidence,
        "heldout_role_acceptance": summary,
        "measured_replay_acceptance": replay,
        "timing_per_cell": timings,
        "per_cell": rows,
        "outputs": {
            "per_cell_metrics": str(METRICS.relative_to(ROOT)),
            "representative_traces": str(TRACES.relative_to(ROOT)),
            "figure": str(FIGURE.relative_to(ROOT)),
            "representative_traces_1s": str(TRACES_1S.relative_to(ROOT)),
            "figure_1s": str(FIGURE_1S.relative_to(ROOT)),
        },
    }
    write_csv(METRICS, rows)
    write_csv(TRACES, trace_rows)
    trace_rows_1s = average_trace_rows(trace_rows, bin_s=1.0)
    write_csv(TRACES_1S, trace_rows_1s)
    plot_confirmation_traces(
        trace_rows,
        calibration,
        FIGURE,
        cell=base.FIGURE_CELL,
        title=(
            "Post-hoc role-timing diagnosis: held-out 2 requests/s "
            "(native query samples)"
        ),
        note=(
            "No smoothing/interpolation/lag/warping; two timing scales and "
            "four power scalars use rate-2-repeat-0 only."
        ),
    )
    plot_confirmation_traces(
        trace_rows_1s,
        calibration,
        FIGURE_1S,
        cell=base.FIGURE_CELL,
        title=(
            "Post-hoc role-timing diagnosis: held-out 2 requests/s "
            "(non-overlapping 1 s means)"
        ),
        note=(
            "Arithmetic means of raw query samples in fixed 1 s bins; no "
            "interpolation, fitted lag, or warping."
        ),
        sample_label="Measured 1 s mean",
    )
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"timing scales={timing_scales}; prefill accepted="
        f"{summary['prefill']['accepted']}, decode accepted="
        f"{summary['decode']['accepted']}"
    )


if __name__ == "__main__":
    main()
