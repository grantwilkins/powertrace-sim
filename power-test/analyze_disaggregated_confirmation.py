"""Evaluate the frozen cache-disabled disaggregated confirmation protocol."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "feature-test")]

from disaggregated_analysis_core import (  # noqa: E402
    campaign_sha256,
    cell_key,
    load_events,
    load_power,
    role_trace_metrics,
    sha256,
    timing_metrics,
)
from disaggregated_confirmation import (  # noqa: E402
    fit_nonnegative_gain,
    pair_common_bin_samples,
    pointwise_loss_ratio,
    role_acceptance,
)
from disaggregated_native import sample_native_power  # noqa: E402
from disaggregated_plot import plot_confirmation_traces  # noqa: E402
from disaggregated_reporting import write_csv  # noqa: E402
from model.disaggregated import simulate_disaggregated  # noqa: E402
from model.release import DEFAULT_ARTIFACT, load_artifact  # noqa: E402
from model.timing.ledger import NATIVE_DT_S  # noqa: E402
from profiling.disaggregated_prefill.campaign import (  # noqa: E402
    validate_events,
    validate_power,
    validate_prefill_observability,
    validate_result,
    validate_uncached_metrics,
    validate_workload,
)

DATA_ROOT = ROOT / "data/disagg"
OUT_DIR = ROOT / "results/disaggregated"
REPORT = OUT_DIR / "gpt_oss_20b_a100_pd_confirmation_report.json"
METRICS = OUT_DIR / "gpt_oss_20b_a100_pd_confirmation_per_cell.csv"
TRACES = OUT_DIR / "gpt_oss_20b_a100_pd_confirmation_250ms.csv"
FIGURE = OUT_DIR / "gpt_oss_20b_a100_pd_confirmation_power.png"
ROLES = ("prefill", "decode")
FIGURE_CELL = "rate-2-repeat-2"


def _run_root() -> Path:
    roots = sorted(DATA_ROOT.glob("gpt-oss-20b-a100-pd-confirmatory-*"))
    roots = [path for path in roots if (path / "run_metadata.json").is_file()]
    if len(roots) != 1:
        raise ValueError(f"expected one confirmatory run, found {roots}")
    return roots[0]


def _cell_specs(metadata: dict) -> dict[str, dict]:
    return {
        f"rate-{cell['rate']:g}-repeat-{cell['repeat']}": cell
        for cell in metadata["workload"]["cells"]
    }


def _validate_cell(
    run_root: Path, cell: Path, spec: dict, metadata: dict,
) -> None:
    request_path = cell / "requests.json"
    request_ids = validate_result(request_path, int(spec["prompts"]))
    validate_workload(request_path)
    timelines = validate_events(
        run_root / "proxy_events.jsonl", 0, request_ids
    )
    validate_prefill_observability(timelines)
    validate_uncached_metrics(
        request_path,
        cell / "engine_prefill.csv",
        cell / "engine_decode.csv",
    )
    validate_power(
        cell / "power.csv",
        gpu_uuids=tuple(
            metadata["roles"][role]["gpu_uuid"] for role in ROLES
        ),
        workload_start=float((cell / "workload_start_epoch_s").read_text()),
        workload_end=float((cell / "workload_end_epoch_s").read_text()),
    )


def _simulate_cell(
    cell: Path,
    release: dict,
    role_timing_scales: dict[str, float] | None = None,
):
    request = json.loads((cell / "requests.json").read_text())
    start = float((cell / "workload_start_epoch_s").read_text())
    end = float((cell / "workload_end_epoch_s").read_text())
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
        deployment="gpt-oss-20b-a100-tp1",
        artifact=release,
        horizon_s=end - start,
        role_timing_scales=role_timing_scales,
        allow_unsupported=role_timing_scales is not None,
    )
    return request, start, end, result


def _idle_levels(
    run_root: Path, metadata: dict, calibration_cell: str,
) -> tuple[dict[str, float], dict[str, object]]:
    cell = run_root / calibration_cell
    capture_start = float((cell / "start_epoch_s").read_text())
    workload_start = float((cell / "workload_start_epoch_s").read_text())
    timestamps, power = load_power(cell, metadata)
    selected = (
        (timestamps >= capture_start + 2.0)
        & (timestamps < workload_start - 1.0)
    )
    if selected.sum() < 80:
        raise ValueError("calibration cell has too few settled idle samples")
    medians = np.median(power[selected], axis=0)
    return (
        dict(zip(ROLES, map(float, medians))),
        {
            "cell": calibration_cell,
            "window_start_epoch_s": capture_start + 2.0,
            "window_end_epoch_s": workload_start - 1.0,
            "timestamp_samples": int(selected.sum()),
            "role_medians_w": dict(zip(ROLES, map(float, medians))),
        },
    )


def _role_samples(
    cell: Path, metadata: dict, result, start: float,
) -> dict[str, dict[str, np.ndarray]]:
    timestamps, power = load_power(cell, metadata)
    output = {}
    for index, role in enumerate(ROLES):
        source = np.asarray(
            result.roles[role].power["node_gpu_power_w"], dtype=float
        )
        busy = np.asarray(result.roles[role].ledger["busy"], dtype=float)
        time_s, measured, predictors = sample_native_power(
            timestamps,
            power[:, index],
            {"source": source, "busy": busy},
            start_epoch_s=start,
            dt_s=NATIVE_DT_S,
        )
        output[role] = {
            "time_s": time_s,
            "bins": np.floor(time_s / NATIVE_DT_S).astype(int),
            "measured": measured,
            **predictors,
        }
    return output


def _fit_calibration(
    run_root: Path,
    metadata: dict,
    release: dict,
    calibration_cell: str,
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    idles, idle_evidence = _idle_levels(
        run_root, metadata, calibration_cell
    )
    _, start, _, result = _simulate_cell(run_root / calibration_cell, release)
    samples = _role_samples(
        run_root / calibration_cell, metadata, result, start
    )
    calibration = {}
    for role in ROLES:
        source_dynamic = (
            samples[role]["source"] - result.source_idle_w_per_gpu
        )
        calibration[role] = {
            "target_idle_w": idles[role],
            "model_gain": fit_nonnegative_gain(
                samples[role]["measured"],
                source_dynamic,
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


def _energy_error(measured: np.ndarray, predicted: np.ndarray) -> float:
    return 100.0 * abs(float(predicted.mean() - measured.mean())) / float(
        measured.mean()
    )


def _evaluate_role(
    cell: str,
    split: str,
    role: str,
    samples: dict[str, np.ndarray],
    calibration: dict[str, float],
    source_idle_w: float,
    thresholds: dict[str, float],
) -> tuple[dict, dict[str, np.ndarray]]:
    measured = samples["measured"]
    source = samples["source"]
    idle_only = calibration["target_idle_w"] + (source - source_idle_w)
    predicted = calibration["target_idle_w"] + calibration["model_gain"] * (
        source - source_idle_w
    )
    duty_null = (
        calibration["target_idle_w"]
        + calibration["duty_gain_w"] * samples["busy"]
    )
    metrics = role_trace_metrics(measured, predicted)
    null_metrics = role_trace_metrics(measured, duty_null)
    ratio = pointwise_loss_ratio(measured, predicted, duty_null)
    checks = role_acceptance(metrics, ratio, thresholds)
    row = {
        "cell": cell,
        "split": split,
        "role": role,
        "samples": int(measured.size),
        "target_idle_w": calibration["target_idle_w"],
        "model_gain": calibration["model_gain"],
        "duty_gain_w": calibration["duty_gain_w"],
        "zero_shot_energy_error_pct": _energy_error(measured, source),
        "idle_only_energy_error_pct": _energy_error(measured, idle_only),
        "energy_error_pct": _energy_error(measured, predicted),
        **metrics,
        **{f"duty_null_{key}": value for key, value in null_metrics.items()},
        "model_to_duty_null_loss_ratio": ratio,
        **{f"accept_{key}": value for key, value in checks.items()},
    }
    return row, {
        "measured": measured,
        "predicted": predicted,
        "duty_null": duty_null,
    }


def _replay_evidence(
    raw: dict[tuple[float, int, str], dict[str, np.ndarray]],
    threshold: float,
) -> dict[str, dict[str, object]]:
    pairs = {1.0: (0, 1), 2.0: (1, 2)}
    output = {}
    for rate, repeats in pairs.items():
        output[str(rate)] = {}
        for role in ROLES:
            left = raw[(rate, repeats[0], role)]
            right = raw[(rate, repeats[1], role)]
            left_values, right_values = pair_common_bin_samples(
                left["bins"],
                left["measured"],
                right["bins"],
                right["measured"],
            )
            correlation = float(np.corrcoef(left_values, right_values)[0, 1])
            output[str(rate)][role] = {
                "samples": int(left_values.size),
                "correlation": correlation,
                "accepted": correlation >= threshold,
            }
    return output


def _median(rows: list[dict], field: str) -> float:
    return float(np.median([row[field] for row in rows]))


def main() -> None:
    run_root = _run_root()
    metadata = json.loads((run_root / "run_metadata.json").read_text())
    if metadata["protocol"] != "prospective_cache_disabled_confirmation_v1":
        raise ValueError("run is not the frozen cache-disabled confirmation")
    specs = _cell_specs(metadata)
    release = load_artifact()
    calibration_cell = metadata["analysis_protocol"]["calibration_cell"]
    calibration, evidence = _fit_calibration(
        run_root, metadata, release, calibration_cell
    )
    thresholds = metadata["analysis_protocol"]["heldout_role_acceptance"]
    replay_threshold = metadata["analysis_protocol"][
        "measured_replay_acceptance"
    ]["correlation_min"]
    event_table = load_events(run_root / "proxy_events.jsonl")
    rows, trace_rows, timings = [], [], []
    replay_samples = {}

    for cell in sorted((run_root / name for name in specs), key=cell_key):
        spec = specs[cell.name]
        _validate_cell(run_root, cell, spec, metadata)
        request, start, _, result = _simulate_cell(cell, release)
        samples = _role_samples(cell, metadata, result, start)
        rate, repeat = cell_key(cell)
        for role in ROLES:
            row, traces = _evaluate_role(
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
            if cell.name == FIGURE_CELL:
                if not trace_rows:
                    trace_rows = [
                        {
                            "cell": cell.name,
                            "time_s": float(time_s),
                        }
                        for time_s in samples[role]["time_s"]
                    ]
                for trace_row, measured, predicted, duty in zip(
                    trace_rows,
                    traces["measured"],
                    traces["predicted"],
                    traces["duty_null"],
                ):
                    trace_row[f"measured_{role}_w"] = float(measured)
                    trace_row[f"predicted_{role}_w"] = float(predicted)
                    trace_row[f"duty_null_{role}_w"] = float(duty)
        timings.append({
            "cell": cell.name,
            "split": spec["split"],
            **timing_metrics(request, result.requests, event_table),
        })

    heldout = [row for row in rows if row["split"] == "evaluation"]
    replay = _replay_evidence(replay_samples, replay_threshold)
    role_summary = {}
    for role in ROLES:
        selected = [row for row in heldout if row["role"] == role]
        role_summary[role] = {
            "cells": len(selected),
            "accepted_cells": sum(row["accept_accepted"] for row in selected),
            "accepted": all(row["accept_accepted"] for row in selected),
            "correlation_median": _median(selected, "correlation"),
            "std_ratio_median": _median(selected, "std_ratio"),
            "p95_error_pct_median": _median(selected, "p95_error_pct"),
            "energy_error_pct_median": _median(selected, "energy_error_pct"),
            "zero_shot_energy_error_pct_median": _median(
                selected, "zero_shot_energy_error_pct"
            ),
            "model_to_duty_null_loss_ratio_median": _median(
                selected, "model_to_duty_null_loss_ratio"
            ),
        }
    replay_accepted = all(
        evidence["accepted"]
        for rate in replay.values()
        for evidence in rate.values()
    )
    heldout_timing = [row for row in timings if row["split"] == "evaluation"]
    timing_fields = (
        "prefill_medabs_pct",
        "decode_ttft_medabs_pct",
        "decode_duration_medabs_pct",
        "e2e_medabs_pct",
        "prefill_measured_median_ms",
        "prefill_predicted_median_ms",
        "decode_ttft_measured_median_ms",
        "decode_ttft_predicted_median_ms",
        "decode_duration_measured_median_ms",
        "decode_duration_predicted_median_ms",
        "e2e_measured_median_ms",
        "e2e_predicted_median_ms",
    )
    campaign_hash, hashed_files = campaign_sha256(run_root)
    report = {
        "schema_version": "powertrace-disaggregated-confirmation-v1",
        "evidence_role": "prospective cache-disabled minimal calibration",
        "campaign": {
            "path": str(run_root.relative_to(ROOT)),
            "sha256": campaign_hash,
            "hashed_files": hashed_files,
            "cells": len(specs),
            "requests": sum(int(spec["prompts"]) for spec in specs.values()),
            "measurement_gates_passed": True,
        },
        "release": {
            "path": str(DEFAULT_ARTIFACT.relative_to(ROOT)),
            "sha256": sha256(DEFAULT_ARTIFACT),
            "release_status": release["release_status"],
        },
        "model_adjustment": {
            "base_timing_coefficients_changed": False,
            "base_power_coefficients_changed": False,
            "target_scalars": 4,
            "fit_cell": calibration_cell,
            "parameters": calibration,
            "fit_loss": "pointwise squared error in watts per role",
        },
        "heldout_role_acceptance": {
            "thresholds": thresholds,
            "roles": role_summary,
            "accepted": all(
                summary["accepted"] for summary in role_summary.values()
            ),
        },
        "measured_replay_acceptance": {
            "threshold": replay_threshold,
            "loads": replay,
            "accepted": replay_accepted,
        },
        "timing": {
            "heldout_cell_medians": {
                field: _median(heldout_timing, field)
                for field in timing_fields
            },
            "method": "request-ID-paired proxy boundaries; no DTW",
            "per_cell": timings,
        },
        "calibration_evidence": evidence,
        "analysis_contract": metadata["analysis_protocol"],
        "per_cell": rows,
        "outputs": {
            "per_cell_metrics": str(METRICS.relative_to(ROOT)),
            "representative_traces": str(TRACES.relative_to(ROOT)),
            "figure": str(FIGURE.relative_to(ROOT)),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(METRICS, rows)
    write_csv(TRACES, trace_rows)
    plot_confirmation_traces(
        trace_rows, calibration, FIGURE, cell=FIGURE_CELL
    )
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote confirmation outputs; prefill accepted="
        f"{role_summary['prefill']['accepted']}, decode accepted="
        f"{role_summary['decode']['accepted']}, replay accepted="
        f"{replay_accepted}"
    )


if __name__ == "__main__":
    main()
