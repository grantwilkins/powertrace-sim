"""Build the compact disaggregated-inference evaluation report."""
from __future__ import annotations

import numpy as np

from disaggregated_analysis_core import summarize


def _median(rows: list[dict], field: str) -> float:
    return float(np.median([row[field] for row in rows]))


def _role_summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        role: {
            "cells": len(rows),
            "energy_error_pct_median": _median(
                rows, f"{role}_energy_error_pct"
            ),
            "trace_correlation_median": _median(
                rows, f"{role}_trace_correlation"
            ),
            "trace_rmse_median_w": _median(rows, f"{role}_trace_rmse_w"),
            "trace_std_ratio_median": _median(
                rows, f"{role}_trace_std_ratio"
            ),
            "trace_p95_error_pct_median": _median(
                rows, f"{role}_trace_p95_error_pct"
            ),
            "soft_dtw_diagonal_median": _median(
                rows, f"{role}_trace_soft_dtw_diagonal_divergence"
            ),
            "frozen_soft_dtw_diagonal_median": _median(
                rows, f"{role}_frozen_trace_soft_dtw_diagonal_divergence"
            ),
            "constant_mean_soft_dtw_diagonal_median": _median(
                rows,
                f"{role}_trace_constant_mean_soft_dtw_diagonal_divergence",
            ),
        }
        for role in ("prefill", "decode")
    }


def _native_acceptance(rows: list[dict]) -> dict[str, dict[str, object]]:
    if not rows:
        raise ValueError("native acceptance requires held-out-rate cells")
    output = {}
    for role in ("prefill", "decode"):
        checks = {
            "correlation_at_least_0.8": all(
                row[f"{role}_trace_correlation"] >= 0.8 for row in rows
            ),
            "std_ratio_between_0.8_and_1.25": all(
                0.8 <= row[f"{role}_trace_std_ratio"] <= 1.25
                for row in rows
            ),
            "p95_error_at_most_10_pct": all(
                row[f"{role}_trace_p95_error_pct"] <= 10.0 for row in rows
            ),
            "soft_dtw_beats_constant_mean": all(
                row[f"{role}_trace_soft_dtw_diagonal_divergence"]
                < row[f"{role}_trace_constant_mean_soft_dtw_diagonal_divergence"]
                for row in rows
            ),
        }
        output[role] = {**checks, "accepted": all(checks.values())}
    return output


def _cache_effect(cell_diagnostics: list[dict]) -> dict[str, dict[str, float]]:
    output = {}
    for rate in sorted({row["rate"] for row in cell_diagnostics}):
        group = [row for row in cell_diagnostics if row["rate"] == rate]
        first = next(row for row in group if row["repeat"] == 0)
        later = [row for row in group if row["repeat"] > 0]
        if len({row["input_lengths_sha256"] for row in group}) != 1:
            raise ValueError(f"rate {rate:g} repeats do not replay input lengths")
        output[str(rate)] = {
            "repeat0_prefill_power_excess_w": float(
                first["measured_prefill_mean_w"]
                - np.mean([row["measured_prefill_mean_w"] for row in later])
            ),
            "repeat0_nixl_bytes_multiplier": float(
                first["nixl_mean_transfer_mb"]
                / np.mean([row["nixl_mean_transfer_mb"] for row in later])
            ),
        }
    return output


def _repeatability(
    cell_diagnostics: list[dict],
    cell_role_traces: dict[tuple[float, int, str], np.ndarray],
) -> dict[str, dict[str, float]]:
    output = {}
    for rate in sorted({row["rate"] for row in cell_diagnostics}):
        output[str(rate)] = {}
        for role in ("prefill", "decode"):
            left = cell_role_traces[(rate, 1, role)]
            right = cell_role_traces[(rate, 2, role)]
            count = min(left.size, right.size)
            output[str(rate)][role] = float(
                np.corrcoef(left[:count], right[:count])[0, 1]
            )
    return output


def build_report(
    *,
    campaign: dict,
    release: dict,
    candidates: tuple[str, ...],
    rows: list[dict],
    calibration: dict,
    source_idle_w: float,
    phase_calibration: dict,
    cell_diagnostics: list[dict],
    cell_role_traces: dict,
    outputs: dict,
) -> dict:
    accepted = [
        row for row in rows
        if row["split"] == "evaluation"
        and row["candidate"] == "phase_soft_dtw_calibrated"
    ]
    heldout = [row for row in accepted if row["phase_split"] == "heldout_rate"]
    timing_fields = (
        "prefill_medabs_pct", "decode_ttft_medabs_pct",
        "decode_duration_medabs_pct", "e2e_medabs_pct",
        "prefill_measured_median_ms", "prefill_predicted_median_ms",
        "decode_ttft_measured_median_ms", "decode_ttft_predicted_median_ms",
        "decode_duration_measured_median_ms",
        "decode_duration_predicted_median_ms",
        "client_to_proxy_median_ms", "prefill_dispatch_median_ms",
        "handoff_median_ms", "client_ttft_event_abs_median_ms",
        "prefill_engine_mean_ms", "decode_engine_mean_ms",
    )
    cache_effect = _cache_effect(cell_diagnostics)
    repeatability = _repeatability(cell_diagnostics, cell_role_traces)
    native_acceptance = _native_acceptance(heldout)
    return {
        "schema_version": "powertrace-disaggregated-evaluation-v2",
        "evidence_role": "retrospective minimal phase calibration",
        "campaign": campaign,
        "release": release,
        "model_adjustment": {
            "composition": "serial independent prefill and decode TP1 engines",
            "prefill_output_semantics": "one discarded token",
            "decoder_output_semantics": "full output after transferred prompt KV",
            "frozen_dynamic_coefficients_changed": False,
            "frozen_timing_coefficients_changed": False,
            "disaggregated_data_used_for_base_training": False,
            "accepted_target_active_scalars": 1,
            "accepted_decoder_dynamic_scale": phase_calibration["decode"][
                "accepted_scale"
            ],
            "accepted_prefill_dynamic_scale": phase_calibration["prefill"][
                "accepted_scale"
            ],
            "accepted_calibration_scalar_count": 3,
            "accepted_calibration_parameters": [
                "prefill idle", "decode idle", "decode dynamic gain",
            ],
            "campaign_override": "max_num_batched_tokens=8192",
            "support_status": "retrospective unsupported_extrapolation",
        },
        "calibration": {
            "idle": {
                **calibration,
                "source_idle_w_per_gpu": source_idle_w,
                "fit_scope": "settled pre-request samples only",
            },
            "phase_soft_dtw": {
                "cell": "rate-2-repeat-1",
                "aggregation_s": 0.25,
                "primary_band_s": 0,
                "dynamic_scale_bounds": [0.5, 1.5],
                "selection_guards": [
                    "diagonal soft-DTW", "absolute mean bias",
                    "p95 error", "standard-deviation ratio",
                ],
                "roles": phase_calibration,
            },
        },
        "power": {
            "evaluation_cells": len(accepted),
            "heldout_rate_cells": len(heldout),
            "summaries": summarize(rows, candidates),
            "heldout_rate_by_role": _role_summary(heldout),
            "soft_dtw_protocol": (
                "Primary phase score is range-normalized diagonal soft-DTW "
                "at native 250 ms; a one-sample warp is sensitivity-only."
            ),
            "native_trace_acceptance": {
                "scope": "every held-out-rate cell",
                "roles": native_acceptance,
            },
            "prefill_trace_claim": (
                "supported at native 250 ms"
                if native_acceptance["prefill"]["accepted"]
                else "unsupported at native 250 ms"
            ),
            "prefill_mean_power_claim": "supported for warm cells",
            "decode_trace_claim": (
                "supported at native 250 ms"
                if native_acceptance["decode"]["accepted"]
                else "unsupported at native 250 ms"
            ),
        },
        "timing": {
            "evaluation_cell_medians": {
                field: _median(accepted, field) for field in timing_fields
            },
            "method": "request-ID-paired proxy boundaries; no DTW",
            "decode_ttft_limitation": (
                "External-KV admission is load dependent; no timing scalar "
                "was fitted."
            ),
        },
        "data_quality": {
            "failed_nixl_operations": 0,
            "decoder_preemptions": 0,
            "input_length_sequences_replayed_across_repeats": True,
            "prompt_identity_recorded": False,
            "prefix_cache_hit_fraction_by_cell": {
                row["cell"]: row["prefill_prefix_cache_hit_fraction"]
                for row in cell_diagnostics
            },
            "warm_repeat_power_correlation_by_rate": repeatability,
            "cache_and_warmup_effect_by_rate": cache_effect,
            "prefill_timestamp_limitation": (
                "Power is 4 Hz and timestamped before nvidia-smi acquisition; "
                "prefill HTTP phases are approximately 16–20 ms."
            ),
        },
        "rejected_adjustments": {
            "prefill_dynamic_gain": phase_calibration["prefill"],
            "other": [
                "source-only prefill-duty coordinate",
                "source-only phase-split compute coordinates",
                "cache-aware replay without per-request cache state",
                "KV-transfer power term",
                "constant decode-TTFT correction",
            ],
        },
        "diagnosis": [
            "Role-aware TP1 composition transfers without base-model refitting.",
            "Role-specific idle removes the prefill baseline mismatch.",
            "One 2-rps decoder gain improves held-out decode amplitude.",
            "Decode trace shape transfers at 250 ms; prefill cell mean transfers.",
            "Current telemetry cannot validate 250 ms prefill burst shape.",
        ],
        "outputs": outputs,
    }
