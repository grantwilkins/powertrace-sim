"""
Claim:
Per-role trace diagnostics measure temporal agreement and amplitude at the
one-GPU level, and a phase gain minimizes diagonal soft-DTW without scaling
idle or pairing latency measurements across requests. Confirmatory calibration
fits one nonnegative role gain by pointwise loss, compares against a
same-parameter duty null, and pairs measured replays without interpolation.

Plausible wrong implementations:
- Report mean-power error as a temporal metric.
- Compute the standard-deviation ratio in the wrong direction.
- Aggregate prefill and decode before measuring role fidelity.
- Treat a constant prediction as positively correlated with a varying trace.
- Scale total power instead of dynamic power above each role's idle.
- Use a warped soft-DTW path as the primary phase objective.
- Pair latency rows by position instead of request ID.
- Average or interpolate native power samples instead of preserving timestamps.
- Assign a sample on a bin boundary to the preceding 250 ms interval.
- Fit a negative gain or let the null use more parameters than the model.
- Average duplicate meter samples while aligning repeated workloads.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "power-test"), str(ROOT / "feature-test")]

from disaggregated_analysis_core import (
    apply_phase_calibration,
    campaign_sha256,
    fit_diagonal_soft_dtw_scale,
    role_trace_metrics,
    timing_metrics,
)
from disaggregated_confirmation import (
    average_trace_rows,
    fit_nonnegative_gain,
    fit_positive_time_scale,
    pair_common_bin_samples,
    pointwise_loss_ratio,
    role_acceptance,
)
from disaggregated_native import sample_native_power
from disaggregated_reporting import _native_acceptance, write_csv
from analyze_disaggregated_inference import _run_root


def test_pilot_analysis_ignores_confirmatory_campaign(tmp_path):
    pilot = tmp_path / "gpt-oss-20b-a100-pd-123"
    confirmation = tmp_path / "gpt-oss-20b-a100-pd-confirmatory-456"
    pilot.mkdir()
    confirmation.mkdir()
    (pilot / "run_metadata.json").write_text("{}")
    (confirmation / "run_metadata.json").write_text("{}")

    assert _run_root(tmp_path) == pilot


def test_trace_averaging_uses_fixed_nonoverlapping_bins():
    rows = [
        {"cell": "a", "time_s": 0.1, "measured_prefill_w": 10.0},
        {"cell": "a", "time_s": 0.9, "measured_prefill_w": 30.0},
        {"cell": "a", "time_s": 1.0, "measured_prefill_w": 50.0},
    ]

    averaged = average_trace_rows(rows, bin_s=1.0)

    assert averaged == [
        {"cell": "a", "time_s": 0.5, "measured_prefill_w": 20.0},
        {"cell": "a", "time_s": 1.5, "measured_prefill_w": 50.0},
    ]


def test_generated_csv_uses_repository_native_lf(tmp_path):
    path = tmp_path / "result.csv"

    write_csv(path, [{"value": 1}, {"value": 2}])

    assert path.read_bytes() == b"value\n1\n2\n"


def test_confirmation_hash_binds_workload_boundaries(tmp_path):
    (tmp_path / "run_metadata.json").write_text("{}")
    (tmp_path / "proxy_events.jsonl").write_text("{}\n")
    cell = tmp_path / "rate-2-repeat-0"
    cell.mkdir()
    for name in (
        "requests.json",
        "power.csv",
        "engine_prefill.csv",
        "engine_decode.csv",
        "start_epoch_s",
        "end_epoch_s",
        "workload_start_epoch_s",
        "workload_end_epoch_s",
    ):
        (cell / name).write_text(name)

    before, count = campaign_sha256(tmp_path)
    (cell / "workload_start_epoch_s").write_text("changed")
    after, _ = campaign_sha256(tmp_path)

    assert count == 10
    assert before != after


def test_correct_mean_does_not_hide_missing_trace_variation():
    measured = np.asarray([0.0, 2.0, 0.0, 2.0])
    predicted = np.ones(4)
    metrics = role_trace_metrics(measured, predicted)

    assert predicted.mean() == measured.mean()
    assert metrics["rmse_w"] == 1.0
    assert metrics["std_ratio"] == 0.0
    assert np.isnan(metrics["correlation"])


def test_trace_scale_and_tail_metrics_have_the_claimed_direction():
    measured = np.asarray([1.0, 2.0, 3.0, 4.0])
    metrics = role_trace_metrics(measured, 2.0 * measured)

    assert metrics["correlation"] == 1.0
    assert metrics["std_ratio"] == 2.0
    assert metrics["rmse_w"] == np.sqrt(7.5)
    assert metrics["p95_error_pct"] == 100.0


def test_phase_gain_minimizes_diagonal_loss_without_scaling_idle():
    measured = np.asarray([10.0, 12.5, 10.0, 12.5])
    source = np.asarray([8.0, 10.0, 8.0, 10.0])

    scale = fit_diagonal_soft_dtw_scale(
        measured,
        source,
        measured_idle_w=10.0,
        predicted_idle_w=8.0,
    )
    calibrated = apply_phase_calibration(
        source,
        source_idle_w=8.0,
        target_idle_w=10.0,
        dynamic_scale=scale,
    )

    assert scale == 1.25
    np.testing.assert_allclose(calibrated, measured)
    assert role_trace_metrics(
        measured, calibrated
    )["soft_dtw_diagonal_divergence"] == 0.0


def test_primary_soft_dtw_does_not_warp_a_shifted_phase_trace():
    measured = np.asarray([10.0, 20.0, 10.0, 10.0])
    shifted = np.asarray([10.0, 10.0, 20.0, 10.0])

    metrics = role_trace_metrics(measured, shifted)

    assert metrics["soft_dtw_diagonal_divergence"] > 0.0
    assert (
        metrics["soft_dtw_one_sample_divergence"]
        < metrics["soft_dtw_diagonal_divergence"]
    )


def test_latency_metrics_pair_proxy_boundaries_by_request_id():
    request = {
        "request_ids": ["second", "first"],
        "request_timestamps": [20.0, 10.0],
        "ttfts": [0.7, 0.6],
    }
    events = {
        "first": {
            "proxy_received": 10.1,
            "prefill_sent": 10.2,
            "prefill_completed": 10.4,
            "decode_sent": 10.5,
            "decode_first_byte": 10.6,
            "decode_completed": 11.1,
        },
        "second": {
            "proxy_received": 20.1,
            "prefill_sent": 20.2,
            "prefill_completed": 20.6,
            "decode_sent": 20.6,
            "decode_first_byte": 20.7,
            "decode_completed": 21.7,
        },
    }
    predicted = [
        {
            "prefill_s": 0.4,
            "decode_ttft_s": 0.1,
            "decode_duration_s": 1.0,
            "e2e_s": 1.6,
        },
        {
            "prefill_s": 0.2,
            "decode_ttft_s": 0.1,
            "decode_duration_s": 0.5,
            "e2e_s": 0.9,
        },
    ]

    metrics = timing_metrics(request, predicted, events)

    assert np.isclose(metrics["prefill_measured_median_ms"], 300.0)
    assert np.isclose(metrics["client_ttft_event_abs_median_ms"], 0.0)


def test_native_samples_use_half_open_bins_without_averaging_or_interpolation():
    timestamps = np.asarray([99.9, 100.0, 100.249, 100.25, 100.5])
    measured = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    time_s, native, predicted = sample_native_power(
        timestamps,
        measured,
        {"phase": np.asarray([10.0, 20.0])},
        start_epoch_s=100.0,
        dt_s=0.25,
    )

    np.testing.assert_allclose(time_s, [0.0, 0.249, 0.25])
    np.testing.assert_allclose(native, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(predicted["phase"], [10.0, 10.0, 20.0])


def test_native_acceptance_keeps_prefill_and_decode_independent():
    row = {}
    for role, correlation, std_ratio, p95, loss, constant in (
        ("prefill", 0.79, 1.0, 5.0, 0.1, 0.2),
        ("decode", 0.9, 1.1, 5.0, 0.1, 0.2),
    ):
        row.update({
            f"{role}_trace_correlation": correlation,
            f"{role}_trace_std_ratio": std_ratio,
            f"{role}_trace_p95_error_pct": p95,
            f"{role}_trace_soft_dtw_diagonal_divergence": loss,
            f"{role}_trace_constant_mean_soft_dtw_diagonal_divergence": constant,
        })

    acceptance = _native_acceptance([row])

    assert acceptance["prefill"]["accepted"] is False
    assert acceptance["decode"]["accepted"] is True


def test_confirmatory_gain_is_pointwise_nonnegative_with_fixed_idle():
    measured = np.asarray([10.0, 12.0, 14.0])
    predictor = np.asarray([0.0, 1.0, 2.0])

    assert fit_nonnegative_gain(
        measured, predictor, measured_idle_w=10.0
    ) == 2.0
    assert fit_nonnegative_gain(
        np.asarray([10.0, 9.0, 8.0]),
        predictor,
        measured_idle_w=10.0,
    ) == 0.0


def test_exploratory_time_scale_preserves_fixed_overhead():
    predicted = np.asarray([0.11, 0.21, 0.31])
    measured = 0.01 + 2.0 * (predicted - 0.01)

    assert fit_positive_time_scale(
        measured, predicted, fixed_overhead_s=0.01
    ) == 2.0


def test_confirmatory_acceptance_uses_equal_parameter_pointwise_null():
    measured = np.asarray([10.0, 20.0, 10.0, 20.0])
    predicted = np.asarray([11.0, 19.0, 11.0, 19.0])
    null = np.asarray([14.0, 16.0, 14.0, 16.0])
    ratio = pointwise_loss_ratio(measured, predicted, null)
    checks = role_acceptance(
        {
            "correlation": 1.0,
            "std_ratio": 0.8,
            "p95_error_pct": 5.0,
        },
        ratio,
        {
            "correlation_min": 0.8,
            "std_ratio_min": 0.8,
            "std_ratio_max": 1.25,
            "p95_error_pct_max": 10.0,
            "model_to_duty_null_loss_ratio_max": 0.95,
        },
    )

    assert ratio == 0.0625
    assert checks == {
        "correlation": True,
        "std_ratio": True,
        "p95_error": True,
        "duty_null_loss": True,
        "accepted": True,
    }


def test_replay_pairing_retains_duplicates_without_interpolation():
    left, right = pair_common_bin_samples(
        np.asarray([0, 0, 1, 2]),
        np.asarray([10.0, 20.0, 30.0, 40.0]),
        np.asarray([0, 1, 1, 2]),
        np.asarray([11.0, 31.0, 32.0, 41.0]),
    )

    np.testing.assert_allclose(left, [10.0, 30.0, 40.0])
    np.testing.assert_allclose(right, [11.0, 31.0, 41.0])
