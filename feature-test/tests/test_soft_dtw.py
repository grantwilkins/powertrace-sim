"""
Claim:
The reported soft-DTW divergence is a symmetric, range-normalized temporal
error at one-second resolution that is zero for identical traces and permits
only the declared ten-second alignment band.

Plausible wrong implementations:
- Report raw soft-DTW, whose entropic self-cost can be negative.
- Normalize measured and predicted traces independently and hide amplitude error.
- Use an asymmetric recurrence or the wrong predecessor indices.
- Interpret the band in native 250 ms bins instead of one-second bins.
- Attribute amplitude error to timing even when warping cannot reduce it.
"""

import numpy as np

from evaluation_core import (  # noqa: E402
    normalized_soft_dtw_diagnostics,
    soft_dtw_cost,
    soft_dtw_divergence,
    trace_metrics,
)


def test_soft_dtw_divergence_is_zero_for_self_and_symmetric():
    left = np.asarray([0.0, 1.0, 0.0, 2.0])
    right = np.asarray([0.0, 0.5, 1.0, 0.0])

    assert np.isclose(soft_dtw_divergence(left, left, band=2), 0.0)
    assert np.isclose(
        soft_dtw_divergence(left, right, band=2),
        soft_dtw_divergence(right, left, band=2),
    )


def test_alignment_band_reduces_shifted_pulse_cost():
    left = np.asarray([0.0, 1.0, 0.0])
    right = np.asarray([0.0, 0.0, 1.0])

    diagonal = soft_dtw_cost(left, right, gamma=0.01, band=0)
    aligned = soft_dtw_cost(left, right, gamma=0.01, band=1)

    assert np.isclose(diagonal, 2.0)
    assert aligned < diagonal


def test_reported_soft_dtw_preserves_shared_power_unit_changes():
    measured_1s = np.tile(np.asarray([100.0, 120.0, 100.0, 140.0]), 16)
    predicted_1s = np.roll(measured_1s, 1)
    measured = np.repeat(measured_1s, 4)
    predicted = np.repeat(predicted_1s, 4)

    original = trace_metrics(measured, predicted)["soft_dtw_divergence"]
    converted = trace_metrics(
        2.5 * measured + 17.0, 2.5 * predicted + 17.0
    )["soft_dtw_divergence"]

    assert original > 0.0
    assert np.isclose(original, converted)


def test_soft_dtw_diagnostic_separates_warpable_shift_from_amplitude_error():
    measured = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])
    shifted = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0])
    scaled = 2.0 * measured

    shift = normalized_soft_dtw_diagnostics(measured, shifted, band=1)
    amplitude = normalized_soft_dtw_diagnostics(measured, scaled, band=1)

    assert shift["soft_dtw_band_effect"] > 0.0
    assert shift["soft_dtw_divergence"] < shift["soft_dtw_diagonal_divergence"]
    assert np.isclose(amplitude["soft_dtw_band_effect"], 0.0)
    assert np.isclose(
        amplitude["soft_dtw_divergence"],
        amplitude["soft_dtw_diagonal_divergence"],
    )
