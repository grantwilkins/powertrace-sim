"""
Claim:
The meter-kernel identification recovers a known boxcar+EMA response from
clean synthetic steps, ignores runs' non-step activity, and rejects starved
event sets.

Plausible wrong implementations:
- Fit on raw power levels instead of normalized step shapes.
- Accept windows contaminated by a second idle/busy transition.
- Let the sub-bin phase nuisance absorb the whole moving-average window.
"""

import numpy as np

from identify_meter_kernel import extract_steps, fit_kernel, kernel_step_response


def _synthetic_stream(window_bins: int, alpha: float, *, n_steps=40, gap=60):
    """Idle/busy square wave passed through boxcar(window)+EMA(alpha)."""
    rng = np.random.default_rng(20260711)
    work = np.zeros(n_steps * gap)
    for k in range(n_steps):
        start = k * gap + 20
        work[start:start + 25] = 1.0
    true_power = 100.0 + 300.0 * work
    padded = np.r_[np.full(window_bins - 1, true_power[0]), true_power]
    csum = np.r_[0.0, np.cumsum(padded)]
    metered = (csum[window_bins:] - csum[:-window_bins]) / window_bins
    out = np.empty_like(metered)
    out[0] = metered[0]
    for i in range(1, metered.size):
        out[i] = alpha * metered[i] + (1.0 - alpha) * out[i - 1]
    out += rng.normal(0.0, 0.5, out.size)
    return out, work


def test_identification_recovers_a_known_kernel():
    power, work = _synthetic_stream(window_bins=4, alpha=1.0)
    run_ids = np.zeros(power.size, dtype=int)
    steps = extract_steps(power, work, run_ids)
    assert steps.shape[0] >= 30
    fitted = fit_kernel(np.median(steps, axis=0), 0.25)
    assert fitted["moving_average_s"] == 1.0
    assert fitted["ema_alpha"] >= 0.9


def test_identification_recovers_a_short_window_kernel():
    power, work = _synthetic_stream(window_bins=1, alpha=0.9)
    run_ids = np.zeros(power.size, dtype=int)
    steps = extract_steps(power, work, run_ids)
    fitted = fit_kernel(np.median(steps, axis=0), 0.25)
    assert fitted["moving_average_s"] == 0.25
    assert fitted["ema_alpha"] >= 0.8


def test_contaminated_windows_are_rejected():
    power, work = _synthetic_stream(window_bins=4, alpha=1.0)
    # A blip inside the pre-onset window disqualifies that event.
    work_blip = work.copy()
    onsets = np.flatnonzero(np.diff((work > 0).astype(int)) == 1) + 1
    work_blip[onsets[0] - 4] = 1.0
    run_ids = np.zeros(power.size, dtype=int)
    clean = extract_steps(power, work, run_ids)
    contaminated = extract_steps(power, work_blip, run_ids)
    assert contaminated.shape[0] < clean.shape[0]


def test_step_response_model_is_causal_and_normalized():
    response = kernel_step_response(4, 1.0, 0.0)
    assert np.all(response[:8] == 0.0)
    np.testing.assert_allclose(response[-1], 1.0, atol=1e-9)
    partial = kernel_step_response(1, 1.0, 0.25)
    np.testing.assert_allclose(partial[8], 0.75)
