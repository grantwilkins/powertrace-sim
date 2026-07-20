"""
Claim:
The no-dose timing diagnostic subtracts only past and current power samples,
and derives request completion rate from arrivals minus the active-request
change using the ledger's count-versus-rate units.

Plausible wrong implementations:
- Use a centered mean that leaks future power across the transition.
- Divide the active-request change by dt in the wrong direction.
- Add rather than subtract the active-request change.
- Change the detrended signal when a constant power offset is added.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from timing_residual_diagnostic import causal_detrend, request_completion_rate


def test_causal_detrend_is_hand_checkable_and_offset_invariant():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])

    residual = causal_detrend(values, window_bins=2)
    shifted = causal_detrend(values + 100.0, window_bins=2)

    np.testing.assert_allclose(residual, [0.0, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(shifted, residual)


def test_causal_detrend_does_not_read_future_samples():
    prefix = causal_detrend(np.asarray([1.0, 4.0, 2.0]), window_bins=3)
    changed_future = causal_detrend(
        np.asarray([1.0, 4.0, 2.0, 1000.0]), window_bins=3
    )

    np.testing.assert_allclose(changed_future[:3], prefix)


def test_request_completion_rate_has_correct_sign_and_units():
    arrivals_per_s = np.asarray([2.0, 0.0, 1.0])
    delta_active = np.asarray([0.0, -1.0, 0.0])

    completions = request_completion_rate(
        arrivals_per_s, delta_active, dt_s=0.5
    )

    np.testing.assert_allclose(completions, [2.0, 2.0, 1.0])
