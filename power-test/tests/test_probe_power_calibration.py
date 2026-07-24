"""
Claim:
Controlled-probe counters are conserved as request-level average rates paired
with native 250 ms power targets, and each level contributes equal total fitting
weight regardless of duration.

Plausible wrong implementations:
- Place a counter delta wholly in one bin instead of spreading its interval.
- Return tokens per bin while labeling the result tokens per second.
- Lose counter mass at a bin boundary.
- Weight long probe levels more heavily than short levels.
- Treat lumpy counter updates as instantaneous work instead of level averages.
"""

import numpy as np

from build_probe_power_calibration import (
    balanced_level_weights,
    counter_window_rate,
)


def test_counter_window_rate_conserves_deltas_and_rate_units():
    rate = counter_window_rate(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 2.0, 6.0]),
        0.0,
        2.0,
    )

    assert np.isclose(rate, 3.0)
    assert np.isclose(2.0 * rate, 6.0)

    partial = counter_window_rate(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, 2.0, 6.0]),
        0.25,
        1.5,
    )
    assert np.isclose(partial, 2.8)


def test_level_weights_are_invariant_to_level_duration():
    levels = np.asarray(["short", "long", "long", "long"])
    weights = balanced_level_weights(levels, target_per_level=12.0)

    assert np.isclose(weights[levels == "short"].sum(), 12.0)
    assert np.isclose(weights[levels == "long"].sum(), 12.0)
    np.testing.assert_allclose(weights, [12.0, 4.0, 4.0, 4.0])
