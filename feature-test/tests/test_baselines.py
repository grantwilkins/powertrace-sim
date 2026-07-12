"""
Claim:
B1 uses only causal exact activity histories with run resets and train-only scaling.
B4 is a per-configuration physics oracle that cannot support transfer claims.

Plausible wrong implementations:
- A history tap reads the future or crosses a run boundary.
- Held-out activity changes B1 normalization or fitted coefficients.
- B4 pools coefficients across configurations.
- B4 silently predicts a configuration absent from its fit.
"""

import numpy as np
import pytest

from baselines import (
    fit_causal_activity_ridge,
    fit_same_configuration_physics_oracle,
    predict_causal_activity_ridge,
    predict_same_configuration_physics_oracle,
)


def test_b1_is_causal_resets_runs_and_ignores_heldout_scaling():
    runs = np.repeat([0, 1, 2], 5)
    activity = np.array([0, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 1, 2, 1, 0], dtype=float)
    delta = np.concatenate([[0], np.diff(activity[:5]), [0], np.diff(activity[5:10]), [0], np.diff(activity[10:])])
    power = 100 + 7 * np.log1p(activity) + 3 * delta
    fit = fit_causal_activity_ridge(runs, activity, delta, power, {0, 1}, dt_s=1, ridge=1e-6,
                                    taps_s=(0, 1))
    changed = activity.copy()
    changed[runs == 2] = 1e6
    refit = fit_causal_activity_ridge(runs, changed, delta, power, {0, 1}, dt_s=1, ridge=1e-6,
                                      taps_s=(0, 1))
    np.testing.assert_allclose(fit["coefficients"], refit["coefficients"])
    np.testing.assert_allclose(fit["history_mean"], refit["history_mean"])

    predicted = predict_causal_activity_ridge(runs, activity, delta, fit)
    future_changed = activity.copy()
    future_changed[4] = 1000
    changed_prediction = predict_causal_activity_ridge(runs, future_changed, delta, fit)
    np.testing.assert_allclose(predicted[:4], changed_prediction[:4])

    first = predict_causal_activity_ridge(np.array([1]), np.array([0.0]), np.array([0.0]), fit)
    prefixed = predict_causal_activity_ridge(
        np.array([0, 0, 1]), np.array([100.0, 100.0, 0.0]), np.zeros(3), fit
    )
    np.testing.assert_allclose(first[0], prefixed[-1])


def test_b4_fits_each_configuration_and_rejects_transfer():
    design = np.array([[1.0], [2.0], [1.0], [2.0]])
    power = np.array([2.0, 4.0, 5.0, 10.0])
    tp = np.ones(4)
    busy = np.ones(4, dtype=bool)
    runs = np.arange(4)
    configs = np.array(["a", "a", "b", "b"])
    fit = fit_same_configuration_physics_oracle(
        design, power, tp, busy, runs, configs, set(runs), cap_quantile=1.0
    )
    predicted = predict_same_configuration_physics_oracle(design, tp, configs, fit)
    np.testing.assert_allclose(predicted, power)
    assert fit["transfer_eligible"] is False

    with pytest.raises(ValueError, match="ineligible for transfer"):
        fit_same_configuration_physics_oracle(
            design, power, tp, busy, runs, configs, set(runs), transfer=True
        )
    with pytest.raises(ValueError, match="unseen configuration"):
        predict_same_configuration_physics_oracle(
            np.array([[1.0]]), np.ones(1), np.array(["unseen"]), fit
        )
