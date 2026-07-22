"""
Claim:
B2 trains per configuration from repeat 0, early-stops on repeat 1, and predicts
repeat 2 deterministically on its unchanged native grid using exact A/delta.

Plausible wrong implementations:
- Normalize or fit the GMM with repeat-2 target values.
- Pool configurations into one checkpoint.
- Carry a recurrent sequence across run boundaries or change output length.
- Permit the per-configuration baseline to make a transfer prediction.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

FEATURE_TEST = Path(__file__).resolve().parents[1] / "feature_test"
sys.path.insert(0, str(FEATURE_TEST))

from gmm_bigru_baseline import fit_s0_gmm_bigru, predict_s0_gmm_bigru


def _tiny_ledger():
    n = 16
    run_ids = np.repeat([0, 1, 2], n)
    activity = np.tile(np.r_[np.zeros(4), np.ones(8), np.zeros(4)], 3)
    delta = np.concatenate([np.r_[0, np.diff(activity[i*n:(i+1)*n])] for i in range(3)])
    power = 100 + 40 * activity
    return run_ids, np.full(3*n, "config-a"), activity, delta, power


def test_b2_is_deterministic_target_blind_and_grid_aligned():
    run_ids, config, activity, delta, power = _tiny_ledger()
    fit = fit_s0_gmm_bigru(
        run_ids, config, activity, delta, power, {0}, {1},
        k=2, hidden_dim=4, epochs=2, patience=1, seed=7,
    )
    changed = power.copy()
    changed[run_ids == 2] = 1e6
    refit = fit_s0_gmm_bigru(
        run_ids, config, activity, delta, changed, {0}, {1},
        k=2, hidden_dim=4, epochs=2, patience=1, seed=7,
    )
    np.testing.assert_allclose(fit["configurations"]["config-a"]["gmm"]["means"],
                               refit["configurations"]["config-a"]["gmm"]["means"])
    np.testing.assert_allclose(fit["configurations"]["config-a"]["feature_mean"],
                               refit["configurations"]["config-a"]["feature_mean"])
    for name, value in fit["configurations"]["config-a"]["model"].state_dict().items():
        np.testing.assert_allclose(
            value.detach().numpy(),
            refit["configurations"]["config-a"]["model"].state_dict()[name].detach().numpy(),
        )

    first = predict_s0_gmm_bigru(run_ids, config, activity, delta, {2}, fit)
    second = predict_s0_gmm_bigru(run_ids, config, activity, delta, {2}, fit)
    np.testing.assert_allclose(first[run_ids == 2], second[run_ids == 2])
    assert np.all(np.isnan(first[run_ids != 2]))
    assert first[run_ids == 2].size == power[run_ids == 2].size
    assert fit["parameter_count"] > 0
    assert fit["device"] == "cpu"
    assert fit["transfer_eligible"] is False


def test_b2_rejects_transfer_and_unseen_configuration():
    run_ids, config, activity, delta, power = _tiny_ledger()
    with pytest.raises(ValueError, match="ineligible for transfer"):
        fit_s0_gmm_bigru(run_ids, config, activity, delta, power, {0}, {1},
                         k=2, hidden_dim=4, epochs=1, transfer=True)
    fit = fit_s0_gmm_bigru(run_ids, config, activity, delta, power, {0}, {1},
                           k=2, hidden_dim=4, epochs=1, patience=1)
    changed_config = config.copy()
    changed_config[run_ids == 2] = "unseen"
    with pytest.raises(ValueError, match="unseen configuration"):
        predict_s0_gmm_bigru(run_ids, changed_config, activity, delta, {2}, fit)


def test_b2_parallel_configs_remain_separate():
    run_ids, config, activity, delta, power = _tiny_ledger()
    args = (np.r_[run_ids, run_ids + 3], np.r_[config, np.full(config.size, "config-b")],
            np.tile(activity, 2), np.tile(delta, 2), np.tile(power, 2), {0, 3}, {1, 4})
    fit = fit_s0_gmm_bigru(*args, k=2, hidden_dim=4, epochs=1, patience=1)
    repeated = fit_s0_gmm_bigru(*args, k=2, hidden_dim=4, epochs=1, patience=1)
    assert set(fit["configurations"]) == {"config-a", "config-b"}
    for config_id in fit["configurations"]:
        for name, value in fit["configurations"][config_id]["model"].state_dict().items():
            np.testing.assert_array_equal(
                value, repeated["configurations"][config_id]["model"].state_dict()[name])
