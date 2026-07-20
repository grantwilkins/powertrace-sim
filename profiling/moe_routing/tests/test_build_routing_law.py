"""
Claim:
Routing-law normalization measures per-token expert touch probability and
fits a dependence exponent against distinct-expert groups at the layer level.

Plausible wrong implementations:
- Count expert assignments over the layer axis.
- Normalize top-k assignments to sum to one instead of top-k.
- Fit aggregate layers with token-count weighting.
- Report unnormalized load entropy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_routing_law import fit_alpha, phase_record, touch_probability  # noqa: E402


def test_touch_probability_is_per_token_and_layer():
    ids = np.array([
        [[0], [0]],
        [[0], [1]],
        [[1], [1]],
        [[1], [1]],
    ])
    probability = touch_probability(ids, n_experts=2)
    np.testing.assert_allclose(probability[0], [0.5, 0.5])
    np.testing.assert_allclose(probability[1], [0.25, 0.75])


def test_independent_groups_recover_unit_exponent():
    probability = np.array([[0.5, 0.5]])
    sizes = (1, 2, 4)
    observed = np.array([[
        1.0,
        2.0 * (1.0 - 0.5 ** 2),
        2.0 * (1.0 - 0.5 ** 4),
    ]])
    alpha, rmse = fit_alpha(probability, sizes, observed)
    assert alpha == pytest.approx(1.0, abs=1e-4)
    assert rmse < 1e-5


def test_uniform_expert_load_has_unit_normalized_entropy():
    ids = np.array([[[0]], [[1]], [[0]], [[1]]])
    record = phase_record(ids, (1,), np.array([[1.0]]), n_experts=2)
    assert record["normalized_load_entropy"] == pytest.approx([1.0])
