"""Claim: routing evidence retains exact layer/token/expert assignments.

Plausible wrong implementations caught here: top-k along the token axis,
discarding the layer axis, or replacing observed expert overlap with independent
uniform draws when computing distinct-expert growth.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router_capture import distinct_expert_curve, topk_expert_ids  # noqa: E402


def test_topk_experts_are_per_token_and_score_ordered():
    logits = np.array([[0.1, 3.0, 2.0], [5.0, 1.0, 4.0]])
    np.testing.assert_array_equal(
        topk_expert_ids(logits, 2), [[1, 2], [0, 2]]
    )


def test_distinct_curve_preserves_layer_overlap():
    ids = np.array([
        [[0, 1], [0, 1]],
        [[0, 1], [2, 3]],
        [[2, 3], [0, 1]],
        [[2, 3], [2, 3]],
    ])
    curve = distinct_expert_curve(ids, [1, 2, 4])
    np.testing.assert_allclose(curve[:, 0], [2, 2])
    np.testing.assert_allclose(curve[:, 1], [2, 4])
    np.testing.assert_allclose(curve[:, 2], [4, 4])
