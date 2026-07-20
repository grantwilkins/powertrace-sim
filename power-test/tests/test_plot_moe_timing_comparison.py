"""
Claim:
The timing comparison uses the same run and measured power in both caches.

Plausible wrong implementations:
- Compare different repetitions from the same model/TP/rate cell.
- Treat unequal trace horizons as different clocks.
- Plot two different measured traces as one ground truth.
"""

import numpy as np
import pytest

from plot_moe_timing_comparison import (
    common_ground_truth,
    representative_run_id,
)


def _cache(run_ids):
    return {
        "model_names": np.array(["gpt-oss-20b"]),
        "model_idx": np.zeros(len(run_ids), dtype=int),
        "tp": np.ones(len(run_ids)),
        "rate": np.full(len(run_ids), 4.0),
        "run_id": np.asarray(run_ids),
    }


def test_representative_run_is_the_lowest_shared_repetition():
    uniform = _cache([10, 10, 12, 12])
    routing = _cache([11, 11, 12, 12])
    assert representative_run_id(
        uniform, routing, "gpt-oss-20b", 1, 4.0) == 12


def test_common_ground_truth_allows_only_horizon_truncation():
    expected = common_ground_truth(
        np.array([1.0, np.nan, 3.0, 4.0]),
        np.array([1.0, np.nan, 3.0]),
    )
    np.testing.assert_allclose(expected, [1.0, np.nan, 3.0], equal_nan=True)

    with pytest.raises(ValueError, match="same ground truth"):
        common_ground_truth(np.array([1.0, 2.0]), np.array([1.0, 2.1]))
