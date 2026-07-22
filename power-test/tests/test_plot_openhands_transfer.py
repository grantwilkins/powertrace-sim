"""
Claim:
The OpenHands appendix applies the declared platform-calibration equation at
the node level without changing workload timing or double-scaling TP.

Plausible wrong implementations:
- Add the source idle instead of replacing it with the target idle.
- Apply the ordinary gain to the full source power, including idle.
- Apply the extra prefill gain with the wrong sign.
- Multiply an already node-level contribution by TP again.
"""

import numpy as np

from plot_openhands_transfer import (
    EXTRA_PREFILL_GAIN,
    ORDINARY_DYNAMIC_GAIN,
    SOURCE_IDLE_W_PER_GPU,
    TARGET_IDLE_W_PER_GPU,
    platform_calibrate,
)


def test_idle_is_replaced_once_at_node_level():
    source = np.asarray([2.0 * SOURCE_IDLE_W_PER_GPU])
    result = platform_calibrate(source, np.asarray([0.0]), tp=2)
    np.testing.assert_allclose(result, [2.0 * TARGET_IDLE_W_PER_GPU])


def test_dynamic_and_prefill_terms_follow_declared_equation():
    tp = 2
    ordinary_dynamic_node = 20.0
    prefill_node = 8.0
    source = np.asarray([tp * SOURCE_IDLE_W_PER_GPU + ordinary_dynamic_node])
    result = platform_calibrate(source, np.asarray([prefill_node]), tp=tp)
    expected = (
        tp * TARGET_IDLE_W_PER_GPU
        + ORDINARY_DYNAMIC_GAIN * ordinary_dynamic_node
        + EXTRA_PREFILL_GAIN * prefill_node
    )
    np.testing.assert_allclose(result, [expected])
