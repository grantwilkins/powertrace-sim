"""
Claim:
A phase-aware routing law computes one mixed iteration's expert union and
weight bytes, while dense iterations remain exactly one full weight sweep.

Plausible wrong implementations:
- Add prefill and decode sweeps instead of taking their expert union.
- Normalize top-k probabilities twice or average at the token level.
- Apply the MoE routing fraction to shared weights.
- Change dense or zero-token accounting.
"""

import numpy as np
import pytest

from model.training_data.moe_routing import (
    RoutingLaw,
    expected_iteration_weight_bytes,
)


def _law() -> RoutingLaw:
    probability = np.array([[0.5, 0.5]])
    return RoutingLaw(
        model="toy", source="toy", top_k=1, n_experts=2,
        prefill_alpha=1.0, decode_alpha=1.0,
        prefill_touch_probability=probability,
        decode_touch_probability=probability,
    )


def test_mixed_iteration_uses_expert_union_once():
    arch = {"w_bytes": 100.0, "moe_frac": 0.5,
            "n_experts": 2, "top_k": 1}
    prefill = expected_iteration_weight_bytes(
        arch, prefill_tokens=1, routing_law=_law())
    mixed = expected_iteration_weight_bytes(
        arch, prefill_tokens=1, decode_tokens=1, routing_law=_law())

    assert prefill == pytest.approx(75.0)
    assert mixed == pytest.approx(87.5)
    assert mixed < 2.0 * prefill


def test_shared_weights_are_never_routed_away():
    arch = {"w_bytes": 100.0, "moe_frac": 0.5,
            "n_experts": 2, "top_k": 1}
    value = expected_iteration_weight_bytes(
        arch, decode_tokens=1000, routing_law=_law())
    assert value == pytest.approx(100.0)


def test_dense_and_zero_token_semantics_are_exact():
    dense = {"w_bytes": 100.0, "moe_frac": 0.0,
             "n_experts": 1, "top_k": 1}
    assert expected_iteration_weight_bytes(dense, decode_tokens=4) == 100.0
    assert expected_iteration_weight_bytes(dense) == 0.0
