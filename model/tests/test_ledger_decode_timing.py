"""
Claim:
Measured inter-token intervals place each post-first decode completion at its
recorded event time, preserving work and half-open time-bin semantics.

Plausible wrong implementations:
- Spread all requested output tokens uniformly across the request duration.
- Assign an interval ending on a bin edge to the following bin.
- Lose or duplicate work at the reconstruction horizon.
- Use request-level rather than per-iteration context for KV reads.
"""

import numpy as np
import pytest

from model.training_data.ledger_view import schedule_work_rates


ARCH = {
    "n_layers": 1,
    "n_kv": 1,
    "head_dim": 1,
    "d_model": 1,
    "w_bytes": 1.0,
    "moe_frac": 0.0,
}


def _ledger(*, decode_end, itls, edges):
    return schedule_work_rates(
        arrivals=[0.0],
        prefill_starts=[0.0],
        prefill_ends=[0.0],
        decode_ends=[decode_end],
        input_tokens=[10.0],
        output_tokens=[3.0],
        edges=edges,
        arch=ARCH,
        tp=1,
        decode_itls=[itls],
    )


def test_measured_intervals_conserve_iterations_and_kv_context():
    ledger = _ledger(decode_end=1.0, itls=[0.25, 0.75], edges=[0.0, 0.5, 1.0, 1.5])

    np.testing.assert_allclose(ledger["dec_tok"], [2.0, 0.0, 2.0])
    assert np.sum(ledger["dec_tok"]) * 0.5 == pytest.approx(2.0)
    np.testing.assert_allclose(ledger["kv_read"], [88.0, 0.0, 96.0])


def test_event_on_bin_edge_enters_the_half_open_bin_to_its_right():
    ledger = _ledger(decode_end=1.0, itls=[0.5, 0.5], edges=[0.0, 0.5, 1.0, 1.5])

    np.testing.assert_allclose(ledger["dec_tok"], [0.0, 2.0, 2.0])


@pytest.mark.parametrize("itls", ([0.5, 0.0], [0.5, float("nan")], [0.4, 0.5]))
def test_invalid_or_inconsistent_measured_intervals_fail(itls):
    message = "finite and positive" if not np.isfinite(itls).all() or min(itls) <= 0 else "duration"
    with pytest.raises(ValueError, match=message):
        _ledger(decode_end=1.0, itls=itls, edges=[0.0, 0.5, 1.0])


def test_measured_interval_count_must_match_post_first_decode_steps():
    with pytest.raises(ValueError, match="output tokens minus one"):
        _ledger(decode_end=1.0, itls=[1.0], edges=[0.0, 0.5, 1.0])
