"""Claim: exact replay preserves prefix identity and both arrival constraints.

Plausible wrong implementations caught here: regenerating a prefix each round,
letting a semaphore replace release times, and counting cached or reasoning
tokens outside the server's authoritative prompt/completion totals.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_replay_driver import (  # noqa: E402
    TokenSession, _usage_details, deterministic_tokens,
    validate_exact_accounting,
)
from trace_replay_runner import round_release_epoch  # noqa: E402


def _row(index, prefix, input_tokens=2, output_tokens=3):
    return SimpleNamespace(
        session_id="s", round_idx=index, prefix_tokens=prefix,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cached_prefix_tokens=prefix,
    )


def test_seeded_prompt_reuses_exact_prior_prefix():
    session = TokenSession("s", vocab_size=100, seed=7)
    first = session.prompt(_row(0, 4))
    session.commit(_row(0, 4), first, [90, 91, 92])
    second = session.prompt(_row(1, 5))
    assert second[:5] == session.history[:5]
    assert first[:4] == deterministic_tokens("s:prefix", 4, 100, 7)
    assert len(second) == 7


def test_release_requires_arrival_and_closed_loop_readiness():
    assert round_release_epoch(100.0, 4.0, 102.0) == 104.0
    assert round_release_epoch(100.0, 4.0, 109.0) == 109.0


def test_usage_accounting_is_bounded_and_complete():
    assert _usage_details({
        "prompt_tokens": 20, "completion_tokens": 8,
        "prompt_tokens_details": {"cached_tokens": 12},
        "completion_tokens_details": {"reasoning_tokens": 6},
    }) == (20, 8, 12, 6)
    with pytest.raises(ValueError, match="cached/reasoning"):
        _usage_details({
            "prompt_tokens": 20, "completion_tokens": 8,
            "completion_tokens_details": {"reasoning_tokens": 9},
        })
    validate_exact_accounting(
        _row(1, 32), planned_prompt_tokens=34, prompt_tokens=34,
        completion_tokens=3, cached_tokens=16, prefix_cache=True,
        cache_block_tokens=16,
    )
    with pytest.raises(ValueError, match="cached_tokens"):
        validate_exact_accounting(
            _row(1, 64), planned_prompt_tokens=66, prompt_tokens=66,
            completion_tokens=3, cached_tokens=0, prefix_cache=True,
            cache_block_tokens=16,
        )
