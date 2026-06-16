"""Loader length-budget behavior (runs in the full env — imports benchmark_dataset).

Confirms the pruning caps are flexible: default 1024/2048, but track the served
context window once configured, so long-context prompts are kept for big-context
models instead of being dropped at 1024.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client

import benchmark_dataset as bd  # noqa: E402


def _reset():
    bd._LENGTH_BUDGET["max_prompt_len"] = 1024
    bd._LENGTH_BUDGET["max_total_len"] = 2048


def test_default_caps_match_historical_behavior():
    _reset()
    assert bd.is_valid_sequence(800, 800)        # within 1024/2048
    assert not bd.is_valid_sequence(1500, 10)    # prompt > 1024 -> dropped (old behavior)


def test_configure_tracks_served_context_window():
    _reset()
    bd.configure_length_budget(32768)
    assert bd.is_valid_sequence(8000, 200)       # long prompt now kept
    assert bd.is_valid_sequence(30000, 1000)     # prompt+output <= 32768 kept
    assert not bd.is_valid_sequence(40000, 10)   # beyond the context -> still dropped
    _reset()


def test_configure_none_is_noop():
    _reset()
    bd.configure_length_budget(None)
    assert not bd.is_valid_sequence(1500, 10)    # unchanged from default
    _reset()
