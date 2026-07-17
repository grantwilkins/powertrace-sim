"""Claim: cache comparisons use identical token-level work, not just lengths.

Plausible wrong implementations caught here: pairing different normalized plans
or accepting equal request lengths even when generated token IDs—and therefore
the next-turn prefix/cache behavior—diverged.
"""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_trace_replays import compare_bundle_data  # noqa: E402


def _pair():
    off = {
        "run_id": "off",
        "probe": {"trace_plan_sha256": "abc", "prefix_cache": False},
    }
    on = {
        "run_id": "on",
        "probe": {"trace_plan_sha256": "abc", "prefix_cache": True},
    }
    requests = {
        "session_ids": ["s"], "turn_idx": [0], "input_lens": [8],
        "output_lens": [2], "prefix_tokens": [4], "new_input_tokens": [4],
        "planned_output_tokens": [2], "prompt_sha256": ["p"],
        "output_sha256": ["o"],
    }
    return off, requests, on, copy.deepcopy(requests)


def test_pair_requires_token_identity():
    off, off_requests, on, on_requests = _pair()
    assert compare_bundle_data(
        off, off_requests, on, on_requests
    )["status"] == "identical"
    on_requests["output_sha256"] = ["different"]
    with pytest.raises(ValueError, match="output_sha256"):
        compare_bundle_data(off, off_requests, on, on_requests)
