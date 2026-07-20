"""Claim: cache comparisons use identical token-level work, not just lengths.

Plausible wrong implementations caught here: positional comparison after global
completion order changes, accepting duplicate/missing turn keys, pairing
different normalized plans, or accepting equal lengths with different token IDs.
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
        "model": "m", "hardware": "A100", "tp": 1, "arch": {"x": 1},
        "versions": {"vllm": "1"}, "server": {"max_num_seqs": 8},
        "probe": {
            "trace_plan_sha256": "abc", "prefix_cache": False,
            "decode_constraint": "singleton_allowed_token_v1",
        },
    }
    on = {
        "run_id": "on",
        "model": "m", "hardware": "A100", "tp": 1, "arch": {"x": 1},
        "versions": {"vllm": "1"}, "server": {
            "max_num_seqs": 8, "enable_prefix_caching": True,
        },
        "probe": {
            "trace_plan_sha256": "abc", "prefix_cache": True,
            "decode_constraint": "singleton_allowed_token_v1",
        },
    }
    requests = {
        "session_ids": ["s"], "turn_idx": [0], "input_lens": [8],
        "output_lens": [2], "prefix_tokens": [4], "new_input_tokens": [4],
        "planned_output_tokens": [2], "prompt_sha256": ["p"],
        "output_sha256": ["o"], "source_ids": ["source"],
        "forced_output_token_id": [17], "request_seed": [9],
        "decode_constraint": ["singleton_allowed_token_v1"],
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


def test_pairing_is_keyed_not_global_completion_order():
    off, off_requests, on, on_requests = _pair()
    for requests in (off_requests, on_requests):
        for field, values in list(requests.items()):
            requests[field] = values + [
                1 if field == "turn_idx" else (
                    "t" if field == "session_ids" else values[0]
                )
            ]
    for field in on_requests:
        on_requests[field] = list(reversed(on_requests[field]))
    assert compare_bundle_data(
        off, off_requests, on, on_requests
    )["status"] == "identical"


def test_duplicate_key_and_server_drift_fail():
    off, off_requests, on, on_requests = _pair()
    off_requests["session_ids"].append("s")
    off_requests["turn_idx"].append(0)
    for field in set(off_requests) - {"session_ids", "turn_idx"}:
        off_requests[field].append(off_requests[field][0])
    with pytest.raises(ValueError, match="duplicate replay key"):
        compare_bundle_data(off, off_requests, on, on_requests)

    off, off_requests, on, on_requests = _pair()
    on["server"]["max_num_seqs"] = 9
    with pytest.raises(ValueError, match="server"):
        compare_bundle_data(off, off_requests, on, on_requests)
