"""Faithful-replay behaviour of the live session sender (Gap 1).

Drives ``send_session`` against a fake endpoint that streams a reply *different*
from the trace's, and asserts: real text is sent (no ``_filler``), generation is
forced to exact length (``ignore_eos``), and the context grows with the TRACE's
reply — not the live one — so the next prompt matches the real trajectory.
"""

import asyncio
import json

import agentic
import session_driver

# --- a tiny fake aiohttp that streams two content tokens then usage + [DONE] ---
_SSE = [
    b'data: {"choices":[{"delta":{"content":"LIVE"}}]}',
    b'data: {"choices":[{"delta":{"content":"GEN"}}]}',
    b'data: {"usage":{"prompt_tokens":42}}',
    b'data: [DONE]',
]


class _Content:
    def __aiter__(self):
        self._it = iter(_SSE)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Post:
    content = _Content()
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeHttp:
    def __init__(self):
        self.payloads = []

    def post(self, url, json):
        self.payloads.append(json)
        return _Post()


class BoomTok:
    """Any use means the code fell back to synthetic filler — fail loudly."""
    vocab_size = 1000

    def __call__(self, *a, **k):
        raise AssertionError("tokenizer used despite real text")

    def decode(self, *a, **k):
        raise AssertionError("_filler used despite real text")


def _replay_session():
    turns = [
        agentic.SessionTurn(turn_idx=0, new_input_tokens=2, output_tokens=2,
                            post_gap_s=0.0, user_text="hello", assistant_text="TRACE ONE",
                            tool_class="bash", observation_tokens=5),
        agentic.SessionTurn(turn_idx=1, new_input_tokens=2, output_tokens=2,
                            post_gap_s=0.0, user_text="again", assistant_text="TRACE TWO"),
    ]
    return agentic.SessionPlan(session_id="r0", prefix_tokens=3, turns=turns,
                               system_text="SYS REAL")


def test_faithful_replay_injects_trace_reply_and_real_text():
    http = FakeHttp()
    recs = asyncio.run(session_driver.send_session(
        http, "http://x/v1", "m", _replay_session(), prefix_cache=True,
        tokenizer=BoomTok()))

    first = http.payloads[0]
    assert first["ignore_eos"] is True and first["max_tokens"] == 2  # forced length
    assert first["messages"][0] == {"role": "system", "content": "SYS REAL"}
    assert first["messages"][1] == {"role": "user", "content": "hello"}

    # Second turn's context carries the TRACE reply ("TRACE ONE"), not "LIVEGEN".
    second_msgs = http.payloads[1]["messages"]
    assert {"role": "assistant", "content": "TRACE ONE"} in second_msgs
    assert all("LIVEGEN" not in m["content"] for m in second_msgs)

    # Records reflect the LIVE generation (2 tokens, server prompt_tokens) + provenance.
    assert recs[0]["output_len"] == 2 and recs[0]["input_len"] == 42
    assert recs[0]["tool_class"] == "bash" and recs[0]["observation_tokens"] == 5


class _FailPost(_Post):
    status = 400


class FailSecondHttp:
    """Succeeds on turn 1, returns HTTP 400 on turn 2 (e.g. context-length)."""
    def __init__(self):
        self.n = 0

    def post(self, url, json):
        self.n += 1
        return _Post() if self.n == 1 else _FailPost()


def test_failed_turn_stops_session_without_degenerate_record():
    recs = asyncio.run(session_driver.send_session(
        FailSecondHttp(), "http://x/v1", "m", _replay_session(),
        prefix_cache=False, tokenizer=BoomTok()))
    assert len(recs) == 1            # only the successful turn; the 400 turn is dropped
    assert recs[0]["output_len"] == 2


def test_requests_json_includes_provenance_arrays():
    http = FakeHttp()
    recs = asyncio.run(session_driver.send_session(
        http, "http://x/v1", "m", _replay_session(), prefix_cache=False,
        tokenizer=BoomTok()))
    rj = session_driver.build_requests_json(recs)
    assert rj["tool_class"][0] == "bash"
    assert rj["observation_tokens"][0] == 5
    assert len(rj["input_lens"]) == 2  # still a reconstruction-compatible superset
