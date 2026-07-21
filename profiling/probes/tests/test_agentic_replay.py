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
    b'data: {"usage":{"prompt_tokens":42,"completion_tokens":2,'
    b'"prompt_tokens_details":{"cached_tokens":16}}}',
    b'data: [DONE]',
]


class _Content:
    def __init__(self, payload):
        token = payload.get("allowed_token_ids", [17])[0]
        active = [
            index for index, raw in enumerate(_SSE)
            if b'"choices"' in raw
        ]
        remaining = int(payload.get("max_tokens", len(active)))
        self.lines = []
        for index, raw in enumerate(_SSE):
            if index in active:
                chunk = json.loads(raw.decode().split("data: ", 1)[1])
                take = 1 if index != active[-1] else remaining
                remaining -= take
                chunk["choices"][0]["logprobs"] = {"content": [
                    {"token": f"token_id:{token}", "logprob": 0.0}
                    for _ in range(take)
                ]}
                raw = f"data: {json.dumps(chunk)}".encode()
            self.lines.append(raw)

    def __aiter__(self):
        self._it = iter(self.lines)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Post:
    status = 200

    def __init__(self, payload):
        self.content = _Content(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeHttp:
    def __init__(self):
        self.payloads = []

    def post(self, url, json):
        self.payloads.append(json)
        return _Post(json)


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
    assert first["return_tokens_as_token_ids"] is True
    assert first["logprobs"] is True
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
        return _Post(json) if self.n == 1 else _FailPost(json)


def test_failed_turn_fails_the_session():
    import pytest

    with pytest.raises(RuntimeError, match="turn 1 failed with HTTP 400"):
        asyncio.run(session_driver.send_session(
            FailSecondHttp(), "http://x/v1", "m", _replay_session(),
            prefix_cache=False, tokenizer=BoomTok()))


def test_requests_json_includes_provenance_arrays():
    http = FakeHttp()
    recs = asyncio.run(session_driver.send_session(
        http, "http://x/v1", "m", _replay_session(), prefix_cache=False,
        tokenizer=BoomTok()))
    rj = session_driver.build_requests_json(recs)
    assert rj["tool_class"][0] == "bash"
    assert rj["observation_tokens"][0] == 5
    assert len(rj["input_lens"]) == 2  # still a reconstruction-compatible superset


def test_missing_zero_cached_tokens_is_recorded_as_zero(monkeypatch):
    usage_without_details = [
        b'data: {"choices":[{"delta":{"content":"LIVE"}}]}',
        b'data: {"choices":[{"delta":{"content":"GEN"}}]}',
        b'data: {"usage":{"prompt_tokens":42,"completion_tokens":2}}',
        b'data: [DONE]',
    ]
    monkeypatch.setattr(__import__(__name__), "_SSE", usage_without_details)
    session = _replay_session()
    session.turns[:] = session.turns[:1]
    records = asyncio.run(session_driver.send_session(
        FakeHttp(), "http://x/v1", "m", session, prefix_cache=True,
        tokenizer=BoomTok()))
    assert records[0]["cached_prompt_tokens"] == 0


def test_reasoning_is_a_completion_subset_not_extra_decode(monkeypatch):
    reasoning_sse = [
        b'data: {"choices":[{"delta":{"reasoning_content":"THINK"}}]}',
        b'data: {"choices":[{"delta":{"content":"ANSWER"}}]}',
        b'data: {"usage":{"prompt_tokens":42,"completion_tokens":5,'
        b'"completion_tokens_details":{"reasoning_tokens":3}}}',
        b'data: [DONE]',
    ]
    monkeypatch.setattr(__import__(__name__), "_SSE", reasoning_sse)
    session = _replay_session()
    session.turns[:] = session.turns[:1]
    session.turns[0] = agentic.SessionTurn(
        turn_idx=0, new_input_tokens=2, output_tokens=5, post_gap_s=0.0,
        user_text="hello", assistant_text="TRACE",
    )
    records = asyncio.run(session_driver.send_session(
        FakeHttp(), "http://x/v1", "m", session, prefix_cache=False,
        tokenizer=BoomTok()))
    assert records[0]["output_len"] == 5
    assert records[0]["reasoning_tokens"] == 3
