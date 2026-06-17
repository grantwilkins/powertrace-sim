"""Live multi-turn session sender + the rich agentic requests.json (§5-D, planned).

Emits the SAME bundle as the probes. ``requests.json`` is a *superset* of the
probe schema: the five standard arrays (so ``build_ledger_bundle`` reconstructs it
unchanged — note ``itls`` is the scalar TPOT per turn, which ``parse_request_json``
expands via ``itl * (out_tok-1)``) plus the agentic extensions
``session_ids``/``turn_idx``/``post_gap_s``/``prefix_cache``.

Each turn records its **epoch** ``request_timestamp`` (``time.time()``) so it aligns
with the power/engine logs — the same fix vLLM's multi-turn benchmark needs (it uses
monotonic ``perf_counter``). ``turn_record``/``build_requests_json`` are pure and
unit-tested; ``send_session`` is the live layer.
"""

from __future__ import annotations

import time


def turn_record(*, session_id, turn_idx, input_len, output_len, ttft, tpot,
                request_timestamp, post_gap_s, prefix_cache,
                tool_class="", observation_tokens=0) -> dict:
    """One per-turn record (pure)."""
    return {
        "session_id": session_id,
        "turn_idx": int(turn_idx),
        "input_len": int(input_len),
        "output_len": int(output_len),
        "ttft": float(ttft),
        "itl": float(tpot),                 # scalar TPOT (s); see module docstring
        "request_timestamp": float(request_timestamp),
        "post_gap_s": float(post_gap_s),
        "prefix_cache": int(bool(prefix_cache)),
        "tool_class": str(tool_class),          # provenance (replay); ignored by the ledger
        "observation_tokens": int(observation_tokens),
    }


def build_requests_json(records: list[dict]) -> dict:
    """Assemble the rich agentic requests.json from per-turn records (pure)."""
    return {
        # standard reconstruction-ledger arrays
        "input_lens": [r["input_len"] for r in records],
        "output_lens": [r["output_len"] for r in records],
        "ttfts": [r["ttft"] for r in records],
        "itls": [r["itl"] for r in records],
        "request_timestamps": [r["request_timestamp"] for r in records],
        # agentic extensions
        "session_ids": [r["session_id"] for r in records],
        "turn_idx": [r["turn_idx"] for r in records],
        "post_gap_s": [r["post_gap_s"] for r in records],
        "prefix_cache": [r["prefix_cache"] for r in records],
        # replay provenance (ignored by the reconstruction ledger; for slicing)
        "tool_class": [r.get("tool_class", "") for r in records],
        "observation_tokens": [r.get("observation_tokens", 0) for r in records],
    }


def _filler(tokenizer, n_tokens: int) -> str:
    """A user message of ~n_tokens (synthetic; replaced by real text on replay)."""
    import numpy as np
    vocab = max(tokenizer.vocab_size - 100, 1000)
    ids = (np.arange(n_tokens) % vocab + 10).tolist()
    return tokenizer.decode(ids)


async def send_session(http, base_url, model, session, prefix_cache,
                       tokenizer) -> list[dict]:
    """Live: drive one conversation turn-by-turn over /v1/chat/completions.

    Context grows because real assistant replies are appended to ``messages``; a
    lognormal ``post_gap_s`` idle elapses between turns (the tool-execution gap).
    """
    import asyncio
    import json

    url = base_url.rsplit("/v1", 1)[0].rstrip("/") + "/v1/chat/completions"
    messages = []
    system_text = getattr(session, "system_text", "")
    if system_text:                                           # replay: real system prompt
        messages.append({"role": "system", "content": system_text})
    elif session.prefix_tokens > 0:                           # synthetic: filler prefix
        messages.append({"role": "system", "content": _filler(tokenizer, session.prefix_tokens)})

    records = []
    for turn in session.turns:
        replay = turn.assistant_text is not None
        user_content = turn.user_text or _filler(tokenizer, turn.new_input_tokens)
        messages.append({"role": "user", "content": user_content})
        payload = {"model": model, "messages": messages,
                   "max_tokens": turn.output_tokens, "stream": True,
                   "stream_options": {"include_usage": True}, "temperature": 0.0}
        if replay:
            # Force exact decode length so the measured decode work matches the
            # trace (the live content differs from the trace, but token count does
            # not, and only counts/timing feed the power ledger).
            payload["ignore_eos"] = True
        t_send = time.time()
        ttft, last, n_out, reply, prompt_tokens = None, t_send, 0, [], None
        async with http.post(url, json=payload) as resp:
            if resp.status >= 400:
                # A turn the server rejected (e.g. context-length) breaks the
                # conversation downstream; stop here rather than record a bad turn.
                break
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                if chunk.get("usage"):
                    prompt_tokens = chunk["usage"].get("prompt_tokens")
                choices = chunk.get("choices") or [{}]
                delta = (choices[0].get("delta") or {}).get("content", "")
                if not delta:
                    continue
                now = time.time()
                if ttft is None:
                    ttft = now - t_send
                last = now
                n_out += 1
                reply.append(delta)
        latency = max(last - t_send, 1e-6)
        tpot = (latency - (ttft or 0.0)) / max(n_out - 1, 1)
        # Faithful replay: grow context with the trace's REAL reply so the next
        # turn's prompt (and prefix-cache hit pattern) matches the real trajectory.
        ctx_reply = turn.assistant_text if replay else "".join(reply)
        messages.append({"role": "assistant", "content": ctx_reply})
        records.append(turn_record(
            session_id=session.session_id, turn_idx=turn.turn_idx,
            input_len=prompt_tokens if prompt_tokens else session.prefix_tokens,
            output_len=n_out, ttft=ttft or latency, tpot=tpot,
            request_timestamp=t_send, post_gap_s=turn.post_gap_s,
            prefix_cache=prefix_cache, tool_class=turn.tool_class,
            observation_tokens=turn.observation_tokens))
        if turn.post_gap_s > 0:
            await asyncio.sleep(turn.post_gap_s)   # tool-execution gap (idle)
    return records
