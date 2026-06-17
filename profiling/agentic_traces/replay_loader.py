"""Build a text-carrying ``AgenticPlan`` from real agent traces (Gap 1).

Glues the corpus adapter (real text + tool metadata) to the gap sampler (per-turn
idle drawn from the per-tool-class model) and hands the result to
``agentic.from_text_transcript``. Token counts and gaps are resolved here so the
plan module stays tokenizer-free. Deterministic given ``seed``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                        # tool_classes, gap_sampler, adapters
sys.path.insert(0, str(_HERE.parents[0] / "probes"))  # agentic

import agentic
import swe_smith_adapter
from gap_sampler import DEFAULT_PARAMS, GapSampler

CORPORA = {"swe_smith": swe_smith_adapter.load_swe_smith}

# Headroom below max_model_len for chat-template tokens (role markers etc., a few
# per message) that our content-only token counts don't include. The server rejects
# a turn when prompt + max_tokens exceeds the window, so we keep a margin.
SAFETY_MARGIN = 1024


def _ntok(tokenizer, text: str) -> int:
    return len(tokenizer(text)["input_ids"]) if text else 0


def _to_item(session, tokenizer, sampler, rng, max_model_len):
    """One adapter ``TextSession`` -> a ``from_text_transcript`` item, truncated so
    each turn's prompt AND its generation fit the served context window."""
    sys_tokens = _ntok(tokenizer, session.system_text)
    budget = max_model_len - SAFETY_MARGIN

    kept, ctx = [], sys_tokens
    for t in session.turns:
        n_in = _ntok(tokenizer, t.user_text)
        n_out = _ntok(tokenizer, t.assistant_text)
        if ctx + n_in + n_out > budget:          # prompt + generation would overflow
            break
        kept.append((t, n_in, n_out))
        ctx += n_in + n_out

    turns = []
    for i, (t, n_in, n_out) in enumerate(kept):
        is_last = i == len(kept) - 1             # no gap after the session's final turn
        gap = 0.0 if is_last else sampler.sample(t.tool_class, t.observation_tokens, rng)
        turns.append((n_in, n_out, gap, t.user_text, t.assistant_text,
                      t.tool_class, t.observation_tokens))
    return (session.session_id, session.system_text, sys_tokens, turns)


def build_replay_plan(*, corpus="swe_smith", n_sessions, seed, tokenizer,
                      prefix_cache, gap_params=DEFAULT_PARAMS, max_model_len):
    """Load real traces, draw conditioned gaps, return a replay ``AgenticPlan``."""
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; choose from {sorted(CORPORA)}")

    sessions = CORPORA[corpus](n_sessions, seed, tokenizer)
    sampler = GapSampler.from_file(gap_params)
    rng = np.random.default_rng(seed)

    items = [_to_item(s, tokenizer, sampler, rng, max_model_len) for s in sessions]
    items = [it for it in items if it[3]]        # drop sessions truncated to nothing
    return agentic.from_text_transcript(items, prefix_cache=prefix_cache)
