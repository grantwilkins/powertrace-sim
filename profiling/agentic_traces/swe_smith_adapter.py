"""Adapter: ``SWE-bench/SWE-smith-trajectories`` rows -> normalised text sessions.

Each dataset row is one agent trajectory: an OpenAI-style ``messages`` list
(system / user / assistant-with-tool_calls / tool-observation). We fold it into a
sequence of *turns*, where turn k = (the input that opens it, the real assistant
reply, and the tool the assistant calls). The tool's result is the *next* turn's
input, and its token length conditions the post-turn gap (see ``gap_sampler``).

Pure given a tokenizer; the live dataset download lives in ``load_swe_smith``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import tool_classes

DATASET = "SWE-bench/SWE-smith-trajectories"
SPLIT = "tool"  # the split with structured Anthropic-style tool_calls (vs xml / ticks)


@dataclass(frozen=True)
class TextTurn:
    user_text: str           # input opening this turn (initial task, or prior tool result)
    assistant_text: str      # the real assistant reply (reasoning + tool call)
    tool_class: str          # class of the tool this turn calls ("" if none / final answer)
    observation_tokens: int  # token length of that tool's result (conditions the post-gap)


@dataclass(frozen=True)
class TextSession:
    session_id: str
    system_text: str
    turns: list[TextTurn]


def _text(msg: dict) -> str:
    """Message text, flattening Anthropic-style content blocks to a string."""
    content = msg.get("content")
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content or ""


def _tool_name(msg: dict) -> str:
    calls = msg.get("tool_calls") or []
    return calls[0].get("function", {}).get("name", "") if calls else ""


def _assistant_text(msg: dict) -> str:
    """Reply as context: the content, plus a compact rendering of any tool call so
    the injected turn carries the action (keeps the grown prompt faithful)."""
    calls = msg.get("tool_calls") or []
    action = json.dumps([c.get("function", {}) for c in calls]) if calls else ""
    return "\n".join(filter(None, [_text(msg), action]))


def session_from_messages(messages, tokenizer, session_id: str) -> TextSession:
    """Fold one trajectory's ``messages`` into a ``TextSession``.

    ``messages`` may be the raw OpenAI list or a JSON string (the SWE-smith ``tool``
    split stores it serialised); both are accepted.
    """
    if isinstance(messages, str):
        messages = json.loads(messages)
    system_text = next((_text(m) for m in messages if m["role"] == "system"), "")

    # Pair each assistant reply with the input that preceded it (user or tool obs).
    paired, pending = [], None
    for m in messages:
        if m["role"] in ("user", "tool"):
            pending = _text(m)
        elif m["role"] == "assistant" and pending is not None:
            tool = _tool_name(m)
            cls = tool_classes.classify(tool) if tool else ""
            paired.append((pending, _assistant_text(m), cls))
            pending = None

    # turn k's observation = turn k+1's input (the result of turn k's tool call).
    turns = []
    for k, (user_text, asst_text, cls) in enumerate(paired):
        next_input = paired[k + 1][0] if k + 1 < len(paired) else ""
        obs_tokens = len(tokenizer(next_input)["input_ids"]) if next_input else 0
        turns.append(TextTurn(user_text, asst_text, cls, obs_tokens))
    return TextSession(session_id, system_text, turns)


def _has_tool_classes(session: TextSession) -> bool:
    """Keep only structured trajectories: the split is heterogeneous, and some
    rows carry no ``tool_calls`` (tool unknowable) — those can't condition the
    per-tool gap model, so we skip them."""
    return any(t.tool_class for t in session.turns)


def load_swe_smith(n_sessions: int, seed: int, tokenizer) -> list[TextSession]:
    """Load + shuffle the dataset, return the first ``n_sessions`` structured ones.

    Non-streaming so it loads from the local HF cache offline (the GPU job has no
    network); the ``tool`` split is small enough that this is cheap.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT).shuffle(seed=seed)
    sessions = (
        session_from_messages(row["messages"], tokenizer, f"swe_{i}")
        for i, row in enumerate(ds))
    usable = (s for s in sessions if s.turns and _has_tool_classes(s))
    return list(itertools.islice(usable, n_sessions))
