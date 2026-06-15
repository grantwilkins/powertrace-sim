"""Agentic / multi-turn session plans (pure, unit-tested) — §5-D, planned.

A *session* is a sequence of turns where each turn's prompt is the full prior
context + a new user/tool message, so context grows monotonically; between turns a
**tool-execution gap** (idle, lognormal) elapses. This is the part vLLM's
multi-turn benchmark lacks (it has Poisson request pacing, not a per-turn gap, and
monotonic timestamps rather than epoch).

This module produces only the *plan* (deterministic given a seed); the live HTTP
session sender lives in ``session_driver``. The plan is what we unit-test; the rich
per-turn output schema is documented in ``profiling/BUNDLE_SCHEMA.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SessionTurn:
    """One turn: new user tokens this turn, assistant reply length, then a gap."""
    turn_idx: int
    new_input_tokens: int     # tokens added this turn (user/tool message)
    output_tokens: int        # assistant reply length
    post_gap_s: float         # tool-execution / think-time idle after this turn


@dataclass
class SessionPlan:
    """One conversation: a system prefix then a sequence of growing turns."""
    session_id: str
    prefix_tokens: int        # system prompt (shared, prefix-cacheable)
    turns: list[SessionTurn]

    def context_lengths(self) -> list[int]:
        """Full prompt length seen at each turn (prefix + all prior + this input)."""
        out, ctx = [], self.prefix_tokens
        for t in self.turns:
            ctx += t.new_input_tokens
            out.append(ctx)
            ctx += t.output_tokens   # assistant reply becomes context next turn
        return out


@dataclass
class AgenticPlan:
    """A set of sessions plus the regime they run under."""
    sessions: list[SessionPlan]
    prefix_cache: bool        # whether the server has prefix caching on
    label: str = "agentic"

    @property
    def total_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


def _lognormal(rng, mean, sigma, size=None):
    """Lognormal with the given (approx) MEAN and log-sigma."""
    mu = np.log(max(mean, 1e-6)) - 0.5 * sigma**2
    return rng.lognormal(mu, sigma, size)


def build_synthetic_sessions(
    n_sessions: int = 4,
    min_turns: int = 2,
    max_turns: int = 8,
    user_tokens_mean: int = 256,
    assistant_tokens_mean: int = 256,
    prefix_tokens: int = 512,
    gap_mean_s: float = 3.0,
    gap_sigma: float = 0.8,
    prefix_cache: bool = True,
    seed: int = 0,
) -> AgenticPlan:
    """Deterministic synthetic agentic sessions (growing context + tool gaps)."""
    rng = np.random.default_rng(seed)
    sessions = []
    for s in range(n_sessions):
        n_turns = int(rng.integers(min_turns, max_turns + 1))
        turns = []
        for k in range(n_turns):
            turns.append(SessionTurn(
                turn_idx=k,
                new_input_tokens=int(max(1, _lognormal(rng, user_tokens_mean, 0.6))),
                output_tokens=int(max(1, _lognormal(rng, assistant_tokens_mean, 0.6))),
                # no gap after the final turn
                post_gap_s=(0.0 if k == n_turns - 1
                            else float(_lognormal(rng, gap_mean_s, gap_sigma))),
            ))
        sessions.append(SessionPlan(
            session_id=f"sess_{seed}_{s}", prefix_tokens=prefix_tokens, turns=turns))
    return AgenticPlan(sessions=sessions, prefix_cache=prefix_cache)


def from_transcript(transcripts, prefix_cache: bool = False) -> AgenticPlan:
    """Replay real agent transcripts.

    Each transcript is an ordered list of ``(new_input_tokens, output_tokens,
    post_gap_s)`` triples; context growth and gaps come straight from the trace.
    """
    sessions = []
    for i, turns_in in enumerate(transcripts):
        turns = [
            SessionTurn(turn_idx=k, new_input_tokens=int(ni),
                        output_tokens=int(no), post_gap_s=float(g))
            for k, (ni, no, g) in enumerate(turns_in)
        ]
        sessions.append(SessionPlan(
            session_id=f"replay_{i}", prefix_tokens=0, turns=turns))
    return AgenticPlan(sessions=sessions, prefix_cache=prefix_cache, label="agentic_replay")
