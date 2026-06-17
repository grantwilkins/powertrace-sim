"""Unit tests for the replay loader (context truncation + gap assignment)."""

import numpy as np

import replay_loader as rl
from gap_sampler import GapSampler
from swe_smith_adapter import TextSession, TextTurn

PARAMS = {"classes": {"bash": {"mu": 0.0, "sigma": 0.5, "b1": 0.1},
                      "local_io": {"mu": -2.0, "sigma": 0.3, "b1": 0.05}}}


class FakeTok:
    def __call__(self, text):
        return {"input_ids": text.split()}


def _session():
    return TextSession("s0", "sys prompt here", [
        TextTurn("alpha beta", "reply one", "local_io", 2),
        TextTurn("gamma delta", "reply two", "bash", 3),
        TextTurn("epsilon", "reply three", "", 0),
    ])


def test_item_carries_text_and_zero_final_gap():
    item = rl._to_item(_session(), FakeTok(), GapSampler(PARAMS),
                       np.random.default_rng(0), max_model_len=10_000)
    session_id, system_text, sys_tokens, turns = item
    assert session_id == "s0" and sys_tokens == 3            # "sys prompt here"
    assert len(turns) == 3
    assert turns[-1][2] == 0.0                               # no gap after final turn
    assert all(turns[i][2] > 0 for i in range(2))            # interior gaps sampled
    # tuple shape: (n_in, n_out, gap, user_text, assistant_text, tool_class, obs_tokens)
    assert turns[0][3] == "alpha beta" and turns[0][5] == "local_io"


def test_truncates_when_context_overflows():
    # turn0 needs sys(3) + n_in(2) + n_out(2) = 7 tokens; turn1 would add 4 more.
    # Size the budget (max_model_len - SAFETY_MARGIN) to admit only the first turn.
    item = rl._to_item(_session(), FakeTok(), GapSampler(PARAMS),
                       np.random.default_rng(0), max_model_len=rl.SAFETY_MARGIN + 8)
    _, _, _, turns = item
    assert len(turns) == 1                                   # later turns overflow
    assert turns[-1][2] == 0.0                               # kept turn becomes the last


def test_from_text_transcript_builds_text_plan():
    import agentic
    item = rl._to_item(_session(), FakeTok(), GapSampler(PARAMS),
                       np.random.default_rng(0), max_model_len=10_000)
    plan = agentic.from_text_transcript([item], prefix_cache=True)
    s = plan.sessions[0]
    assert s.system_text == "sys prompt here" and s.prefix_tokens == 3
    assert s.turns[0].user_text == "alpha beta"
    assert s.turns[0].assistant_text == "reply one"
    assert s.context_lengths() == sorted(s.context_lengths())  # grows monotonically
    assert plan.prefix_cache is True and plan.label == "agentic_replay"
