"""Unit tests for the SWE-smith adapter (pure; word-splitting fake tokenizer)."""

import swe_smith_adapter as ad
import tool_classes as tc


class FakeTok:
    """Whitespace tokenizer: token count == word count."""
    def __call__(self, text):
        return {"input_ids": text.split()}


def _tool_call(name):
    return [{"type": "function", "id": "c1", "function": {"name": name, "arguments": "{}"}}]


MESSAGES = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "fix the bug"},
    {"role": "assistant", "content": "let me look", "tool_calls": _tool_call("str_replace_editor")},
    {"role": "tool", "tool_call_id": "c1", "content": "file with five word body"},  # 5 tokens
    {"role": "assistant", "content": "now run tests", "tool_calls": _tool_call("execute_bash")},
    {"role": "tool", "tool_call_id": "c1", "content": "tests pass"},                # 2 tokens
    {"role": "assistant", "content": "all done"},                                   # final, no tool
]


def test_folds_into_turns_with_tool_metadata():
    s = ad.session_from_messages(MESSAGES, FakeTok(), "swe_0")
    assert s.system_text == "you are an agent"
    assert [t.user_text for t in s.turns] == [
        "fix the bug", "file with five word body", "tests pass"]
    # tool class = the tool THIS turn's assistant calls (final turn calls none)
    assert [t.tool_class for t in s.turns] == [tc.LOCAL_IO, tc.BASH, ""]
    # observation tokens = size of the NEXT turn's input (this turn's tool result)
    assert [t.observation_tokens for t in s.turns] == [5, 2, 0]


def test_assistant_text_carries_the_action():
    s = ad.session_from_messages(MESSAGES, FakeTok(), "swe_0")
    assert "let me look" in s.turns[0].assistant_text
    assert "str_replace_editor" in s.turns[0].assistant_text  # tool call rendered for context


def test_truncated_trajectory_has_no_trailing_observation():
    msgs = MESSAGES[:3]  # system, user, assistant(tool) — no following observation
    s = ad.session_from_messages(msgs, FakeTok(), "swe_0")
    assert len(s.turns) == 1
    assert s.turns[0].observation_tokens == 0


def test_empty_trajectory_yields_no_turns():
    s = ad.session_from_messages(
        [{"role": "system", "content": "x"}], FakeTok(), "swe_0")
    assert s.turns == []


def test_accepts_json_string_messages():
    """The SWE-smith `tool` split stores `messages` as a JSON string."""
    import json
    s = ad.session_from_messages(json.dumps(MESSAGES), FakeTok(), "swe_0")
    assert [t.user_text for t in s.turns] == [
        "fix the bug", "file with five word body", "tests pass"]


def test_flattens_content_blocks():
    """Anthropic-style list content is flattened to text."""
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "two words"}]},
        {"role": "assistant", "content": "ok", "tool_calls": _tool_call("bash")},
        {"role": "tool", "content": [{"type": "text", "text": "obs here now"}]},  # 3 tokens
        {"role": "assistant", "content": "done"},
    ]
    s = ad.session_from_messages(msgs, FakeTok(), "swe_0")
    assert s.turns[0].user_text == "two words"
    assert s.turns[0].observation_tokens == 3
    assert s.turns[1].user_text == "obs here now"
