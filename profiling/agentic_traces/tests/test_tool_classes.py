"""Unit tests for the tool-class taxonomy."""

import pytest

import tool_classes as tc


@pytest.mark.parametrize("name,expected", [
    ("str_replace_editor", tc.LOCAL_IO),
    ("edit", tc.LOCAL_IO),
    ("read_file", tc.LOCAL_IO),
    ("grep", tc.LOCAL_IO),
    ("execute_bash", tc.BASH),
    ("python", tc.BASH),
    ("run_pytest", tc.BASH),
    ("web_fetch", tc.REMOTE),
    ("web_search", tc.REMOTE),
    ("call_api", tc.REMOTE),
    ("spawn_subagent", tc.SUBAGENT),
    ("delegate_task", tc.SUBAGENT),
])
def test_classify_known(name, expected):
    assert tc.classify(name) == expected


def test_unknown_defaults_to_bash():
    assert tc.classify("totally_unheard_of") == tc.BASH
    assert tc.classify("") == tc.BASH


def test_every_class_is_covered():
    seen = {tc.classify(n) for n in
            ("edit", "execute_bash", "web_fetch", "spawn_subagent")}
    assert seen == set(tc.CLASSES)
