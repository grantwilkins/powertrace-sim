import pytest

import agentic
import openhands_preflight as preflight


def _plan():
    return agentic.from_text_transcript(
        [
            (
                "session", "system", 1,
                [
                    (2, 3, 0.25, "input", "output", "bash", 2),
                    (4, 5, 0.0, "observation", "done", "", 0),
                ],
            )
        ],
        source="openhands", revision="commit", seed=7,
        pack_index=0, pack_count=1,
    )


def test_summary_covers_replay_horizon_and_token_work():
    summary = preflight.summarize_plan(_plan())
    assert summary["sessions"] == 1
    assert summary["turns"] == 2
    assert summary["prompt_context_tokens"] == 13
    assert summary["output_tokens"] == 8
    assert summary["tool_wait_s"] == 0.25
    assert summary["max_turn_wait_s"] == 0.25
    assert summary["max_session_wait_s"] == 0.25


def test_summary_requires_sealed_hash_and_wait_limits():
    summary = preflight.summarize_plan(_plan())
    sessions = {
        "expected_plan_sha256": [summary["sha256"]],
        "max_turn_wait_s": 1.0,
        "max_session_wait_s": 1.0,
    }
    preflight.validate_summary(summary, sessions)

    sessions["expected_plan_sha256"] = ["0" * 64]
    with pytest.raises(ValueError, match="does not match sealed hash"):
        preflight.validate_summary(summary, sessions)

    sessions["expected_plan_sha256"] = [summary["sha256"]]
    sessions["max_session_wait_s"] = 0.1
    with pytest.raises(ValueError, match="max_session_wait_s"):
        preflight.validate_summary(summary, sessions)
