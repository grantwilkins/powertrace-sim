import openhands_adapter


class Tokenizer:
    def __call__(self, text):
        return {"input_ids": text.split()}


def _row(instance_id, offset=0):
    return {
        "instance_id": instance_id,
        "instruction": "fix the bug",
        "history": [
            {
                "id": 1, "source": "agent", "timestamp": 10 + offset,
                "action": "run", "args": {"command": "pytest"},
            },
            {
                "id": 2, "source": "agent", "timestamp": 12.5 + offset,
                "cause": 1, "observation": "run", "content": "one two three",
            },
            {
                "id": 3, "source": "agent", "timestamp": 15 + offset,
                "action": "edit", "args": {"path": "a.py"},
            },
            {
                "id": 4, "source": "agent", "timestamp": 19 + offset,
                "cause": 3, "observation": "edit", "content": "done",
            },
        ],
    }


def test_preserves_real_text_and_observed_gaps():
    session = openhands_adapter.session_from_row(_row("x"), Tokenizer())
    assert session.session_id == "x"
    assert session.system_text == ""
    assert [turn.post_gap_s for turn in session.turns] == [2.5, 4.0]
    assert session.turns[0].user_text == "fix the bug"
    assert session.turns[1].user_text == "one two three"
    assert session.turns[0].observation_tokens == 3


def test_does_not_replay_observations_as_turns_or_reuse_future_user_gap():
    row = _row("x")
    row["history"].extend([
        {
            "id": 5, "source": "agent", "timestamp": 20,
            "action": "message", "args": {"content": "finished"},
        },
        {
            "id": 6, "source": "user", "timestamp": 700,
            "action": "message", "args": {"content": "continue"},
            "message": "continue",
        },
        {
            "id": 7, "source": "agent", "timestamp": 701,
            "action": "finish", "args": {}, "message": "done",
        },
    ])
    session = openhands_adapter.session_from_row(row, Tokenizer())
    assert len(session.turns) == 4
    assert [turn.post_gap_s for turn in session.turns] == [2.5, 4.0, 0.0, 0.0]
    assert session.turns[-1].user_text == "continue"


def test_rejects_tool_action_without_cause_linked_observation():
    import pytest

    row = _row("x")
    row["history"] = row["history"][:1]
    with pytest.raises(ValueError, match="cause-linked observation"):
        openhands_adapter.session_from_row(row, Tokenizer())


def test_hash_packs_are_disjoint_and_deterministic():
    rows = [_row(str(index), index * 20) for index in range(30)]
    packs = [
        openhands_adapter.select_rows(
            rows, n_sessions=30, pack_index=index, pack_count=3
        )
        for index in range(3)
    ]
    ids = [{row["instance_id"] for row in pack} for pack in packs]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set().union(*ids) == {str(index) for index in range(30)}
    assert packs[1] == openhands_adapter.select_rows(
        rows, n_sessions=30, pack_index=1, pack_count=3
    )


def test_requires_immutable_revision():
    import pytest

    with pytest.raises(ValueError, match="immutable dataset revision"):
        openhands_adapter.load_openhands(1, 0, Tokenizer())


def test_local_staged_dataset_is_used(monkeypatch, tmp_path):
    path = tmp_path / "output.jsonl"
    path.write_text(__import__("json").dumps(_row("x")) + "\n")

    monkeypatch.setenv(openhands_adapter.LOCAL_DATA_ENV, str(path))
    sessions = openhands_adapter.load_openhands(
        1, 0, Tokenizer(), revision="commit"
    )
    assert sessions[0].session_id == "x"


def test_session_offset_selects_new_usable_sessions(monkeypatch, tmp_path):
    import json

    path = tmp_path / "output.jsonl"
    path.write_text(
        "".join(json.dumps(_row(str(index), index * 20)) + "\n"
                for index in range(8))
    )
    monkeypatch.setenv(openhands_adapter.LOCAL_DATA_ENV, str(path))
    first = openhands_adapter.load_openhands(
        2, 0, Tokenizer(), revision="commit"
    )
    second = openhands_adapter.load_openhands(
        2, 0, Tokenizer(), revision="commit", session_offset=2
    )
    assert {session.session_id for session in first}.isdisjoint(
        session.session_id for session in second
    )
