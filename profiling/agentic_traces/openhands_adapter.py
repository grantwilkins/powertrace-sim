"""OpenHands evaluation histories -> text sessions with observed tool gaps."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

import tool_classes
from swe_smith_adapter import TextSession, TextTurn

DATASET = "OpenHands/openhands-evaluation-outputs"
DATA_FILE = (
    "outputs/SWE-bench_Lite-test/CodeActAgent/"
    "claude-3-5-sonnet-20241022_maxiter_100_N_v2.2-no-hint/output.jsonl"
)
LOCAL_DATA_ENV = "OPENHANDS_DATASET_PATH"


def _epoch(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _text(event: dict) -> str:
    for key in ("content", "message", "observation"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    extras = event.get("extras") or {}
    for key in ("content", "command", "code"):
        value = extras.get(key)
        if isinstance(value, str) and value:
            return value
    args = event.get("args") or {}
    for key in ("content", "thought", "command", "code"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _assistant_text(event: dict) -> str:
    body = _text(event)
    action = event.get("tool_name") or event.get("action") or ""
    args = event.get("args") or {}
    rendered = json.dumps(
        {"action": action, "args": args}, sort_keys=True, default=str
    ) if action else ""
    return "\n".join(value for value in (body, rendered) if value)


def _session_id(row: dict) -> str:
    value = row.get("instance_id") or (row.get("instance") or {}).get("instance_id")
    if value:
        return str(value)
    payload = json.dumps(row.get("metadata") or {}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def session_from_row(row: dict, tokenizer) -> TextSession:
    """Preserve original text and action->observation wall-clock gaps."""
    history = row.get("history") or row.get("events") or []
    system = str(row.get("system_prompt") or "")
    pending_input = str(
        row.get("instruction")
        or (row.get("instance") or {}).get("problem_statement")
        or ""
    )
    turns = []
    for index, event in enumerate(history):
        source = str(event.get("source") or "").lower()
        if source in {"environment", "user"}:
            pending_input = _text(event)
            continue
        if source != "agent" or not pending_input:
            continue
        next_environment = next(
            (
                candidate for candidate in history[index + 1:]
                if str(candidate.get("source") or "").lower()
                in {"environment", "user"}
            ),
            None,
        )
        observation = _text(next_environment or {})
        gap = 0.0
        if next_environment is not None:
            gap = _epoch(next_environment["timestamp"]) - _epoch(event["timestamp"])
            if gap < 0:
                raise ValueError("OpenHands event timestamps are not monotonic")
        action = event.get("tool_name") or event.get("action") or ""
        turns.append(TextTurn(
            user_text=pending_input,
            assistant_text=_assistant_text(event),
            tool_class=tool_classes.classify(str(action)) if action else "",
            observation_tokens=(
                len(tokenizer(observation)["input_ids"]) if observation else 0
            ),
            post_gap_s=gap,
        ))
        pending_input = observation
    return TextSession(_session_id(row), system, turns)


def select_rows(
    rows, *, n_sessions: int, pack_index: int, pack_count: int, seed: int = 0
):
    if n_sessions <= 0 or pack_count <= 0 or not 0 <= pack_index < pack_count:
        raise ValueError("invalid OpenHands pack selection")
    selected = (
        row for row in rows
        if int(hashlib.sha256(
            f"{seed}:{_session_id(row)}".encode()
        ).hexdigest(), 16)
        % pack_count == pack_index
    )
    return list(itertools.islice(selected, n_sessions))


def _local_rows(path: str):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_openhands(
    n_sessions: int, seed: int, tokenizer, *, pack_index: int = 0,
    pack_count: int = 1, revision: str | None = None,
) -> list[TextSession]:
    """Load a pinned offline dataset revision and one disjoint hash pack."""
    if not revision:
        raise ValueError("OpenHands replay requires an immutable dataset revision")
    local_path = os.environ.get(LOCAL_DATA_ENV)
    if local_path:
        rows = _local_rows(str(Path(local_path).resolve()))
    else:
        from datasets import load_dataset

        source = f"hf://datasets/{DATASET}@{revision}/{DATA_FILE}"
        rows = load_dataset(
            "json", data_files={"test": source}, split="test", streaming=True
        )
    chosen = select_rows(
        rows, n_sessions=n_sessions * 4, pack_index=pack_index,
        pack_count=pack_count, seed=seed,
    )
    sessions = [session_from_row(row, tokenizer) for row in chosen]
    sessions = [session for session in sessions if session.turns][:n_sessions]
    if len(sessions) != n_sessions:
        raise ValueError(
            f"OpenHands pack {pack_index}/{pack_count} contains only "
            f"{len(sessions)} usable sessions, requested {n_sessions}"
        )
    return sessions
