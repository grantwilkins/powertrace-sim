"""Validate and summarize every pack in a sealed OpenHands campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "probes"))

import transformers

from replay_loader import build_replay_plan


def summarize_plan(plan) -> dict:
    session_waits = [
        sum(turn.post_gap_s for turn in session.turns)
        for session in plan.sessions
    ]
    turn_waits = [
        turn.post_gap_s
        for session in plan.sessions
        for turn in session.turns
    ]
    return {
        "pack_index": plan.pack_index,
        "sha256": plan.sha256,
        "sessions": len(plan.sessions),
        "turns": plan.total_turns,
        "prompt_context_tokens": sum(
            sum(session.context_lengths()) for session in plan.sessions
        ),
        "output_tokens": sum(
            turn.output_tokens
            for session in plan.sessions
            for turn in session.turns
        ),
        "tool_wait_s": sum(turn_waits),
        "max_turn_wait_s": max(turn_waits, default=0.0),
        "max_session_wait_s": max(session_waits, default=0.0),
    }


def validate_summary(summary: dict, sessions: dict) -> None:
    pack = summary["pack_index"]
    expected = sessions["expected_plan_sha256"]
    if summary["sha256"] != expected[pack]:
        raise ValueError(
            f"OpenHands pack {pack} replay hash {summary['sha256']} "
            f"does not match sealed hash {expected[pack]}"
        )
    for field in ("max_turn_wait_s", "max_session_wait_s"):
        limit = float(sessions[field])
        if summary[field] > limit:
            raise ValueError(
                f"OpenHands pack {pack} {field}={summary[field]:.3f} "
                f"exceeds {limit:.3f}"
            )


def validate_pack(campaign: dict, pack: int) -> dict:
    sessions = campaign["sessions"]
    pack_count = int(sessions["pack_count"])
    if not 0 <= pack < pack_count:
        raise ValueError(f"pack index {pack} outside [0, {pack_count})")
    tokenizer = transformers.AutoTokenizer.from_pretrained(campaign["model"])
    plan = build_replay_plan(
        corpus="openhands", n_sessions=int(sessions["n_sessions"]),
        seed=int(sessions["seed"]), tokenizer=tokenizer, prefix_cache=False,
        max_model_len=int(campaign["server"]["max_model_len"]),
        pack_index=pack, pack_count=pack_count,
        dataset_revision=sessions["dataset_revision"],
    )
    summary = summarize_plan(plan)
    validate_summary(summary, sessions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--pack-index", required=True, type=int)
    args = parser.parse_args()
    campaign = json.loads(Path(args.campaign).read_text())
    print(json.dumps(validate_pack(campaign, args.pack_index), sort_keys=True))


if __name__ == "__main__":
    main()
