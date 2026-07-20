"""Claim: normalization preserves exact order, units, prefixes, and token counts.

Plausible wrong implementations caught here: treating milliseconds as seconds,
sorting rounds globally instead of per session, or accepting missing/overflowed
rounds that the live replay would later omit.
"""

import json
import gzip
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_plan import (  # noqa: E402
    assign_poisson_session_arrivals, load_bundle_requests_json,
    load_burstgpt_csv, load_plan,
    load_tracelab_csv, load_tracelab_jsonl, select_context_bands,
    select_densest_arrival_window, select_sessions, write_plan,
)


def test_tracelab_ms_normalization_and_round_trip(tmp_path):
    csv_path = tmp_path / "trace.csv"
    csv_path.write_text(
        "id,input_len,output_len,arrival_time,round_idx,"
        "tool_wait_after_ms,prefix_len\n"
        "a,11,7,1000,0,250,5\n"
        "a,3,2,1250,1,0,23\n"
        "b,4,6,1100,0,0,8\n"
    )
    plan = load_tracelab_csv(csv_path, revision="commit-1", seed=9)
    assert [row.ready_s for row in plan.rounds] == [0.0, 0.25, 0.1]
    assert plan.rounds[0].tool_wait_s == 0.25
    assert (plan.rounds[1].prefix_tokens, plan.rounds[1].input_tokens) == (23, 3)

    output = tmp_path / "plan.json"
    write_plan(plan, output)
    loaded = load_plan(output, max_model_len=64)
    assert loaded.sha256 == plan.sha256
    assert json.loads(output.read_text())["schema_version"] == 1


def test_plan_rejects_noncontiguous_round_and_context_overflow(tmp_path):
    payload = {
        "schema_version": 1, "source": "x", "revision": "r", "seed": 0,
        "rounds": [{
            "session_id": "s", "round_idx": 1, "ready_s": 0.0,
            "prefix_tokens": 10, "input_tokens": 10, "output_tokens": 10,
            "tool_wait_s": 0.0, "source_id": "",
        }],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contiguous"):
        load_plan(path)
    payload["rounds"][0]["round_idx"] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        load_plan(path, max_model_len=29)


def test_selection_is_seeded_bounded_and_context_stratified(tmp_path):
    csv_path = tmp_path / "trace.csv"
    csv_path.write_text(
        "id,input_len,output_len,arrival_time,round_idx,"
        "tool_wait_after_ms,prefix_len\n"
        "short,10,2,0,0,0,10\n"
        "long_a,20,2,10,0,0,100\n"
        "long_a,5,2,20,1,0,122\n"
        "long_b,20,2,30,0,0,110\n"
    )
    plan = load_tracelab_csv(csv_path, revision="r", seed=3)
    selected = select_sessions(
        plan, max_sessions=1, max_rounds_per_session=1,
        min_max_context=100,
    )
    assert len(selected.by_session()) == 1
    assert len(selected.rounds) == 1
    assert selected.rounds[0].prefix_tokens + selected.rounds[0].input_tokens >= 100
    stratified = select_context_bands(
        plan, [(0, 100, 1), (100, 1000, 2)], max_rounds_per_session=2
    )
    assert len(stratified.by_session()) == 3


def test_public_jsonl_mapping_and_compact_arrivals(tmp_path):
    rows = [
        {
            "session_id": "s1", "round_index": 0, "round_id": "r1",
            "prefix_tokens": 0, "newly_append_tokens": 10, "output_tokens": 3,
            "tools": [{"tool_wall_latency_ms": 250}],
        },
        {
            "session_id": "s1", "round_index": 1, "round_id": "r2",
            "prefix_tokens": 13, "newly_append_tokens": 2, "output_tokens": 1,
            "tools": [{"tool_wall_latency_ms": 999}],
        },
        {
            "session_id": "s2", "round_index": 0, "round_id": "r3",
            "prefix_tokens": 0, "newly_append_tokens": 20, "output_tokens": 2,
            "tools": [],
        },
    ]
    path = tmp_path / "trace.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    plan = load_tracelab_jsonl(path, revision="v0.0.1", seed=0)
    assert plan.rounds[0].tool_wait_s == 0.25
    assert plan.rounds[0].cached_prefix_tokens == 0
    assert plan.rounds[1].cached_prefix_tokens == 13
    assert plan.rounds[1].tool_wait_s == 0.0
    scheduled = assign_poisson_session_arrivals(plan, rate_rps=1.0, seed=1)
    starts = [session[0].ready_s for session in scheduled.by_session().values()]
    assert starts[0] == 0.0 and 0.0 < starts[1] < 10.0


def test_burstgpt_preserves_exact_irregular_arrivals(tmp_path):
    path = tmp_path / "burst.csv"
    path.write_text(
        "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
        "1000,ChatGPT,10,2,12,Conversation log\n"
        "1001,GPT-4,20,3,23,Conversation log\n"
        "1010,ChatGPT,30,4,34,Conversation log\n"
        "1011,GPT-4,40,5,45,Conversation log\n"
        "1012,ChatGPT,50,6,56,Conversation log\n"
    )
    plan = load_burstgpt_csv(path, revision="sha", seed=0)
    selected = select_densest_arrival_window(
        plan, duration_s=2.0, max_requests=10
    )
    assert [row.ready_s for row in selected.rounds] == [0.0, 1.0, 2.0]
    assert [row.input_tokens for row in selected.rounds] == [30, 40, 50]


def test_bundle_requests_plan_preserves_marks_and_scales_release_time(tmp_path):
    source = tmp_path / "bundle" / "requests.json"
    source.parent.mkdir()
    source.write_text(json.dumps({
        "input_lens": [8, 16],
        "output_lens": [3, 5],
        "request_timestamps": [100.0, 102.0],
    }))
    plan = load_bundle_requests_json(source, time_scale=0.5, seed=7)
    assert [row.ready_s for row in plan.rounds] == [0.0, 1.0]
    assert [row.input_tokens for row in plan.rounds] == [8, 16]
    assert [row.output_tokens for row in plan.rounds] == [3, 5]
    assert plan.seed == 7
    assert "time_scale:0.5" in plan.revision


def test_bundle_requests_plan_rejects_ragged_rows(tmp_path):
    path = tmp_path / "requests.json"
    path.write_text(json.dumps({
        "input_lens": [8, 16],
        "output_lens": [3],
        "request_timestamps": [100.0, 99.0],
    }))
    with pytest.raises(ValueError, match="equal length"):
        load_bundle_requests_json(path)
