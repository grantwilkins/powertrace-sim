"""Unit tests for the agentic/multi-turn session plan + rich requests.json (§5-D)."""

import agentic
import session_driver
import session_runner


def test_synthetic_sessions_deterministic():
    a = agentic.build_synthetic_sessions(n_sessions=4, seed=0)
    b = agentic.build_synthetic_sessions(n_sessions=4, seed=0)
    assert [s.session_id for s in a.sessions] == [s.session_id for s in b.sessions]
    assert [[t.new_input_tokens for t in s.turns] for s in a.sessions] == \
           [[t.new_input_tokens for t in s.turns] for s in b.sessions]
    c = agentic.build_synthetic_sessions(n_sessions=4, seed=1)
    assert a.sessions[0].turns[0].new_input_tokens != c.sessions[0].turns[0].new_input_tokens \
        or len(a.sessions[0].turns) != len(c.sessions[0].turns)


def test_context_grows_monotonically():
    a = agentic.build_synthetic_sessions(n_sessions=2, min_turns=4, max_turns=4,
                                         prefix_tokens=500, seed=3)
    for s in a.sessions:
        ctx = s.context_lengths()
        assert ctx == sorted(ctx)               # non-decreasing
        assert ctx[0] >= 500                     # includes the system prefix
        assert len(ctx) == len(s.turns)


def test_no_gap_after_final_turn():
    a = agentic.build_synthetic_sessions(n_sessions=3, seed=5)
    for s in a.sessions:
        assert s.turns[-1].post_gap_s == 0.0
        # earlier turns generally have a positive tool-gap
        if len(s.turns) > 1:
            assert any(t.post_gap_s > 0 for t in s.turns[:-1])


def test_from_transcript_replay():
    plan = agentic.from_transcript(
        [[(100, 50, 2.0), (80, 60, 1.5)], [(200, 100, 0.0)]], prefix_cache=False)
    assert len(plan.sessions) == 2
    assert plan.sessions[0].turns[0].new_input_tokens == 100
    assert plan.total_turns == 3
    assert plan.prefix_cache is False


def test_requests_json_is_reconstruction_compatible_superset():
    recs = [
        session_driver.turn_record(
            session_id="s0", turn_idx=0, input_len=512, output_len=64,
            ttft=0.3, tpot=0.02, request_timestamp=1000.0, post_gap_s=2.5,
            prefix_cache=True),
        session_driver.turn_record(
            session_id="s0", turn_idx=1, input_len=700, output_len=80,
            ttft=0.1, tpot=0.02, request_timestamp=1003.0, post_gap_s=0.0,
            prefix_cache=True),
    ]
    rj = session_driver.build_requests_json(recs)
    # standard reconstruction-ledger arrays present
    for k in ("input_lens", "output_lens", "ttfts", "itls", "request_timestamps"):
        assert len(rj[k]) == 2
    # epoch timestamps preserved (alignment), context grows across turns
    assert rj["request_timestamps"] == [1000.0, 1003.0]
    assert rj["input_lens"][1] > rj["input_lens"][0]
    # agentic extensions present
    for k in ("session_ids", "turn_idx", "post_gap_s", "prefix_cache"):
        assert len(rj[k]) == 2
    assert rj["post_gap_s"] == [2.5, 0.0]
    assert rj["prefix_cache"] == [1, 1]


_ARCH = {"n_layers": 36, "n_kv": 8, "head_dim": 128, "w_bytes": 16.4e9}  # Qwen3-8B
_HBM = 80 * 1024**3  # A100-80GB


def _ac(**kw):
    base = dict(arch=_ARCH, avg_context=15000, tp=1, hbm_bytes_per_gpu=_HBM,
                max_num_seqs=256)
    return session_runner.auto_concurrency(**{**base, **kw})


def test_auto_concurrency_fits_a_healthy_batch():
    assert 1 < _ac() <= 256                       # 8B @ TP1 admits a real batch


def test_auto_concurrency_shrinks_with_context():
    assert _ac(avg_context=4000) > _ac(avg_context=60000)


def test_auto_concurrency_grows_with_tp():
    assert _ac(tp=8) > _ac(tp=1)                   # more GPUs -> more KV budget


def test_auto_concurrency_clamps_to_max_num_seqs():
    assert _ac(avg_context=10, max_num_seqs=4) == 4


def test_auto_concurrency_floors_at_one_when_weights_fill_hbm():
    assert _ac(arch={**_ARCH, "w_bytes": _HBM * 0.99}) == 1


def test_auto_concurrency_fp8_kv_roughly_doubles():
    assert _ac(kv_cache_dtype="fp8") >= 2 * _ac(kv_cache_dtype="auto") - 1


def test_session_window_records_epoch_and_context():
    plan = agentic.build_synthetic_sessions(n_sessions=1, min_turns=3, max_turns=3, seed=0)
    w = session_runner.build_session_window(plan.sessions[0], 1000.0, 1050.0, 3)
    assert w["t_start_epoch"] == 1000.0 and w["t_end_epoch"] == 1050.0
    assert w["n_turns"] == 3 and w["completed_turns"] == 3
    assert w["context_lengths"] == plan.sessions[0].context_lengths()
