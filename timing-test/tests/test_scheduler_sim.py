"""
Claim:
The simulation reproduces hand-worked continuous-batching schedules:
decode-first token budgets, chunked prefill across iterations, first token
emitted by the final prompt chunk, seat and KV admission limits, idle time
jumps, and inter-token latencies equal to the durations of the iterations
between a sequence's consecutive tokens.

Plausible wrong implementations:
- Emit the first token one iteration after prefill completes.
- Let prefill chunks starve decodes (prefill-first) or ignore the budget.
- Admit past max_num_seqs or KV capacity.
- Advance the clock during idle instead of jumping to the next arrival.
- Lose cached-prefix context while charging only the executed suffix.
- Average independent prefill contexts before calculating attention work.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from scheduler_sim import EngineConfig, derive_kv_capacity_tokens, simulate_requests  # noqa: E402

ARCH = dict(n_active=1.0, w_bytes=8.0, d_model=4.0, n_layers=1, n_kv=1,
            head_dim=2, moe_frac=0.0, n_experts=1, top_k=1, swa_window=0, fp8=0)


def _unit_iterations(decode_batch, context_mean, chunk_tokens, chunk_context):
    return 1.0  # one second per iteration, regardless of composition


def _simulate(requests, engine, timing=_unit_iterations):
    return simulate_requests(
        requests, arch=ARCH, hardware="A100", tp=1, eff_flops=1.0, eff_bw=1.0,
        t_launch_s=0.0, engine=engine, iteration_seconds=timing)


def test_chunked_prefill_emits_first_token_with_final_chunk():
    engine = EngineConfig(max_num_seqs=4, chunk_budget_tokens=2,
                          kv_capacity_tokens=1000)
    [r] = _simulate([(0.0, 4, 3)], engine)
    # Prompt of 4 at budget 2: chunks in iterations 1 and 2; first token at
    # t=2; decode tokens at t=3 and t=4.
    assert r["ttft_s"] == pytest.approx(2.0)
    assert r["itl_s"] == pytest.approx([1.0, 1.0])
    assert r["e2e_s"] == pytest.approx(4.0)


def test_decode_first_budget_shares_iteration_with_prefill():
    engine = EngineConfig(max_num_seqs=4, chunk_budget_tokens=3,
                          kv_capacity_tokens=1000)
    log = []

    def recorder(decode_batch, context_mean, chunk_tokens, chunk_context):
        log.append((decode_batch, chunk_tokens))
        return 1.0

    out = _simulate([(0.0, 2, 3), (0.0, 4, 1)], engine, recorder)
    # Iteration 1: no decodes yet; budget 3 covers request A's prompt (2)
    # plus one token of B. Iteration 2: A decodes (1 token of budget),
    # B prefills 2 more; iteration 3: A decodes, B finishes its prompt.
    assert log[0] == (0, 3)
    assert log[1] == (1, 2)
    assert log[2] == (1, 1)
    a, b = out
    assert a["ttft_s"] == pytest.approx(1.0)
    assert b["ttft_s"] == pytest.approx(3.0)
    assert a["e2e_s"] == pytest.approx(3.0)


def test_seat_limit_defers_admission():
    engine = EngineConfig(max_num_seqs=1, chunk_budget_tokens=10,
                          kv_capacity_tokens=1000)
    first, second = _simulate([(0.0, 1, 2), (0.0, 1, 1)], engine)
    # Request one occupies the only seat for 2 iterations (prefill+token at
    # t=1, decode at t=2); request two starts after and finishes at t=3.
    assert first["e2e_s"] == pytest.approx(2.0)
    assert second["ttft_s"] == pytest.approx(3.0)


def test_kv_capacity_blocks_admission():
    engine = EngineConfig(max_num_seqs=8, chunk_budget_tokens=10,
                          kv_capacity_tokens=12.0)
    log = []

    def recorder(decode_batch, context_mean, chunk_tokens, chunk_context):
        log.append((decode_batch, chunk_tokens))
        return 1.0

    _simulate([(0.0, 8, 2), (0.0, 8, 2)], engine, recorder)
    # Each sequence needs 8+1 KV tokens; 12 < 18, so they run serially.
    assert log[0] == (0, 8)
    assert max(d + c for d, c in log) <= 9


def test_iteration_trace_records_composition_on_wall_time():
    engine = EngineConfig(max_num_seqs=4, chunk_budget_tokens=2,
                          kv_capacity_tokens=1000)
    trace = []
    simulate_requests([(0.0, 4, 3)], arch=ARCH, hardware="A100", tp=1,
                      eff_flops=1.0, eff_bw=1.0, t_launch_s=0.0,
                      engine=engine, iteration_seconds=_unit_iterations,
                      iteration_trace=trace)
    # Two prefill chunks then two decode iterations, one second each; the
    # single sequence contributes one prefill chunk per prefill iteration.
    assert [(row[0], row[1], row[2], row[4], row[6]) for row in trace] == [
        (0.0, 1.0, 0, 2, 1), (1.0, 2.0, 0, 2, 1),
        (2.0, 3.0, 1, 0, 0), (3.0, 4.0, 1, 0, 0)]


def test_idle_time_jumps_to_next_arrival():
    engine = EngineConfig(max_num_seqs=4, chunk_budget_tokens=10,
                          kv_capacity_tokens=1000)
    first, second = _simulate([(0.0, 1, 1), (10.0, 1, 1)], engine)
    assert first["e2e_s"] == pytest.approx(1.0)
    assert second["ttft_s"] == pytest.approx(1.0)  # starts at 10.0, done 11.0


def test_kv_capacity_derivation_subtracts_weights():
    arch = dict(ARCH, w_bytes=40e9)
    capacity = derive_kv_capacity_tokens(arch, hardware="A100", tp=1)
    # (0.9*80e9 - 40e9) / (2*1*1*2*2) = 32e9 / 8
    assert capacity == pytest.approx(4e9)
    with pytest.raises(ValueError):
        derive_kv_capacity_tokens(dict(ARCH, w_bytes=100e9), hardware="A100", tp=1)


def test_cached_prefix_counts_for_context_and_kv_but_not_prefill_tokens():
    engine = EngineConfig(
        max_num_seqs=1, chunk_budget_tokens=8, kv_capacity_tokens=15
    )
    trace = []
    [row] = simulate_requests(
        [(0.0, 4, 1, 10)], arch=ARCH, hardware="A100", tp=1,
        eff_flops=1.0, eff_bw=1.0, t_launch_s=0.0, engine=engine,
        iteration_seconds=_unit_iterations, iteration_trace=trace,
    )
    assert row["initial_context"] == 10
    assert row["n_in"] == 4
    assert trace[0][4] == 4
    assert trace[0][5] == 10


def test_scheduler_passes_each_prefill_chunk_context_to_work_model(monkeypatch):
    import scheduler_sim

    captured = []
    original = scheduler_sim.iteration_work

    def recording_work(*args, **kwargs):
        captured.append(kwargs["prefill_chunks"])
        return original(*args, **kwargs)

    monkeypatch.setattr(scheduler_sim, "iteration_work", recording_work)
    engine = EngineConfig(
        max_num_seqs=2, chunk_budget_tokens=4, kv_capacity_tokens=1000
    )
    simulate_requests(
        [(0.0, 1, 1), (0.0, 3, 1, 100)],
        arch=ARCH, hardware="A100", tp=1, eff_flops=1.0, eff_bw=1.0,
        t_launch_s=0.0, engine=engine, iteration_seconds=_unit_iterations,
    )
    assert captured[0] == [(1, 0), (3, 100)]
