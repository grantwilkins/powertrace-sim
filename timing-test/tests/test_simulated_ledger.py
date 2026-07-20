"""
Claim:
The simulated-ledger emitter projects the scheduler's iteration trace onto
uniform bins with the measured ledger's channel semantics: prefill tokens
spread over their iteration, decode tokens as completion events at the
iteration end (so a run's dec_tok integrates to sum(n_out - 1), matching
the measured exact-ITL path), batch as the time-averaged decode batch,
KV reads weighted by per-token context, exact weight bytes conserved from
the timing model's engine iterations, queue-state channels from the simulator's
true admission times, and exact iteration coordinates from the simulator's
iteration-completion records.

Plausible wrong implementations:
- Drop the token that completes exactly at the end of the grid.
- Spread decode work over the iteration instead of an end event (shifts
  work one bin early relative to the measured convention).
- Count the prefill-emitted first token in dec_tok (double counting).
- Compute batch from events (iterations per bin) instead of coverage.
- Reuse dec_tok / batch as iteration rate, creating boundary-bin spikes.
- Double-count a mixed prefill/decode iteration as two engine iterations.
- Divide weight bytes by bin width or iteration duration twice.
- Lose weight bytes when an iteration crosses a bin boundary.
- Confuse admitted and arrived when splitting running/waiting.
- Count every prefill token as a vocabulary projection.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[2]))
from scheduler_sim import EngineConfig, simulate_requests  # noqa: E402
from simulated_ledger import PER_BIN_CHANNELS, emit_bins  # noqa: E402
from model.training_data.ledger_view import kv_bytes_per_token  # noqa: E402
from model.training_data.moe_routing import RoutingLaw  # noqa: E402

ARCH = dict(n_active=1.0, w_bytes=8.0, d_model=4.0, n_layers=1, n_kv=1,
            head_dim=2, moe_frac=0.0, n_experts=1, top_k=1, swa_window=0,
            fp8=0, family="toy")


def _unit_iterations(decode_batch, context_mean, chunk_tokens, chunk_context):
    return 1.0


def _simulate(requests, chunk_budget=2):
    engine = EngineConfig(max_num_seqs=4, chunk_budget_tokens=chunk_budget,
                          kv_capacity_tokens=1000)
    trace = []
    per_request = simulate_requests(
        requests, arch=ARCH, hardware="A100", tp=1, eff_flops=1.0, eff_bw=1.0,
        t_launch_s=0.0, engine=engine, iteration_seconds=_unit_iterations,
        iteration_trace=trace)
    return trace, per_request


def test_hand_worked_single_request_bins():
    # (0.0, 4, 3) with chunk budget 2 and 1 s iterations: prefill [0,1) and
    # [1,2) of 2 tokens each (first output token at t=2), decode iterations
    # [2,3) and [3,4) with batch 1, contexts 5 and 6.
    trace, per_request = _simulate([(0.0, 4, 3)])
    out = emit_bins(trace, per_request, arch=ARCH, tp=1, dt=0.5)

    assert np.allclose(out["pre_tok"], [2, 2, 2, 2, 0, 0, 0, 0, 0])
    # Decode completions are events at iteration ends t=3 and t=4, one token
    # each -> rate 1/0.5. Right-open bins put an edge event in the later bin
    # ([3.0, 3.5)), and the t=4 token lands in the extra bin past t_max
    # instead of being dropped.
    assert np.allclose(out["dec_tok"], [0, 0, 0, 0, 0, 0, 2, 0, 2])
    assert float(out["dec_tok"].sum() * 0.5) == pytest.approx(2.0)  # n_out - 1
    # batch is coverage: decode batch 1 throughout [2, 4).
    assert np.allclose(out["batch"], [0, 0, 0, 0, 1, 1, 1, 1, 0])
    assert np.allclose(out["busy"], [1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert np.allclose(out["pre_active"], [1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert np.allclose(out["prefill_duty"], [1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert np.allclose(out["decode_duty"], [0, 0, 0, 0, 1, 1, 1, 1, 0])
    assert np.allclose(out["engine_iterations_rate"],
                       [0, 0, 2, 0, 2, 0, 2, 0, 2])
    assert np.allclose(out["engine_iteration_tokens_rate"],
                       [0, 0, 4, 0, 4, 0, 2, 0, 2])
    assert np.allclose(out["engine_tokens_per_iteration"],
                       [0, 0, 2, 0, 2, 0, 1, 0, 1])
    # One output projection on the final prefill chunk and one on each of
    # the two decode iterations.
    assert float(out["logit_tokens_rate"].sum() * 0.5) == 3.0
    assert float(out["engine_iterations_rate"].sum() * 0.5) == 4.0
    # KV read events carry the decoding context (5 then 6 tokens).
    kv_tok = kv_bytes_per_token(ARCH)
    assert np.allclose(out["kv_read"],
                       np.array([0, 0, 0, 0, 0, 0, 5, 0, 6]) * kv_tok / 0.5)
    assert out["n"] == 9 and len(out["A_t"]) == 9


def test_token_conservation_across_offset_arrivals():
    requests = [(0.0, 5, 4), (0.3, 3, 6), (2.7, 7, 2)]
    trace, per_request = _simulate(requests, chunk_budget=3)
    out = emit_bins(trace, per_request, arch=ARCH, tp=1, dt=0.25)
    assert float(out["pre_tok"].sum() * 0.25) == pytest.approx(5 + 3 + 7)
    assert float(out["dec_tok"].sum() * 0.25) == pytest.approx(3 + 5 + 1)
    assert float(out["arrivals"].sum() * 0.25) == pytest.approx(3)
    assert float(out["input_tokens_arriving"].sum() * 0.25) == pytest.approx(15)
    assert float(out["output_tokens_requested"].sum() * 0.25) == pytest.approx(12)


def test_schema_matches_measured_ledger_channels():
    trace, per_request = _simulate([(0.0, 4, 3)])
    out = emit_bins(trace, per_request, arch=ARCH, tp=1, dt=0.5)
    measured_channels = {
        "pre_tok", "dec_tok", "batch", "pre_active", "iters",
        "w_read", "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm",
        "arrivals", "input_tokens_arriving", "output_tokens_requested",
        "A_t", "delta_A_t", "running_requests", "waiting_requests",
    }
    assert measured_channels <= set(out)
    simulator_channels = {
        "busy", "prefill_duty", "decode_duty",
        "engine_iteration_tokens_rate", "engine_iterations_rate",
        "engine_tokens_per_iteration", "logit_tokens_rate",
    }
    assert set(PER_BIN_CHANNELS) == measured_channels | simulator_channels
    nb = out["n"]
    assert all(np.asarray(out[key]).shape == (nb,) for key in PER_BIN_CHANNELS)


def test_waiting_vs_running_uses_admission_time():
    # Second request arrives at t=0 but only one seat exists: it waits until
    # the first request finishes at t=2 (prefill [0,1) then decode [1,2)).
    engine = EngineConfig(max_num_seqs=1, chunk_budget_tokens=8,
                          kv_capacity_tokens=1000)
    trace = []
    per_request = simulate_requests(
        [(0.0, 2, 2), (0.0, 2, 2)], arch=ARCH, hardware="A100", tp=1,
        eff_flops=1.0, eff_bw=1.0, t_launch_s=0.0, engine=engine,
        iteration_seconds=_unit_iterations, iteration_trace=trace)
    assert per_request[0]["admitted_s"] == pytest.approx(0.0)
    assert per_request[1]["admitted_s"] == pytest.approx(2.0)
    out = emit_bins(trace, per_request, arch=ARCH, tp=1, dt=1.0)
    # Bins end at 1,2,3,4,5: both unfinished until each finishes at 2 and 4.
    assert np.allclose(out["A_t"], [2, 1, 1, 0, 0])
    assert np.allclose(out["running_requests"], [1, 1, 1, 0, 0])
    assert np.allclose(out["waiting_requests"], [1, 0, 0, 0, 0])


def test_mixed_iteration_is_counted_once():
    # Request 1 decodes while request 2 prefills in the same second engine
    # iteration. It is one iteration with three scheduled tokens, not one
    # decode iteration plus one prefill iteration.
    trace = [
        (0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0),
        (1.0, 2.0, 1.0, 2.0, 2.0, 0.0, 1.0),
    ]
    per_request = [
        {"arrival_s": 0.0, "admitted_s": 0.0, "n_in": 2, "n_out": 2,
         "e2e_s": 2.0},
        {"arrival_s": 1.0, "admitted_s": 1.0, "n_in": 2, "n_out": 1,
         "e2e_s": 1.0},
    ]
    out = emit_bins(trace, per_request, arch=ARCH, tp=1, dt=1.0)
    assert np.allclose(out["engine_iterations_rate"], [0.0, 1.0, 1.0])
    assert np.allclose(out["engine_tokens_per_iteration"], [0.0, 2.0, 3.0])
    assert out["prefill_duty"][1] == 1.0
    assert out["decode_duty"][1] == 1.0


def test_mixed_iteration_weight_bytes_are_conserved_across_bins():
    # One 0.5 s mixed iteration reads the toy model's 8 weight bytes once.
    # Its [0.1, 0.6) interval overlaps 0.25 s bins by 0.15, 0.25, 0.10 s,
    # so the per-bin rates are 8 * overlap / 0.5 / 0.25.
    trace = [(0.1, 0.6, 2.0, 4.0, 2.0, 0.0, 1.0)]
    per_request = [
        {"arrival_s": 0.0, "admitted_s": 0.0, "n_in": 1, "n_out": 1,
         "e2e_s": 0.6},
    ]
    out = emit_bins(trace, per_request, arch=ARCH, tp=1, dt=0.25)

    np.testing.assert_allclose(out["w_read"], [9.6, 16.0, 6.4])
    np.testing.assert_allclose(out["w_read_pre"], [4.8, 8.0, 3.2])
    np.testing.assert_allclose(out["w_read_dec"], [4.8, 8.0, 3.2])
    np.testing.assert_allclose(
        out["w_read"], out["w_read_pre"] + out["w_read_dec"])
    assert float(out["w_read"].sum() * 0.25) == pytest.approx(ARCH["w_bytes"])


def test_moe_mixed_weight_union_is_conserved_across_bins():
    arch = dict(ARCH, moe_frac=0.5, n_experts=2, top_k=1)
    law = RoutingLaw(
        model="toy", source="toy", top_k=1, n_experts=2,
        prefill_alpha=1.0, decode_alpha=1.0,
        prefill_touch_probability=np.full((1, 2), 0.5),
        decode_touch_probability=np.full((1, 2), 0.5),
    )
    trace = [(0.0, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0)]
    per_request = [{
        "arrival_s": 0.0, "admitted_s": 0.0, "n_in": 1, "n_out": 1,
        "e2e_s": 1.0,
    }]
    out = emit_bins(
        trace, per_request, arch=arch, tp=1, dt=0.5, routing_law=law)

    # Shared half plus a 3/4 expert union: 8*(.5 + .5*.75) = 7 bytes.
    assert float(out["w_read"].sum() * 0.5) == pytest.approx(7.0)
    np.testing.assert_allclose(
        out["w_read"], out["w_read_pre"] + out["w_read_dec"])
