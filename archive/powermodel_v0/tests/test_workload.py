"""Unit tests for the predict-time inference chain (saturation + operating points)."""

import numpy as np
import pytest

from powermodel import workload as W
from powermodel.tests.test_arch import GEMMA_DENSE, GEMMA_MOE


def test_kv_capacity_decreases_with_context():
    cap_short = W.kv_capacity_batch(GEMMA_DENSE, "A100", 2.0, 256.0)
    cap_long = W.kv_capacity_batch(GEMMA_DENSE, "A100", 2.0, 8192.0)
    assert cap_short > cap_long > 0


def test_saturation_caps_requested_concurrency():
    srv = W.ServerConfig(max_num_seqs=256)
    # request 1000 concurrent but max_num_seqs=256 -> capped at <=256
    wl = W.Workload(input_len=8, output_len=2048, concurrency=1000)
    B = W.sustained_decode_batch(GEMMA_DENSE, "A100", 2.0, srv, wl)
    assert B <= 256.0


def test_decode_step_time_increases_with_batch_for_moe():
    # MoE: more experts touched at larger batch -> more weight bytes -> slower step
    t1 = W.decode_step_time(GEMMA_MOE, "A100", 2.0, 1.0)
    t256 = W.decode_step_time(GEMMA_MOE, "A100", 2.0, 256.0)
    assert t256 > t1


def test_decode_operating_point_throughput_positive():
    st = W.decode_operating_point(GEMMA_DENSE, "A100", 2.0, batch=16.0,
                                  ctx=1000.0, n_bins=10)
    assert np.all(st["dec_tok"] > 0)
    assert np.all(st["decode_batch"] == 16.0)
    assert np.all(st["pre_tok"] == 0)
    # decode tokens/s = batch * iters/s
    assert st["dec_tok"][0] == pytest.approx(16.0 * st["iters_dec"][0])


def test_prefill_operating_point_longer_context_slower_per_seq():
    st_short = W.prefill_operating_point(GEMMA_DENSE, "A100", 2.0, 256.0, n_bins=4)
    st_long = W.prefill_operating_point(GEMMA_DENSE, "A100", 2.0, 65536.0, n_bins=4)
    # long context: attention dominates -> lower iterations/s (slower per seq)
    assert st_long["iters_pre"][0] < st_short["iters_pre"][0]


def test_serve_mixed_has_both_phases():
    srv = W.ServerConfig()
    wl = W.Workload(input_len=512, output_len=256, request_rate=4.0)
    st = W.serve(GEMMA_DENSE, "A100", 2.0, srv, wl, n_bins=4)
    assert np.all(st["dec_tok"] > 0)
    assert np.all(st["pre_tok"] > 0)
    assert np.all(st["decode_batch"] >= 0)
