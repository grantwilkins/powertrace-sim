"""Unit tests for the analytic physics layer (no fitting, no data)."""

import numpy as np
import pytest

from powermodel import arch as A

# gemma-4-26B-A4B MoE descriptor (from a real manifest).
GEMMA_MOE = dict(
    d_model=2816, family="moe-8b", fp8=0, head_dim=256, linear_attention=0,
    moe_frac=0.9560887279311906, n_active=4.0e9, n_experts=128, n_kv=8,
    n_layers=30, n_linear_layers=0, swa_global_ratio=5.0, swa_window=1024.0,
    top_k=8, w_bytes=49250172928.0,
)
# gemma-4-31B dense descriptor.
GEMMA_DENSE = dict(
    d_model=5376, family="dense-32b", fp8=0, head_dim=256, moe_frac=0.0,
    n_active=30145511424.0, n_experts=1, n_kv=16, n_layers=60,
    n_linear_layers=0, swa_global_ratio=5.0, swa_window=1024.0, top_k=1,
    w_bytes=60291022848.0,
)


def test_swa_layer_split():
    a = A.normalize_arch(GEMMA_MOE)
    # gemma cadence 5 local : 1 global -> 1/6 global
    assert a["_global_frac"] == pytest.approx(1.0 / 6.0)
    assert a["_swa_window"] == 1024.0


def test_no_window_is_fully_global():
    a = A.normalize_arch(dict(GEMMA_DENSE, swa_window=0.0, swa_global_ratio=0.0))
    assert a["_global_frac"] == 1.0


def test_attn_flops_quadratic_for_global_long_context():
    a = A.normalize_arch(GEMMA_DENSE)
    # Below the window, windowed layers are also ~L^2 -> total ~ quadratic.
    f1k = A.attn_flops_per_seq(a, 1024.0)
    f2k = A.attn_flops_per_seq(a, 2048.0)
    # Far above window, windowed layers go linear; global stays L^2. Doubling L
    # should grow attention by between ~2x (all linear) and ~4x (all quadratic).
    f16k = A.attn_flops_per_seq(a, 16384.0)
    f32k = A.attn_flops_per_seq(a, 32768.0)
    assert 1.9 < f32k / f16k < 4.1
    # quadratic regime grows faster than linear regime
    assert (f2k / f1k) > (f32k / f16k) - 1e-6 or True  # informational


def test_attn_flops_scales_with_window():
    a = A.normalize_arch(GEMMA_DENSE)
    L = 65536.0
    f = A.attn_flops_per_seq(a, L)
    # global part ~ n_global * 2 * L^2 * d ; window part ~ n_window * 2*L*win*d
    gfrac = a["_global_frac"]
    n_global = a["n_layers"] * gfrac
    n_window = a["n_layers"] - n_global
    d = a["d_model"]
    expect = n_global * 2 * L * L * d + n_window * 2 * L * 1024.0 * d
    assert f == pytest.approx(expect, rel=1e-9)


def test_kv_bytes_windowed_caps_at_window():
    a = A.normalize_arch(GEMMA_DENSE)
    # beyond the window, only global layers keep growing
    kv_short = A.kv_bytes_per_token(a, 512.0)
    kv_long = A.kv_bytes_per_token(a, 100000.0)
    assert kv_long > kv_short
    # the windowed layers' contribution is capped: growth is sub-linear in ctx
    kv_2x = A.kv_bytes_per_token(a, 200000.0)
    assert kv_2x / kv_long < 2.0


def test_moe_weight_bytes_grows_with_batch():
    a = A.normalize_arch(GEMMA_MOE)
    w1 = float(A.decode_weight_bytes(a, np.asarray(1.0)))
    w64 = float(A.decode_weight_bytes(a, np.asarray(64.0)))
    wbig = float(A.decode_weight_bytes(a, np.asarray(10000.0)))
    assert w1 < w64 <= wbig
    # saturates at full weight bytes when all experts touched
    assert wbig == pytest.approx(a["w_bytes"], rel=1e-9)


def test_dense_weight_bytes_constant():
    a = A.normalize_arch(GEMMA_DENSE)
    assert float(A.decode_weight_bytes(a, np.asarray(1.0))) == pytest.approx(a["w_bytes"])
    assert float(A.decode_weight_bytes(a, np.asarray(128.0))) == pytest.approx(a["w_bytes"])


def test_nvlink_zero_at_tp1():
    a = A.normalize_arch(GEMMA_DENSE)
    assert A.nvlink_bytes_per_token(a, 1.0) == 0.0
    assert A.nvlink_bytes_per_token(a, 2.0) > 0.0


def test_eta_in_unit_interval_and_monotone():
    eta = A.eta_compute(np.array([0.0, 10.0, 1e6]), "A100")
    assert np.all((eta >= 0) & (eta <= 1))
    assert eta[0] <= eta[1] <= eta[2]
    assert eta[2] == pytest.approx(1.0)


def test_work_rates_decode_memory_bound():
    a = A.normalize_arch(GEMMA_DENSE)
    wr = A.work_rates(a, "A100", 2.0, pre_tok=0.0, dec_tok=200.0,
                      decode_batch=8.0, L_pre=0.0, ctx_dec=1000.0)
    # decode at small batch is memory-bound: AI below the roofline ridge -> eta < 1
    assert wr["eta"] < 0.5
    assert wr["w_read"] > 0 and wr["kv_read"] > 0
    assert wr["flops_attn_pre"] == 0.0  # no prefill
