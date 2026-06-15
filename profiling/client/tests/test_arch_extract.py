"""Acceptance gate for ``arch_extract``: it must reproduce the curated arch
descriptors before it is trusted on new models, and assign sane values + the
additive flags for the new lineup.

Fixtures are minimal inline config dicts holding only the fields the extractor
reads. Expected values mirror ``feature-test/build_ledger_cache.py`` ARCH (kept
inline here to keep the test self-contained across packages).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client

from arch_extract import extract_arch  # noqa: E402

GIB = 1024.0**3

# --------------------------------------------------------------------------- #
# Existing-model configs (subset of fields the extractor reads)
# --------------------------------------------------------------------------- #

LLAMA_8B = dict(
    hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
    num_key_value_heads=8, head_dim=128, intermediate_size=14336,
    vocab_size=128256, tie_word_embeddings=False, torch_dtype="bfloat16",
)
LLAMA_70B = dict(
    hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
    num_key_value_heads=8, head_dim=128, intermediate_size=28672,
    vocab_size=128256, tie_word_embeddings=False, torch_dtype="bfloat16",
)
LLAMA_405B = dict(
    hidden_size=16384, num_hidden_layers=126, num_attention_heads=128,
    num_key_value_heads=8, head_dim=128, intermediate_size=53248,
    vocab_size=128256, tie_word_embeddings=False, torch_dtype="bfloat16",
)
# gpt-oss is MXFP4 -> w_bytes is the measured shard footprint, passed in.
GPT_OSS_120B = dict(
    hidden_size=2880, num_hidden_layers=36, num_attention_heads=64,
    num_key_value_heads=8, head_dim=64, intermediate_size=2880,
    moe_intermediate_size=2880, num_local_experts=128, num_experts_per_tok=4,
    sliding_window=128, vocab_size=201088, tie_word_embeddings=False,
    quantization_config={"quant_method": "mxfp4"},
)
GPT_OSS_20B = dict(
    hidden_size=2880, num_hidden_layers=24, num_attention_heads=64,
    num_key_value_heads=8, head_dim=64, intermediate_size=2880,
    moe_intermediate_size=2880, num_local_experts=32, num_experts_per_tok=4,
    sliding_window=128, vocab_size=201088, tie_word_embeddings=False,
    quantization_config={"quant_method": "mxfp4"},
)

# Expected curated values (mirror build_ledger_cache.ARCH).
EXPECTED_DENSE = {
    "llama-3-8b": dict(n_active=8.03e9, d_model=4096, n_layers=32, n_kv=8,
                       head_dim=128, fp8=0, cfg=LLAMA_8B, dtype_hint=None),
    "llama-3-70b": dict(n_active=70.55e9, d_model=8192, n_layers=80, n_kv=8,
                        head_dim=128, fp8=0, cfg=LLAMA_70B, dtype_hint=None),
    "llama-3-405b": dict(n_active=405.85e9, d_model=16384, n_layers=126, n_kv=8,
                         head_dim=128, fp8=1, cfg=LLAMA_405B, dtype_hint="FP8"),
}


@pytest.mark.parametrize("name", list(EXPECTED_DENSE))
def test_reproduces_curated_dict_dense(name):
    exp = EXPECTED_DENSE[name]
    a = extract_arch(exp["cfg"], dtype_hint=exp["dtype_hint"])
    assert a["n_active"] == pytest.approx(exp["n_active"], rel=0.01)
    assert a["d_model"] == exp["d_model"]
    assert a["n_layers"] == exp["n_layers"]
    assert a["n_kv"] == exp["n_kv"]
    assert a["head_dim"] == exp["head_dim"]
    assert a["fp8"] == exp["fp8"]
    assert a["moe_frac"] == 0.0
    assert a["n_experts"] == 1 and a["top_k"] == 1
    # Dense w_bytes = bytes_per_param * n_active (2 for bf16, 1 for fp8).
    bpp = 1.0 if exp["fp8"] else 2.0
    assert a["w_bytes"] == pytest.approx(bpp * a["n_active"], rel=1e-6)
    # additive fields stay zero for existing softmax/dense models
    assert a["linear_attention"] == 0
    assert a["swa_global_ratio"] == 0.0


@pytest.mark.parametrize(
    "cfg,n_experts,top_k,curated_w_gib,curated_n_active",
    [
        (GPT_OSS_120B, 128, 4, 60.8, 5.1e9),
        (GPT_OSS_20B, 32, 4, 12.8, 3.6e9),
    ],
)
def test_reproduces_curated_dict_moe(cfg, n_experts, top_k, curated_w_gib, curated_n_active):
    # MXFP4 footprint is measured, not derivable -> supplied to the extractor.
    a = extract_arch(cfg, weight_footprint_bytes=curated_w_gib * GIB)
    assert a["n_experts"] == n_experts
    assert a["top_k"] == top_k
    assert a["fp8"] == 0  # MXFP4 is not fp8 (matches curated convention)
    assert a["w_bytes"] == pytest.approx(curated_w_gib * GIB, rel=1e-6)
    assert 0.0 < a["moe_frac"] < 1.0
    assert a["swa_window"] == 128
    # MoE analytic active-param count is approximate; require the right order.
    assert a["n_active"] == pytest.approx(curated_n_active, rel=0.30)
    assert a["linear_attention"] == 0


def test_mxfp4_not_treated_as_two_bytes_per_param():
    a = extract_arch(GPT_OSS_20B, weight_footprint_bytes=12.8 * GIB)
    # If MXFP4 were mistaken for bf16, w_bytes would be ~2x the param count
    # (>>12.8 GiB). The footprint path guards this.
    assert a["w_bytes"] == pytest.approx(12.8 * GIB, rel=1e-6)


def test_dtype_to_bytes():
    fp8 = extract_arch(dict(LLAMA_70B, torch_dtype="float8_e4m3fn"))
    assert fp8["fp8"] == 1
    assert fp8["w_bytes"] == pytest.approx(1.0 * fp8["n_active"], rel=1e-6)
    bf16 = extract_arch(LLAMA_70B)
    assert bf16["fp8"] == 0
    assert bf16["w_bytes"] == pytest.approx(2.0 * bf16["n_active"], rel=1e-6)


# --------------------------------------------------------------------------- #
# New-model configs: sanity bounds + additive-flag detection
# --------------------------------------------------------------------------- #

QWEN3_235B_A22B = dict(
    hidden_size=4096, num_hidden_layers=94, num_attention_heads=64,
    num_key_value_heads=4, head_dim=128, moe_intermediate_size=1536,
    num_experts=128, num_experts_per_tok=8, vocab_size=151936,
    tie_word_embeddings=False, torch_dtype="bfloat16", decoder_sparse_step=1,
)
QWEN3_30B_A3B = dict(
    hidden_size=2048, num_hidden_layers=48, num_attention_heads=32,
    num_key_value_heads=4, head_dim=128, moe_intermediate_size=768,
    num_experts=128, num_experts_per_tok=8, vocab_size=151936,
    tie_word_embeddings=True, torch_dtype="bfloat16",
)
QWEN3_32B = dict(
    hidden_size=5120, num_hidden_layers=64, num_attention_heads=64,
    num_key_value_heads=8, head_dim=128, intermediate_size=25600,
    vocab_size=151936, tie_word_embeddings=False, torch_dtype="bfloat16",
)
QWEN3_8B = dict(
    hidden_size=4096, num_hidden_layers=36, num_attention_heads=32,
    num_key_value_heads=8, head_dim=128, intermediate_size=12288,
    vocab_size=151936, tie_word_embeddings=False, torch_dtype="bfloat16",
)
QWEN3_14B = dict(
    hidden_size=5120, num_hidden_layers=40, num_attention_heads=40,
    num_key_value_heads=8, head_dim=128, intermediate_size=17408,
    vocab_size=151936, tie_word_embeddings=False, torch_dtype="bfloat16",
)
GEMMA3_27B = dict(
    hidden_size=5376, num_hidden_layers=62, num_attention_heads=32,
    num_key_value_heads=16, head_dim=128, intermediate_size=21504,
    sliding_window=1024, sliding_window_pattern=6, vocab_size=262144,
    tie_word_embeddings=True, torch_dtype="bfloat16",
)
GEMMA3_12B = dict(
    hidden_size=3840, num_hidden_layers=48, num_attention_heads=16,
    num_key_value_heads=8, head_dim=256, intermediate_size=15360,
    sliding_window=1024, sliding_window_pattern=6, vocab_size=262144,
    tie_word_embeddings=True, torch_dtype="bfloat16",
)
LLAMA4_SCOUT = dict(
    text_config=dict(
        hidden_size=5120, num_hidden_layers=48, num_attention_heads=40,
        num_key_value_heads=8, head_dim=128, intermediate_size=8192,
        intermediate_size_mlp=16384, num_local_experts=16, num_experts_per_tok=1,
        vocab_size=202048, tie_word_embeddings=False,
    ),
    torch_dtype="bfloat16",
)
MINIMAX_M27 = dict(
    hidden_size=3072, num_hidden_layers=62, num_attention_heads=32,
    num_key_value_heads=8, head_dim=128, moe_intermediate_size=1024,
    num_experts=256, num_experts_per_tok=8, vocab_size=200064,
    tie_word_embeddings=False, torch_dtype="bfloat16",
    attention_type="hybrid", linear_attn_period=8,
)


def test_n_active_override_supersedes_analytic():
    """MoE analytic n_active is approximate; an explicit count must win."""
    a = extract_arch(QWEN3_235B_A22B, n_active_override=22.0e9)
    assert a["n_active"] == 22.0e9
    # dense w_bytes still tracks n_total, not the override
    assert a["n_experts"] == 128


def test_top_k_does_not_latch_onto_n_group():
    """DeepSeek-style n_group (expert-GROUP count) must not be used as top-k."""
    # real per-token count present -> used
    a = extract_arch(dict(QWEN3_235B_A22B, n_group=8))
    assert a["top_k"] == 8  # from num_experts_per_tok
    # only n_group present (no real top-k) -> NOT detected as MoE on n_group alone
    cfg = dict(QWEN3_235B_A22B)
    cfg.pop("num_experts_per_tok")
    cfg["n_group"] = 8
    a2 = extract_arch(cfg)
    assert a2["n_experts"] == 1  # falls back to dense, not top_k=8


def test_qwen3_235b_active_in_range():
    a = extract_arch(QWEN3_235B_A22B)
    assert 18e9 <= a["n_active"] <= 26e9  # "A22B"
    assert a["n_experts"] == 128 and a["top_k"] == 8
    assert 0.0 < a["moe_frac"] < 1.0
    assert a["linear_attention"] == 0


def test_qwen3_30b_a3b_active_in_range():
    a = extract_arch(QWEN3_30B_A3B)
    assert 2e9 <= a["n_active"] <= 4.5e9  # "A3B"


@pytest.mark.parametrize(
    "cfg,lo,hi",
    [(QWEN3_8B, 7e9, 9e9), (QWEN3_14B, 12e9, 16e9), (QWEN3_32B, 30e9, 35e9)],
)
def test_qwen3_dense_ladder_in_range(cfg, lo, hi):
    a = extract_arch(cfg)
    assert lo <= a["n_active"] <= hi
    assert a["moe_frac"] == 0.0 and a["n_experts"] == 1


def test_llama4_scout_nested_text_config():
    a = extract_arch(LLAMA4_SCOUT)
    assert a["d_model"] == 5120 and a["n_layers"] == 48  # descended into text_config
    assert a["n_experts"] == 16
    # ~17B active; fixture config is approximate, so accept the order of magnitude.
    assert 12e9 <= a["n_active"] <= 26e9


@pytest.mark.parametrize("cfg", [GEMMA3_27B, GEMMA3_12B])
def test_gemma3_sliding_window(cfg):
    a = extract_arch(cfg)
    assert a["swa_window"] == 1024
    assert a["swa_global_ratio"] == pytest.approx(5.0)  # 5 local : 1 global
    assert a["n_experts"] == 1  # dense
    assert a["linear_attention"] == 0


def test_minimax_linear_attention_flag():
    a = extract_arch(MINIMAX_M27)
    assert a["linear_attention"] == 1
    assert a["n_linear_layers"] > 0
    assert a["n_experts"] == 256 and a["top_k"] == 8
    assert 6e9 <= a["n_active"] <= 14e9  # ~9.8B active
