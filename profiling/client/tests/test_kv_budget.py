"""Pure tests for the KV-cache budgeting math (reserved-memory model)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client

import kv_budget  # noqa: E402


def test_kv_bytes_per_token_formula():
    # 2(K&V) * layers * kv_heads * head_dim * dtype_bytes
    assert kv_budget.kv_bytes_per_token(36, 8, 128, 2) == 2 * 36 * 8 * 128 * 2


def test_length_budget_tracks_context_window():
    mp, mt = kv_budget.length_budget(32768)
    assert mt == 32768           # prompt+output <= max_model_len
    assert mp == 32768 - 4       # minus min output room
    # far above the old fixed 1024/2048 that dropped long prompts
    assert mp > 1024 and mt > 2048


def test_kv_pool_after_weights_is_smaller_than_raw_hbm():
    gpu = 80 * kv_budget.GIB
    w = 16 * kv_budget.GIB
    pool = kv_budget.kv_pool_bytes(gpu, w, tp=1, gpu_memory_utilization=0.9)
    assert 0 < pool < gpu
    # weights + overhead are excluded, so pool < util*hbm - weights
    assert pool <= gpu * 0.9 - w + 1


def test_kv_capacity_and_suggested_len_are_sane():
    # Qwen3-8B-ish on an 80GB A100: capacity should be large, single-seq len huge.
    cap = kv_budget.kv_capacity_tokens(
        80 * kv_budget.GIB, 16 * kv_budget.GIB,
        n_layers=36, n_kv_heads=8, head_dim=128, tp=1)
    assert cap > 100_000
    assert kv_budget.suggested_max_model_len(cap, target_concurrency=1) > cap // 2
    # higher concurrency -> shorter per-request budget
    assert (kv_budget.suggested_max_model_len(cap, 64)
            < kv_budget.suggested_max_model_len(cap, 1))


def test_tp_scales_pool():
    gpu = 80 * kv_budget.GIB
    w = 140 * kv_budget.GIB  # a big model needing multiple GPUs
    assert kv_budget.kv_pool_bytes(gpu, w, tp=1) == 0.0          # doesn't fit on 1
    assert kv_budget.kv_pool_bytes(gpu, w, tp=4) > 0.0           # fits across 4
