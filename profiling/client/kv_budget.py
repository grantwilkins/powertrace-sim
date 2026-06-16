"""KV-cache memory budgeting — how serving context length is bounded by the GPU.

Why this exists: the dataset pruner used a fixed ``max_prompt_len=1024`` /
``max_total_len=2048``, which silently drops any prompt longer than 1024 tokens —
so a long-context workload never reaches the server even when the model/GPU can
serve it. The cap should instead track the **served context window**
(``max_model_len``), which operators size from the GPU's KV-cache budget.

The standard model (what vLLM does internally; rule-of-thumb numbers below):

    usable          = gpu_mem_bytes * tp * gpu_memory_utilization   # util≈0.9 default
    kv_pool         = usable - weight_bytes - activation_overhead
    kv_bytes/token  = 2(K&V) * n_layers * n_kv_heads * head_dim * kv_dtype_bytes
    kv_capacity_tok = kv_pool / (kv_bytes/token)

``kv_capacity_tok`` bounds the SUM of tokens over the running batch (batch ×
seqlen), not a single request — a single request's ceiling is ``max_model_len``.
Rule of thumb: after weights, ~40–60% of HBM ends up as KV. ``gpu_memory_utilization``
defaults to 0.9.

How people push past HBM today (so caps shouldn't be artificially small):
  * LMCache — tiered KV offload (GPU→CPU DRAM→NVMe→Redis/S3) + cross-engine prefix
    reuse and prefill/decode disaggregation; reports 1.9–8.1× TTFT cuts on long
    context with ~50% hit rates in RAG / multi-turn.
  * Mooncake (Kimi/Moonshot) — a KVCache-centric *disaggregated* architecture:
    separate prefill and decode clusters, a global KV pool over CPU/DRAM/SSD/NIC,
    and an RDMA/NVLink Transfer Engine + Store; trades storage for compute.

All functions here are pure (unit-tested).
"""

from __future__ import annotations

GIB = 1024.0**3


def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int,
                       kv_dtype_bytes: float = 2.0) -> float:
    """Bytes of KV cache one token occupies (both K and V, all layers)."""
    return 2.0 * int(n_layers) * int(n_kv_heads) * int(head_dim) * float(kv_dtype_bytes)


def kv_pool_bytes(gpu_mem_bytes: float, weight_bytes: float, tp: int = 1,
                  gpu_memory_utilization: float = 0.9,
                  activation_overhead_frac: float = 0.1) -> float:
    """KV-cache pool left after weights + activation/CUDA-graph overhead (aggregate
    over the TP group). ``weight_bytes`` is the whole model's weights."""
    usable = float(gpu_mem_bytes) * int(tp) * float(gpu_memory_utilization)
    return max(usable - float(weight_bytes) - usable * float(activation_overhead_frac), 0.0)


def kv_capacity_tokens(gpu_mem_bytes: float, weight_bytes: float, *, n_layers: int,
                       n_kv_heads: int, head_dim: int, tp: int = 1,
                       kv_dtype_bytes: float = 2.0,
                       gpu_memory_utilization: float = 0.9) -> float:
    """Total KV tokens the pool can hold concurrently (batch × seqlen budget)."""
    per_tok = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kv_dtype_bytes)
    if per_tok <= 0:
        return 0.0
    return kv_pool_bytes(gpu_mem_bytes, weight_bytes, tp, gpu_memory_utilization) / per_tok


def suggested_max_model_len(kv_capacity_tok: float, target_concurrency: int = 1) -> int:
    """Per-request context that fits if ``target_concurrency`` such requests run."""
    return int(kv_capacity_tok / max(int(target_concurrency), 1))


def length_budget(max_model_len: int, min_len: int = 4) -> tuple[int, int]:
    """Dataset pruning caps derived from the served context window.

    The server enforces ``prompt + output <= max_model_len``, so that is the total
    cap; the prompt cap is the same minus the minimum output room. Returns
    ``(max_prompt_len, max_total_len)``.
    """
    m = int(max_model_len)
    return max(m - int(min_len), 1), m
