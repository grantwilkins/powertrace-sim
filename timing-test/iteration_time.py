"""Architecture-derived work per engine iteration and roofline iteration time.

Work is computed from the architecture registry's descriptors only
(`model/training_data/arch.py`); no model-name constants. Time applies the
roofline rule with per-hardware efficiencies and a per-(hardware, GPU-count)
launch/synchronization overhead, per timing-test/DESIGN.md section 1.

Conventions:
- `decode_batch` sequences each generate one token in the iteration.
- `context_mean` is the mean tokens of visible context per decode sequence.
- A prefill chunk of `prefill_chunk` tokens extends a prompt whose already
  processed prefix is `prefill_context` tokens (0 for the first chunk).
- Sliding-window attention (gpt-oss) alternates full and windowed layers,
  so effective context is 0.5*c + 0.5*min(c, window), matching the ledger.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from model.training_data.moe_routing import (
    RoutingLaw,
    expected_iteration_weight_bytes,
)

HARDWARE_PROFILES = {
    # Datasheet scales, identical to the power model's profiles.
    "A100": {"peak_flops_s": 312e12, "hbm_bytes_s": 2.0e12, "hbm_capacity": 80e9},
    "H100": {"peak_flops_s": 990e12, "hbm_bytes_s": 3.35e12, "hbm_capacity": 80e9},
}


def _effective_context(context: float, arch: Mapping[str, object]) -> float:
    window = float(arch.get("swa_window", 0) or 0)
    if window <= 0:
        return max(context, 0.0)
    context = max(context, 0.0)
    return 0.5 * context + 0.5 * min(context, window)


def kv_bytes_per_token(arch: Mapping[str, object]) -> float:
    """KV-cache bytes written per generated/prefilled token (BF16 KV)."""
    return 2.0 * float(arch["n_layers"]) * float(arch["n_kv"]) * float(arch["head_dim"]) * 2.0


def expected_weight_bytes_per_sweep(arch: Mapping[str, object], tokens: float) -> float:
    """Expected resident weight bytes read by one iteration handling `tokens`.

    Dense: the full weights. Mixture-of-experts: dense/shared fraction plus
    the expected unique routed experts touched by `tokens` independent
    uniform top-k draws (the same expectation as the power ledger).
    """
    return expected_iteration_weight_bytes(arch, decode_tokens=tokens)


def _attention_flops(arch: Mapping[str, object], tokens: float, context: float) -> float:
    n_q = float(arch["d_model"]) / float(arch["head_dim"])
    return 4.0 * n_q * float(arch["head_dim"]) * float(arch["n_layers"]) * tokens * \
        _effective_context(context, arch)


def _dtype_scale(arch: Mapping[str, object]) -> float:
    """FLOP demand scale for the double-rate dtype fraction (as in physics)."""
    frac = arch.get("fp8_flop_frac")
    if frac is None:
        return 0.5 if float(arch.get("fp8", 0) or 0) > 0 else 1.0
    return 1.0 - 0.5 * min(max(float(frac), 0.0), 1.0)


def transformer_bw_scale(
    arch: Mapping[str, object], params: Mapping[str, object], hardware: str
) -> float:
    """FP8 transformer-stream support; BF16 components retain base bandwidth."""
    if float(arch.get("fp8", 0) or 0) <= 0:
        return 1.0
    if "fp8_stream_scale" not in params:
        raise ValueError(
            f"{hardware} FP8 timing requires fp8_stream_scale calibration"
        )
    return float(params["fp8_stream_scale"])


def _component_weight_bytes(
    arch: Mapping[str, object], *, tokens: float, logit_tokens: float
) -> float | None:
    required = (
        "transformer_weight_bytes",
        "input_embedding_weight_bytes",
        "input_embedding_params",
        "output_head_weight_bytes",
    )
    if any(key not in arch for key in required):
        return None
    if tokens <= 0:
        return 0.0
    row_bytes = (
        float(arch["input_embedding_weight_bytes"])
        / max(float(arch["input_embedding_params"]), 1.0)
        * float(arch["d_model"])
    )
    return (
        float(arch["transformer_weight_bytes"])
        + (float(arch["output_head_weight_bytes"]) if logit_tokens > 0 else 0.0)
        + row_bytes * tokens
    )


def iteration_work(
    arch: Mapping[str, object], *, decode_batch: float = 0.0,
    context_mean: float = 0.0, prefill_chunk: float = 0.0,
    prefill_context: float = 0.0, prefill_groups: float = 1.0,
    prefill_chunks: Sequence[tuple[float, float]] | None = None,
    prefill_logits: float = 0.0,
    routing_law: RoutingLaw | None = None,
) -> dict:
    """FLOPs and HBM bytes for one engine iteration (decode + optional chunk)."""
    if decode_batch < 0 or prefill_chunk < 0 or prefill_logits < 0:
        raise ValueError("decode_batch, prefill_chunk, and prefill_logits must be non-negative")
    if prefill_chunks is not None and (prefill_chunk or prefill_context):
        raise ValueError("Use scalar prefill fields or prefill_chunks, not both")
    chunks = (
        [(float(tokens), float(context)) for tokens, context in prefill_chunks]
        if prefill_chunks is not None
        else ([(float(prefill_chunk), float(prefill_context))]
              if prefill_chunk > 0 else [])
    )
    if any(tokens <= 0 or context < 0 for tokens, context in chunks):
        raise ValueError("Prefill chunks need positive tokens and non-negative context")
    prefill_tokens = sum(tokens for tokens, _ in chunks)
    prefill_groups = float(len(chunks)) if prefill_chunks is not None else prefill_groups
    kv_tok = kv_bytes_per_token(arch)
    scale = _dtype_scale(arch)
    linear_tokens = decode_batch + prefill_tokens
    logit_tokens = decode_batch + prefill_logits

    transformer_flops = 0.0
    output_head_flops = 0.0
    attn_flops = 0.0
    attn_bytes = 0.0
    if "transformer_active_params" in arch and "output_head_params" in arch:
        transformer_flops = (
            2.0 * scale * float(arch["transformer_active_params"]) * linear_tokens
        )
        output_head_flops = (
            2.0 * float(arch["output_head_params"]) * logit_tokens
        )
    else:
        transformer_flops = scale * 2.0 * float(arch["n_active"]) * linear_tokens
    if decode_batch > 0:
        attn_flops += _attention_flops(arch, decode_batch, context_mean)
        attn_bytes += decode_batch * (
            _effective_context(context_mean, arch) + 1.0) * kv_tok
    for chunk_tokens, prior_context in chunks:
        # Causal chunk: token j attends to prefill_context + j prior tokens.
        # Attention KV traffic is NOT charged per query token: fused
        # attention kernels tile keys/values through on-chip memory, so a
        # chunk's queries share KV reads and prefill is compute-bound
        # (Splitwise ISCA'24; the 65k-token prefill probe confirms the
        # per-query accounting over-predicts by ~8x). One streaming pass of
        # the visible context plus the chunk's KV write is charged.
        mean_visible = prior_context + chunk_tokens / 2.0
        attn_flops += _attention_flops(arch, chunk_tokens, mean_visible)
        attn_bytes += (_effective_context(mean_visible, arch)
                       + chunk_tokens) * kv_tok
    component_bytes = (
        _component_weight_bytes(arch, tokens=linear_tokens, logit_tokens=logit_tokens)
        if float(arch.get("moe_frac", 0.0)) == 0.0 else None
    )
    gemm_bytes = component_bytes
    if gemm_bytes is None:
        gemm_bytes = expected_iteration_weight_bytes(
            arch, decode_tokens=decode_batch, prefill_tokens=prefill_tokens,
            prefill_groups=prefill_groups, routing_law=routing_law)
    detail = {}
    if component_bytes is not None:
        row_bytes = (
            float(arch["input_embedding_weight_bytes"])
            / max(float(arch["input_embedding_params"]), 1.0)
            * float(arch["d_model"])
        )
        detail = {
            "transformer_flops": transformer_flops,
            "transformer_bytes": (
                float(arch["transformer_weight_bytes"]) if linear_tokens > 0 else 0.0
            ),
            "output_head_flops": output_head_flops,
            "output_head_bytes": (
                float(arch["output_head_weight_bytes"]) if logit_tokens > 0 else 0.0
            ),
            "embedding_bytes": row_bytes * linear_tokens,
        }
    # Each decoding sequence samples one token (softmax, sampling, and
    # scheduler bookkeeping are per generated token, largely CPU-side).
    return {"gemm_flops": transformer_flops + output_head_flops,
            "gemm_bytes": gemm_bytes,
            "attn_flops": attn_flops, "attn_bytes": attn_bytes,
            "sampled_tokens": decode_batch, "logit_tokens": logit_tokens,
            **detail}


def launch_overhead_s(arch: Mapping[str, object], *, base_s: float,
                      per_message_s: float) -> float:
    """Per-iteration launch/synchronization overhead.

    Each transformer layer issues its kernels and, under tensor
    parallelism, two collectives; the per-iteration overhead therefore
    scales with layer count: base + 2 * n_layers * per_message latency
    (the latency term of the standard alpha-beta communication model).
    `per_message_s` is fitted per (hardware, GPU count); at one GPU it
    covers kernel-launch chain latency only.
    """
    if base_s < 0 or per_message_s < 0:
        raise ValueError("overhead terms must be non-negative")
    return base_s + 2.0 * float(arch["n_layers"]) * per_message_s


def iteration_time_s(
    work: Mapping[str, float], *, hardware: str, tp: int,
    eff_flops: float, eff_bw: float, t_launch_s: float,
    t_sample_s: float = 0.0,
    transformer_bw_scale: float = 1.0,
) -> float:
    """Per-operator roofline durations, summed (GenZ/LLMCompass convention).

    The linear (GEMM) operators and the attention operators run
    sequentially within a layer, so each class is individually compute- or
    bandwidth-bound and their times ADD; a single whole-iteration max
    under-predicts whenever both are large (high batch). Work is divided
    across `tp` GPUs; the launch/sync overhead is per iteration; the
    sampling term is per generated token (serial engine work, not divided
    by GPU count).
    """
    if tp < 1:
        raise ValueError("tp must be >= 1")
    if not (0.0 < eff_flops <= 1.0 and 0.0 < eff_bw <= 1.0):
        raise ValueError("efficiencies must be in (0, 1]")
    if t_sample_s < 0:
        raise ValueError("t_sample_s must be non-negative")
    if transformer_bw_scale <= 0.0:
        raise ValueError("transformer_bw_scale must be positive")
    profile = HARDWARE_PROFILES[hardware]
    compute_rate = eff_flops * profile["peak_flops_s"] * tp
    memory_rate = eff_bw * profile["hbm_bytes_s"] * tp
    component_keys = {
        "transformer_flops", "transformer_bytes", "output_head_flops",
        "output_head_bytes", "embedding_bytes",
    }
    if component_keys <= set(work):
        transformer_s = max(
            float(work["transformer_flops"]) / compute_rate,
            float(work["transformer_bytes"]) / (
                memory_rate * transformer_bw_scale
            ),
        )
        output_head_s = max(
            float(work["output_head_flops"]) / compute_rate,
            float(work["output_head_bytes"]) / memory_rate,
        )
        gemm_s = (
            transformer_s + output_head_s
            + float(work["embedding_bytes"]) / memory_rate
        )
    else:
        gemm_s = max(float(work["gemm_flops"]) / compute_rate,
                     float(work["gemm_bytes"]) / memory_rate)
    attn_s = max(float(work["attn_flops"]) / compute_rate,
                 float(work["attn_bytes"]) / memory_rate)
    return (t_launch_s + gemm_s + attn_s
            + t_sample_s * float(work.get("sampled_tokens", 0.0)))
