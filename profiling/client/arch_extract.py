"""Derive the per-model architecture descriptor from a HuggingFace ``config.json``.

This replaces the hand-curated ``ARCH`` dict in
``feature-test/build_ledger_cache.py`` (lines 30-67) so that a *new* model on
already-characterized hardware is nearly free: point the manifest emitter at the
served model's config and the physical scalars the power model needs fall out.

The function returns exactly the fields the ledger consumes:

    n_active, w_bytes, d_model, n_layers, n_kv, head_dim,
    moe_frac, n_experts, top_k, swa_window, fp8

plus three **additive** fields that default to ``0`` so existing softmax/dense
models are unchanged (they are only populated for the new architectures):

    swa_global_ratio   local:global attention-layer ratio (Gemma-3 style; 5 -> 5:1)
    linear_attention   1 if the model uses linear/lightning attention (MiniMax)
    n_linear_layers    number of linear-attention layers

Design note (equivalence gate): for the existing 6 models the analytic parameter
count reproduces the curated ``n_active`` exactly for dense models (Llama untied
embeddings) and within tolerance for MoE; ``w_bytes`` for quantized checkpoints
(gpt-oss is MXFP4) is a *measured footprint* that is not derivable from the
config, so callers pass ``weight_footprint_bytes`` (the sum of the safetensors
shard sizes) and the extractor uses it verbatim. ``extract_arch`` never fetches
anything — it is a pure function over a config ``dict`` so it is trivially
testable; ``load_config`` is the thin network helper.
"""

from __future__ import annotations

from typing import Optional

GIB = 1024.0**3

# Approximate stored bytes per weight element for non-standard dtypes.
# MXFP4 packs ~4.25 bits/param (4-bit mantissa + shared 8-bit scale per block).
_MXFP4_BYTES_PER_PARAM = 4.25 / 8.0


def _text_config(config: dict) -> dict:
    """Llama-4 (and other multimodal configs) nest the LM under ``text_config``."""
    tc = config.get("text_config")
    if isinstance(tc, dict):
        # Merge: text_config wins, but keep top-level keys it omits (e.g. vocab).
        merged = dict(config)
        merged.update(tc)
        return merged
    return config


def _get(config: dict, *names, default=None):
    for n in names:
        if n in config and config[n] is not None:
            return config[n]
    return default


def _head_dim(config: dict, d_model: int, n_heads: int) -> int:
    hd = _get(config, "head_dim")
    if hd:
        return int(hd)
    return int(d_model // max(1, n_heads))


def _attn_params(d_model: int, n_heads: int, n_kv: int, head_dim: int) -> float:
    """Q/K/V/O projection parameters for one attention block (no biases)."""
    q = d_model * n_heads * head_dim
    k = d_model * n_kv * head_dim
    v = d_model * n_kv * head_dim
    o = n_heads * head_dim * d_model
    return float(q + k + v + o)


def _gated_mlp_params(d_model: int, intermediate: int) -> float:
    """Gate + up + down for a SwiGLU/GeGLU FFN."""
    return float(3 * d_model * intermediate)


def _embedding_params(vocab: int, d_model: int, tied: bool) -> float:
    e = float(vocab * d_model)
    return e if tied else 2.0 * e


def _bytes_per_param(config: dict, dtype_hint: Optional[str]) -> tuple[float, int]:
    """Return (bytes_per_param, fp8_flag).

    fp8_flag matches the curated convention: FP8 -> 1, everything else
    (bf16/fp16, MXFP4) -> 0.
    """
    quant = config.get("quantization_config") or {}
    qmethod = str(quant.get("quant_method", "")).lower()
    hint = (dtype_hint or "").lower()

    if "fp8" in hint or "float8" in hint or "fp8" in qmethod:
        return 1.0, 1
    if "mxfp4" in hint or "fp4" in hint or "mxfp4" in qmethod or qmethod == "fp4":
        return _MXFP4_BYTES_PER_PARAM, 0

    td = str(_get(config, "torch_dtype", "dtype", default="bfloat16")).lower()
    if "fp8" in td or "float8" in td:
        return 1.0, 1
    if "fp4" in td or "mxfp4" in td:
        return _MXFP4_BYTES_PER_PARAM, 0
    return 2.0, 0  # bf16 / fp16


def _moe_fields(config: dict) -> Optional[dict]:
    """Detect MoE and return {n_experts, top_k, moe_inter, shared_inter,
    n_shared, n_moe_layers, n_dense_layers}; None if dense."""
    n_experts = _get(
        config, "num_experts", "num_local_experts", "n_routed_experts",
        "moe_num_experts",
    )
    # NB: do NOT fall back to ``n_group`` (DeepSeek expert-GROUP count, not top-k).
    top_k = _get(config, "num_experts_per_tok", "moe_topk", "num_selected_experts",
                 "top_k_experts", "num_experts_per_token")  # gemma-4: top_k_experts
    if not n_experts or not top_k:
        return None
    n_experts = int(n_experts)
    top_k = int(top_k)

    d_model = int(_get(config, "hidden_size", "d_model"))
    moe_inter = int(_get(config, "moe_intermediate_size", "expert_intermediate_size",
                         "intermediate_size_moe",
                         default=_get(config, "intermediate_size", default=0)))
    # Shared experts (DeepSeek/Qwen-style). Llama-4 uses a single shared expert
    # sized by ``intermediate_size_mlp`` (or the plain ``intermediate_size``).
    n_shared = int(_get(config, "n_shared_experts", "num_shared_experts",
                        "moe_num_shared_experts", default=0))
    shared_inter = int(_get(config, "shared_expert_intermediate_size",
                            "intermediate_size_mlp", default=0))
    if shared_inter and n_shared == 0:
        n_shared = 1

    n_layers = int(_get(config, "num_hidden_layers", "n_layers"))
    # How many layers are actually MoE vs dense.
    n_dense = int(_get(config, "first_k_dense_replace", default=0))
    sparse_step = int(_get(config, "decoder_sparse_step", default=1)) or 1
    mlp_only = _get(config, "mlp_only_layers", default=[]) or []
    if mlp_only or sparse_step > 1:
        moe_layer_idx = [
            i for i in range(n_layers)
            if (i not in mlp_only) and (i >= n_dense) and (i % sparse_step == 0)
        ]
        n_moe_layers = len(moe_layer_idx)
    else:
        n_moe_layers = max(0, n_layers - n_dense)
    n_dense_layers = n_layers - n_moe_layers

    return dict(
        n_experts=n_experts, top_k=top_k, moe_inter=moe_inter,
        shared_inter=shared_inter, n_shared=n_shared,
        n_moe_layers=n_moe_layers, n_dense_layers=n_dense_layers,
    )


def _linear_attention_layers(config: dict, n_layers: int) -> int:
    """Count linear/lightning-attention layers (MiniMax hybrid attention).

    MiniMax configs expose the hybrid schedule either as an explicit
    ``layer_types`` / ``attn_type_list`` array or as a periodic
    ``linear_attn_period`` / ``attention_type`` pattern. Returns 0 for ordinary
    softmax models so the flag stays additive.
    """
    layer_types = _get(config, "layer_types", "attn_type_list",
                       "full_attn_idxs")  # presence/strings vary by release
    if isinstance(layer_types, list) and layer_types:
        cnt = 0
        for t in layer_types:
            s = str(t).lower()
            if "linear" in s or "lightning" in s or s in ("0", "false"):
                cnt += 1
        return cnt
    # Periodic schedule: every Nth layer is full softmax, rest linear.
    period = _get(config, "linear_attn_period", "full_attn_period",
                  "attn_type_period")
    if period:
        period = int(period)
        n_full = len([i for i in range(n_layers) if i % period == period - 1])
        return n_layers - n_full
    if str(_get(config, "attention_type", default="")).lower() in (
        "linear", "lightning", "hybrid"
    ):
        # Hybrid without an explicit schedule: assume the documented MiniMax
        # 7:1 linear:softmax interleave.
        n_full = max(1, n_layers // 8)
        return n_layers - n_full
    return 0


def extract_arch(
    config: dict,
    *,
    dtype_hint: Optional[str] = None,
    weight_footprint_bytes: Optional[float] = None,
    n_active_override: Optional[float] = None,
    family: Optional[str] = None,
) -> dict:
    """Map a HF ``config.json`` dict to the power-model arch descriptor.

    Parameters
    ----------
    config : the parsed ``config.json`` (top-level; nested ``text_config`` is
        handled automatically).
    dtype_hint : optional served-dtype string (e.g. ``"FP8"`` from a model id
        like ``Llama-3.1-405B-Instruct-FP8``) that overrides the config dtype.
    weight_footprint_bytes : measured resident weight footprint (sum of
        safetensors shard sizes). Required for quantized checkpoints whose
        footprint is not derivable from the config (e.g. gpt-oss MXFP4).
    n_active_override : vendor-stated active parameter count. The analytic MoE
        estimate is only good to ~10-30%, and n_active scales the FLOPs work
        rate directly, so for MoE models prefer the published "A<N>B" count
        (e.g. 22e9 for Qwen3-235B-A22B, 5.1e9 for gpt-oss-120b). Dense counts are
        exact and need no override.
    family : optional family label for fit grouping; if omitted a label is
        synthesized from size/sparsity.

    Note: ``n_active`` includes embedding parameters, matching the curated
    convention in ``build_ledger_cache.ARCH`` (the existing fit absorbs the
    embedding-as-FLOPs into e_f); changing that would break ledger equivalence.
    """
    cfg = _text_config(config)

    d_model = int(_get(cfg, "hidden_size", "d_model"))
    n_layers = int(_get(cfg, "num_hidden_layers", "n_layers"))
    n_heads = int(_get(cfg, "num_attention_heads", "n_heads"))
    n_kv = int(_get(cfg, "num_key_value_heads", "num_kv_heads", default=n_heads))
    head_dim = _head_dim(cfg, d_model, n_heads)
    vocab = int(_get(cfg, "vocab_size", default=0))
    tied = bool(_get(cfg, "tie_word_embeddings", default=False))
    inter = int(_get(cfg, "intermediate_size", "ffn_dim", default=0))

    bpp, fp8 = _bytes_per_param(cfg, dtype_hint)

    embed = _embedding_params(vocab, d_model, tied)
    attn = _attn_params(d_model, n_heads, n_kv, head_dim)
    moe = _moe_fields(cfg)

    if moe is None:
        # Dense.
        mlp = _gated_mlp_params(d_model, inter)
        n_active = embed + n_layers * (attn + mlp)
        n_total = n_active
        moe_frac = 0.0
        n_experts, top_k = 1, 1
    else:
        router = float(d_model * moe["n_experts"])
        expert_ffn = _gated_mlp_params(d_model, moe["moe_inter"])
        shared_ffn = (
            _gated_mlp_params(d_model, moe["shared_inter"]) * moe["n_shared"]
            if moe["shared_inter"] else 0.0
        )
        dense_ffn = _gated_mlp_params(d_model, inter) if inter else expert_ffn

        # Active params per token: top_k routed experts + shared + dense layers.
        moe_active_ffn = moe["top_k"] * expert_ffn + shared_ffn
        n_active = (
            embed
            + moe["n_moe_layers"] * (attn + router + moe_active_ffn)
            + moe["n_dense_layers"] * (attn + dense_ffn)
        )
        # Total resident params: ALL experts.
        moe_total_ffn = moe["n_experts"] * expert_ffn + shared_ffn
        n_total = (
            embed
            + moe["n_moe_layers"] * (attn + router + moe_total_ffn)
            + moe["n_dense_layers"] * (attn + dense_ffn)
        )
        # Fraction of decode weight traffic that is routed-expert (gates the
        # batch-dependent expert-read term in build_run_bins).
        routed = moe["n_moe_layers"] * moe["n_experts"] * expert_ffn
        moe_frac = float(routed / max(n_total - embed, 1.0))
        n_experts, top_k = moe["n_experts"], moe["top_k"]

    if n_active_override is not None:
        n_active = float(n_active_override)

    if weight_footprint_bytes is not None:
        w_bytes = float(weight_footprint_bytes)
    else:
        w_bytes = bpp * n_total

    swa_window = float(_get(cfg, "sliding_window", default=0) or 0)
    # Gemma-3 style interleave: ``sliding_window_pattern: 6`` -> 5 local : 1 global.
    pattern = _get(cfg, "sliding_window_pattern", "global_attn_every_n_layers")
    if pattern:
        swa_global_ratio = float(int(pattern) - 1)
    else:
        # Gemma-4 encodes the split as an explicit per-layer list instead of a
        # period (e.g. 40 ``sliding_attention`` : 8 ``full_attention`` = 5:1).
        lt = _get(cfg, "layer_types", default=None)
        n_full = sum(1 for t in lt if "full" in str(t).lower()) if isinstance(lt, list) else 0
        n_local = sum(1 for t in lt if "slid" in str(t).lower()) if isinstance(lt, list) else 0
        swa_global_ratio = float(n_local / n_full) if n_full else 0.0

    n_linear = _linear_attention_layers(cfg, n_layers)
    linear_attention = 1 if n_linear > 0 else 0

    return dict(
        family=family or _synth_family(n_active, moe is not None),
        n_active=float(n_active),
        w_bytes=float(w_bytes),
        d_model=int(d_model),
        n_layers=int(n_layers),
        n_kv=int(n_kv),
        head_dim=int(head_dim),
        moe_frac=float(moe_frac),
        n_experts=int(n_experts),
        top_k=int(top_k),
        swa_window=float(swa_window),
        fp8=int(fp8),
        # additive (default 0 for existing softmax/dense models)
        swa_global_ratio=float(swa_global_ratio),
        linear_attention=int(linear_attention),
        n_linear_layers=int(n_linear),
    )


def _synth_family(n_active: float, is_moe: bool) -> str:
    b = n_active / 1e9
    kind = "moe" if is_moe else "dense"
    if b < 12:
        size = "8b"
    elif b < 24:
        size = "14b"
    elif b < 48:
        size = "32b"
    elif b < 120:
        size = "70b"
    elif b < 300:
        size = "235b"
    else:
        size = "405b"
    return f"{kind}-{size}"


def load_config(model_id: str) -> dict:
    """Fetch a model's ``config.json`` as a dict (not unit-tested).

    Prefers ``transformers.AutoConfig``, but falls back to the raw ``config.json``
    when AutoConfig can't build the architecture — e.g. very new model_types like
    ``gemma4_unified`` that vLLM serves but the installed transformers doesn't yet
    register. ``extract_arch`` consumes the dict either way (it normalises nested
    ``text_config`` and reads fields by name), so the raw config is equivalent for
    the fields the power model needs.
    """
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_id, trust_remote_code=True).to_dict()
    except Exception:
        import json
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(model_id, "config.json")) as f:
            return json.load(f)
