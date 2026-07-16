"""Analytic physics layer: architecture descriptor -> FLOPs / bytes / efficiency.

Everything here is computed from the architecture descriptor (the ``arch`` block
of a run ``manifest.json``) plus the hardware datasheet. No coefficients are fit
here; these are the *work rates* the power model's coefficients multiply.

The single source of truth for the physics mapping is :func:`work_rates`, shared
by both the calibration ingest path (``ingest.py``) and the predict-time
inference chain (``workload.py``) so the two can never diverge.

Key physics (all SWA-aware):

* **Prefill compute** = linear projections ``2 * n_active * tokens`` PLUS attention
  that respects sliding-window geometry: global layers cost O(L^2), windowed
  layers cost O(L * window). Attention is returned separately so the model can
  charge it its own energy coefficient (different arithmetic intensity than GEMM).
* **Decode compute** = projection GEMV ``2 * n_active * tokens`` (decode attention
  is memory-bound and accounted via KV-read bytes, not FLOPs).
* **Weight bytes**: dense reads ``w_bytes`` per iteration; MoE reads only the
  touched experts ``w_bytes * (dense_frac + moe_frac * min(1, B*top_k/n_experts))``.
* **KV bytes/token** = ``2 * n_layers * n_kv * head_dim * dtype`` with windowed
  layers capping context at ``swa_window``.
* **NVLink** all-reduce bytes per token for TP>1.
* **Efficiency** ``eta(AI) = min(1, AI / AI_ridge)`` from the roofline ridge point,
  applied to the compute (FLOPs) terms to model MFU (replaces the old fitted
  per-family multiplier).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# --------------------------------------------------------------------------- #
# Hardware datasheets (per GPU). bf16 dense peak FLOP/s, HBM B/s, NVLink B/s.
# A100/H100 SXM. NVLink: aggregate bidirectional bytes/s (A100 600 GB/s gen3,
# H100 900 GB/s gen4).
# --------------------------------------------------------------------------- #
HW = {
    "A100": dict(peak_flops=312e12, hbm_bw=2.0e12, nvlink_bw=600e9,
                 hbm_bytes=80e9, gpus_per_node=8),
    "H100": dict(peak_flops=990e12, hbm_bw=3.35e12, nvlink_bw=900e9,
                 hbm_bytes=80e9, gpus_per_node=8),
}

KV_ELEM_BYTES = 2.0  # KV cache element size (fp16/bf16)
DTYPE_BYTES = {0: 2.0, 1: 1.0}  # fp8 flag -> bytes/elem for weights (bf16 vs fp8)


def normalize_arch(a: Dict) -> Dict:
    """Coerce a manifest ``arch`` block into floats + derived SWA layer split.

    ``swa_global_ratio`` is the gemma-style local:global cadence (e.g. 5 => one
    global attention layer per 6, the rest sliding-window). With no window the
    model is treated as fully global (standard causal attention).
    """
    a = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in a.items()}
    a.setdefault("n_linear_layers", 0.0)
    a.setdefault("fp8", 0.0)
    a.setdefault("moe_frac", 0.0)
    a.setdefault("top_k", 1.0)
    a.setdefault("n_experts", 1.0)
    window = float(a.get("swa_window", 0.0) or 0.0)
    ratio = float(a.get("swa_global_ratio", 0.0) or 0.0)
    if window <= 0.0 or ratio <= 0.0:
        a["_global_frac"] = 1.0   # fully global attention
        a["_swa_window"] = 0.0
    else:
        a["_global_frac"] = 1.0 / (ratio + 1.0)
        a["_swa_window"] = window
    return a


def weight_dtype_bytes(arch: Dict) -> float:
    return DTYPE_BYTES[int(round(float(arch.get("fp8", 0.0))))]


# --------------------------------------------------------------------------- #
# Per-quantity physics (scalar or numpy-array inputs)
# --------------------------------------------------------------------------- #

def attn_flops_per_seq(arch: Dict, L):
    """Attention FLOPs to prefill one sequence of length ``L`` (SWA-aware).

    Causal attention over a length-L sequence: global layers cost ~2*L^2*d_model,
    windowed layers cost ~2*L*min(L,window)*d_model (each query attends at most
    ``window`` keys). Constant factors are absorbed by the fitted ``e_f_attn``;
    what matters is the L^2-vs-L*window shape the prefill staircase identifies.
    """
    L = np.asarray(L, dtype=np.float64)
    d_model = float(arch["d_model"])
    n_layers = float(arch["n_layers"])
    gfrac = float(arch["_global_frac"])
    window = float(arch["_swa_window"])
    n_global = n_layers * gfrac
    n_window = n_layers - n_global
    f_global = 2.0 * L * L * d_model
    if window > 0.0:
        f_window = 2.0 * L * np.minimum(L, window) * d_model
    else:
        f_window = f_global
    return n_global * f_global + n_window * f_window


def kv_bytes_per_token(arch: Dict, ctx):
    """Bytes read from the KV cache for one query token attending ``ctx`` context.

    Global layers read the full context; windowed layers read at most
    ``swa_window``. Per layer KV is ``2 * n_kv * head_dim * dtype`` (K and V).
    """
    ctx = np.asarray(ctx, dtype=np.float64)
    per_layer_tok = 2.0 * float(arch["n_kv"]) * float(arch["head_dim"]) * KV_ELEM_BYTES
    n_layers = float(arch["n_layers"])
    gfrac = float(arch["_global_frac"])
    window = float(arch["_swa_window"])
    n_global = n_layers * gfrac
    n_window = n_layers - n_global
    ctx_window = ctx if window <= 0.0 else np.minimum(ctx, window)
    return per_layer_tok * (n_global * ctx + n_window * ctx_window)


def kv_bytes_written_per_token(arch: Dict) -> float:
    """Bytes written to the KV cache per generated/processed token (all layers)."""
    return 2.0 * float(arch["n_layers"]) * float(arch["n_kv"]) * float(arch["head_dim"]) * KV_ELEM_BYTES


def decode_weight_bytes(arch: Dict, batch):
    """Weight bytes read per decode iteration (amortized across the batch).

    Dense: the full ``w_bytes``. MoE: dense part always read, expert part scaled
    by the fraction of experts touched, ``min(1, B*top_k/n_experts)``.
    """
    batch = np.asarray(batch, dtype=np.float64)
    w = float(arch["w_bytes"])
    moe = float(arch["moe_frac"])
    if moe <= 0.0:
        return np.full(batch.shape, w) if batch.shape else w
    touched = np.minimum(1.0, batch * float(arch["top_k"]) / max(float(arch["n_experts"]), 1.0))
    return w * ((1.0 - moe) + moe * touched)


def nvlink_bytes_per_token(arch: Dict, tp: float) -> float:
    """All-reduce bytes per token across the TP group (0 when tp==1).

    Two all-reduces per layer (attention + MLP output), each moving the hidden
    activation; ring all-reduce volume scales as 2*(tp-1)/tp * message bytes.
    """
    tp = float(tp)
    if tp <= 1.0:
        return 0.0
    msg = 2.0 * float(arch["d_model"]) * KV_ELEM_BYTES  # activation bytes/token/layer
    return float(arch["n_layers"]) * 2.0 * msg * 2.0 * (tp - 1.0) / tp


def eta_compute(ai, hw: str):
    """Roofline MFU factor in (0, 1]: min(1, AI / AI_ridge).

    AI_ridge = peak_flops / hbm_bw (FLOP per byte at the roofline knee). Compute
    energy is scaled by this so memory-bound phases (low AI, e.g. MoE decode)
    contribute proportionally less compute power than compute-bound prefill.
    """
    ai = np.asarray(ai, dtype=np.float64)
    ridge = HW[hw]["peak_flops"] / HW[hw]["hbm_bw"]
    return np.clip(ai / ridge, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Assembler: per-second work rates + efficiency (single source of truth)
# --------------------------------------------------------------------------- #

def work_rates(arch: Dict, hw: str, tp: float, *, pre_tok, dec_tok, decode_batch,
               L_pre, ctx_dec, iters_pre=None, iters_dec=None) -> Dict[str, np.ndarray]:
    """Map per-bin work to physics work-rates (per second) + roofline efficiency.

    Inputs are arrays aligned to 1 s bins (scalars broadcast):
      pre_tok      prefill tokens/s
      dec_tok      decode tokens/s
      decode_batch effective concurrent decode sequences (the running batch)
      L_pre        context length of the prefill work in the bin (for attention)
      ctx_dec      mean context length of decoding sequences (for KV read)
      iters_pre    prefill iterations/s   (default: pre_tok / max(L_pre,1))
      iters_dec    decode iterations/s    (default: dec_tok / max(decode_batch,1))

    Returns work-rate features used by the power model, plus the two efficiency
    factors ``eta_pre`` / ``eta_dec`` that scale the compute terms.
    """
    arch = arch if "_global_frac" in arch else normalize_arch(arch)
    pre_tok = np.asarray(pre_tok, dtype=np.float64)
    dec_tok = np.asarray(dec_tok, dtype=np.float64)
    decode_batch = np.asarray(decode_batch, dtype=np.float64)
    L_pre = np.asarray(L_pre, dtype=np.float64)
    ctx_dec = np.asarray(ctx_dec, dtype=np.float64)
    n_active = float(arch["n_active"])

    if iters_pre is None:
        iters_pre = pre_tok / np.maximum(L_pre, 1.0)
    if iters_dec is None:
        iters_dec = dec_tok / np.maximum(decode_batch, 1e-9)
    iters_pre = np.asarray(iters_pre, dtype=np.float64)
    iters_dec = np.asarray(iters_dec, dtype=np.float64)

    # ---- compute FLOPs/s
    flops_proj_pre = 2.0 * n_active * pre_tok
    # attention FLOPs/s = (sequences/s) * attn_flops_per_seq(L); sequences/s ~ pre_tok / L
    seqs_pre = pre_tok / np.maximum(L_pre, 1.0)
    flops_attn_pre = seqs_pre * attn_flops_per_seq(arch, np.maximum(L_pre, 1.0))
    flops_dec = 2.0 * n_active * dec_tok

    # ---- memory bytes/s. Weights are read ONCE per iteration serving the whole
    # running batch (prefill chunk + decode batch share the iteration), so total
    # weight traffic = w_eff(batch) * iters_total -- NOT a per-phase sum (that
    # double-counts in the interleaved regime).
    iters_tot = iters_pre + iters_dec
    w_read = decode_weight_bytes(arch, decode_batch) * iters_tot
    kv_read = dec_tok * kv_bytes_per_token(arch, ctx_dec)
    kv_write = (pre_tok + dec_tok) * kv_bytes_written_per_token(arch)
    nvlink = (pre_tok + dec_tok) * nvlink_bytes_per_token(arch, tp)

    # ---- physical-bandwidth clamp: no GPU reads HBM faster than its bandwidth.
    hbm_cap = tp * HW[hw]["hbm_bw"]
    hbm_total = w_read + kv_read
    scale = np.where(hbm_total > hbm_cap, hbm_cap / np.maximum(hbm_total, 1.0), 1.0)
    w_read = w_read * scale
    kv_read = kv_read * scale

    # ---- roofline efficiency from the ACTUAL per-iteration operating point.
    # eta is computed ONCE from the combined arithmetic intensity of the iteration
    # (all compute that ran + the bytes that iteration moved), then applied to
    # every compute term. This is the key fix: interleaved/chunked prefill rides a
    # memory-bound decode iteration, so its eta is LOW (not 1 as a per-phase
    # prefill AI would give) -- which stops the model from slamming full compute
    # power onto interleaved prefill bursts. Pure compute-bound prefill (large
    # chunk, no decode) still yields eta~1, so e_flop stays identified.
    pf = HW[hw]["peak_flops"]
    bw = HW[hw]["hbm_bw"]
    flops_tot = flops_proj_pre + flops_attn_pre + flops_dec
    bytes_tot = w_read + kv_read + kv_write
    ai = np.divide(flops_tot, np.maximum(bytes_tot, 1.0))
    eta = eta_compute(ai, hw)

    return dict(
        flops_proj_pre=flops_proj_pre, flops_attn_pre=flops_attn_pre,
        flops_dec=flops_dec,
        w_read=w_read, kv_read=kv_read, kv_write=kv_write, nvlink=nvlink,
        eta=eta, ai=ai,
        iters_pre=iters_pre, iters_dec=iters_dec,
        ridge_flops_per_byte=np.full(pre_tok.shape, pf / bw) if pre_tok.shape else pf / bw,
    )
