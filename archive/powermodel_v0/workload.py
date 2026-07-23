"""Predict-time inference chain: provider-known inputs -> per-bin work state.

This is the interface layer. Everything here is computed from variables a cloud
provider knows or can infer; NO live engine telemetry is required. The chain:

    offered load + token pairs + server config
        -> requested concurrency
        -> saturation sub-model (KV-cache budget / max_num_seqs)  [swappable]
        -> sustained decode batch
        -> analytic decode step time  t = max(W_eff/(TP*BW), FLOPs/(TP*peak))
        -> iters/s and decode tok/s
        -> per-bin STATE dict (same schema as ingest)

The STATE dict is then fed to ``model.predict_state`` to get power (+ interval).
Occupancy is phase-split into ``decode_batch`` and ``pre_active``.

The saturation sub-model is deliberately separate from the power physics so it
can be swapped for a scheduler-specific model without touching the energy terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from powermodel import arch as A


@dataclass
class ServerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    kv_fraction: float = 0.9          # fraction of free HBM usable for KV cache


@dataclass
class Workload:
    """A steady-state serving point a provider would specify."""
    input_len: float
    output_len: float
    request_rate: float = 0.0         # requests/s offered (0 => use concurrency)
    concurrency: float = 0.0          # requested max concurrent (0 => use rate)


def kv_capacity_batch(arch: Dict, hw: str, tp: float, ctx: float,
                      kv_fraction: float = 0.9) -> float:
    """Max concurrent sequences that fit in the KV cache at context ``ctx``.

    Free HBM = total HBM across the TP group minus the (sharded) model weights.
    KV bytes/seq = kv_bytes_per_token(ctx) * ctx-equivalent (SWA-aware via the
    per-token helper, which already accounts for windowed layers).
    """
    arch = A.normalize_arch(arch)
    total_hbm = tp * A.HW[hw]["hbm_bytes"]
    free = max(total_hbm - float(arch["w_bytes"]), 0.0) * kv_fraction
    kv_per_seq = float(A.kv_bytes_per_token(arch, ctx))  # bytes for one seq's KV
    if kv_per_seq <= 0:
        return float("inf")
    return free / kv_per_seq


def decode_step_time(arch: Dict, hw: str, tp: float, batch: float) -> float:
    """Analytic decode iteration time (s): roofline max of memory and compute."""
    arch = A.normalize_arch(arch)
    w_eff = float(A.decode_weight_bytes(arch, np.asarray(float(batch)))) if batch else float(arch["w_bytes"])
    flops_step = 2.0 * float(arch["n_active"]) * batch
    t_mem = w_eff / (tp * A.HW[hw]["hbm_bw"])
    t_cmp = flops_step / (tp * A.HW[hw]["peak_flops"])
    return max(t_mem, t_cmp, 1e-6)


def sustained_decode_batch(arch, hw, tp, server: ServerConfig, workload: Workload):
    """Resolve requested load -> sustained decode batch via the saturation model.

    If concurrency is given, sustained batch = min(concurrency, max_num_seqs,
    KV-capacity). If a request rate is given, use Little's law B = lambda * T_dec
    at a fixed point with the analytic decode step time (T_dec scales with batch
    through bandwidth sharing).
    """
    ctx = workload.input_len + workload.output_len / 2.0
    cap = min(server.max_num_seqs,
              kv_capacity_batch(arch, hw, tp, ctx, server.kv_fraction))
    if workload.concurrency > 0:
        return float(min(workload.concurrency, cap))
    # rate-driven: fixed point on B = lambda * output_len * t_dec(B)
    lam = workload.request_rate
    B = 1.0
    for _ in range(50):
        t = decode_step_time(arch, hw, tp, B)
        T_dec = workload.output_len * t
        B_new = min(lam * T_dec, cap)
        if abs(B_new - B) < 1e-3:
            B = B_new
            break
        B = 0.5 * (B + B_new)
    return float(max(B, 0.0))


def _tile(value, n):
    return np.full(int(n), float(value))


def decode_operating_point(arch, hw, tp, batch, ctx, n_bins=120) -> Dict:
    """STATE for a steady pure-decode point at given batch and context."""
    arch = A.normalize_arch(arch)
    t = decode_step_time(arch, hw, tp, batch)
    iters = 1.0 / t
    dec_tok = batch * iters
    return dict(
        pre_tok=_tile(0.0, n_bins), dec_tok=_tile(dec_tok, n_bins),
        decode_batch=_tile(batch, n_bins), pre_active=_tile(0.0, n_bins),
        L_pre=_tile(0.0, n_bins), ctx_dec=_tile(ctx, n_bins),
        iters_pre=_tile(0.0, n_bins), iters_dec=_tile(iters, n_bins),
        busy=_tile(1.0 if batch > 0 else 0.0, n_bins),
    )


def prefill_operating_point(arch, hw, tp, input_len, concurrency=1.0,
                            n_bins=120) -> Dict:
    """STATE for a steady pure-prefill point at given input length.

    Prefill throughput is compute/bandwidth limited: time to prefill one sequence
    ~ max(proj+attn FLOPs / (TP*peak), weight bytes / (TP*BW)). tok/s = L / t_seq.
    """
    arch = A.normalize_arch(arch)
    L = float(input_len)
    flops_seq = 2.0 * float(arch["n_active"]) * L + float(A.attn_flops_per_seq(arch, L))
    t_cmp = flops_seq / (tp * A.HW[hw]["peak_flops"])
    t_mem = float(arch["w_bytes"]) / (tp * A.HW[hw]["hbm_bw"])
    t_seq = max(t_cmp, t_mem, 1e-6) / max(concurrency, 1.0)
    pre_tok = L / t_seq
    iters = 1.0 / t_seq  # one (chunked) prefill pass per sequence time
    return dict(
        pre_tok=_tile(pre_tok, n_bins), dec_tok=_tile(0.0, n_bins),
        decode_batch=_tile(0.0, n_bins), pre_active=_tile(concurrency, n_bins),
        L_pre=_tile(L, n_bins), ctx_dec=_tile(0.0, n_bins),
        iters_pre=_tile(iters, n_bins), iters_dec=_tile(0.0, n_bins),
        busy=_tile(1.0, n_bins),
    )


def serve(arch, hw, tp, server: ServerConfig, workload: Workload, n_bins=120) -> Dict:
    """STATE for a mixed steady serving point (prefill + decode coexisting).

    decode batch from the saturation model; decode tok/s from the analytic step
    time; prefill tok/s = request_rate * input_len (the prompt ingestion rate).
    """
    arch = A.normalize_arch(arch)
    B = sustained_decode_batch(arch, hw, tp, server, workload)
    ctx = workload.input_len + workload.output_len / 2.0
    t = decode_step_time(arch, hw, tp, B)
    iters_dec = 1.0 / t
    dec_tok = B * iters_dec
    # prefill ingestion rate: requests/s * prompt length (or implied by batch/T)
    lam = workload.request_rate
    if lam <= 0 and workload.output_len > 0:
        lam = dec_tok / workload.output_len  # completed seqs/s in steady state
    pre_tok = lam * workload.input_len
    flops_seq = 2.0 * float(arch["n_active"]) * workload.input_len \
        + float(A.attn_flops_per_seq(arch, workload.input_len))
    t_pre_seq = max(flops_seq / (tp * A.HW[hw]["peak_flops"]), 1e-6)
    pre_active = lam * t_pre_seq
    return dict(
        pre_tok=_tile(pre_tok, n_bins), dec_tok=_tile(dec_tok, n_bins),
        decode_batch=_tile(B, n_bins), pre_active=_tile(pre_active, n_bins),
        L_pre=_tile(workload.input_len, n_bins), ctx_dec=_tile(ctx, n_bins),
        iters_pre=_tile(pre_active / max(t_pre_seq, 1e-6) / max(pre_active, 1e-9)
                        if pre_active > 0 else 0.0, n_bins),
        iters_dec=_tile(iters_dec, n_bins),
        busy=_tile(1.0, n_bins),
    )
