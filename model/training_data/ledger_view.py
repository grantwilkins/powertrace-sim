"""Ledger feature view: request timing -> per-bin work rates (data-path Layer 2).

This is the single shared home of the arrivals->work reconstruction engine;
both legacy and bundle builders consume it. The 30-minute fold goes through the shared
``align_arrivals`` policy function (bit-identical operations).

feature-test is not a package, which is why the shared core lives here.
"""

from __future__ import annotations

import numpy as np

from model.training_data.alignment import align_arrivals

KV_ELEM_BYTES = 2.0


def bin_work_rates(pre_tok, dec_tok, batch, pre_active, pre_iter, kv_read,
                   arch, tp, nb):
    """Convert reconstructed activity into conserved physical work rates."""
    dec_iter = dec_tok / np.maximum(batch, 1e-9)
    iters = dec_iter + pre_iter
    if arch["moe_frac"] > 0:
        touched = np.minimum(1.0, batch * arch["top_k"] / arch["n_experts"])
        w_eff_dec = arch["w_bytes"] * ((1 - arch["moe_frac"]) + arch["moe_frac"] * touched)
    else:
        w_eff_dec = np.full(nb, arch["w_bytes"])
    w_read_dec = w_eff_dec * dec_iter
    w_read_pre = arch["w_bytes"] * pre_iter
    w_read = w_read_dec + w_read_pre
    kv_tok = 2.0 * arch["n_layers"] * arch["n_kv"] * arch["head_dim"] * KV_ELEM_BYTES
    kv_write = (pre_tok + dec_tok) * kv_tok
    tok_rate = pre_tok + dec_tok
    comm = (
        tok_rate * arch["n_layers"] * 2.0
        * 2.0 * arch["d_model"] * 2.0 * (tp - 1.0) / max(tp, 1)
    )
    return dict(
        pre_tok=pre_tok, dec_tok=dec_tok, batch=batch, pre_active=pre_active,
        iters=iters, w_read=w_read, w_read_pre=w_read_pre, w_read_dec=w_read_dec,
        kv_read=kv_read, kv_write=kv_write, comm=comm,
    )


def reconstruct_bins(
    req, pw, arch, tp, lambda_prefill, dt=1.0, trim_s=5.0,
    arrival_alignment="fold_1800",
):
    """Reconstruct per-bin work rates from request timing + power.

    The state is reconstructed from TTFT and inter-token latency. With
    ``arch['n_linear_layers'] in (0, absent)`` this uses the softmax KV path.

Legacy inputs retain the historical 30-minute fold. Canonical bundles pass
``arrival_alignment='exact_epoch'`` after their recorded local UTC offset has
converted nvidia-smi wall time to Unix epoch; bundle alignment never folds time.
    """
    if req is None or pw is None or not req["has_timestamps"]:
        return None

    p_ts, p_w = pw["timestamps"], pw["power"]
    t0 = float(p_ts[0])
    if arrival_alignment == "exact_epoch":
        arr = np.asarray(req["request_timestamps"], dtype=np.float64) - t0
        alignment_ok = arr.size > 0 and -2.0 <= float(np.min(arr)) <= 600.0
    elif arrival_alignment == "fold_1800":
        arr, alignment_ok, _ = align_arrivals(
            req["request_timestamps"], t0, policy="fold_1800"
        )
    else:
        raise ValueError(f"Unknown ledger arrival alignment: {arrival_alignment!r}")
    if not alignment_ok:
        return None
    ttft, dec = req["ttfts"], req["decode_times"]
    n_in, n_out = req["input_lens"], req["output_lens"]

    # TTFT includes queueing; the actual prefill burst ends when the first
    # token is emitted and lasts ~ n_in / lambda_prefill. Place it at the end
    # of the TTFT window (queue wait contributes no work).
    pre_e = arr + ttft
    pre_dur = np.minimum(np.maximum(n_in / max(lambda_prefill, 1e-3), 1e-3), ttft)
    pre_s = pre_e - pre_dur
    dec_s, dec_e = pre_e, pre_e + dec
    run_end = min(float(p_ts[-1] - t0), float(np.max(dec_e)))
    run_start = max(trim_s, float(np.min(arr)))
    if run_end - run_start < 10 * dt:
        return None
    n_full_bins = int(np.floor((run_end - run_start) / dt))
    if n_full_bins < 10:
        return None
    edges = run_start + np.arange(n_full_bins + 1, dtype=np.float64) * dt
    nb = edges.size - 1

    p_rel = p_ts - t0
    idx = np.searchsorted(edges, p_rel) - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(p_w)
    pow_sum = np.bincount(idx[ok], weights=p_w[ok], minlength=nb)
    pow_cnt = np.bincount(idx[ok], minlength=nb)
    valid = pow_cnt > 0
    power = np.where(valid, pow_sum / np.maximum(pow_cnt, 1), np.nan)

    pre_rate = n_in / np.maximum(pre_dur, 1e-3)
    dec_rate = n_out / np.maximum(dec, 1e-3)
    kv_tok = 2.0 * arch["n_layers"] * arch["n_kv"] * arch["head_dim"] * KV_ELEM_BYTES
    swa = float(arch["swa_window"])
    n_lin = int(arch.get("n_linear_layers", 0) or 0)

    pre_tok = np.zeros(nb)
    dec_tok = np.zeros(nb)
    batch = np.zeros(nb)
    pre_active = np.zeros(nb)
    pre_iter = np.zeros(nb)
    kv_read = np.zeros(nb)

    for j in range(arr.size):
        lo_bin = max(0, int((pre_s[j] - run_start) // dt))
        hi_bin = min(nb, int((dec_e[j] - run_start) // dt) + 1)
        if hi_bin <= lo_bin:
            continue
        b_lo = edges[lo_bin:hi_bin]
        b_hi = b_lo + dt
        ov_pre = np.clip(np.minimum(pre_e[j], b_hi) - np.maximum(pre_s[j], b_lo), 0.0, None)
        ov_dec = np.clip(np.minimum(dec_e[j], b_hi) - np.maximum(dec_s[j], b_lo), 0.0, None)
        sl = slice(lo_bin, hi_bin)
        pre_tok[sl] += pre_rate[j] * ov_pre / dt
        dec_tok[sl] += dec_rate[j] * ov_dec / dt
        batch[sl] += ov_dec / dt
        pre_active[sl] += ov_pre / dt
        pre_iter[sl] += ov_pre / max(float(pre_dur[j]), 1e-3) / dt
        mid = (b_lo + b_hi) / 2.0
        prog = np.clip((mid - dec_s[j]) / max(float(dec[j]), 1e-3), 0.0, 1.0)
        ctx = n_in[j] + prog * n_out[j]
        ctx_eff = ctx if swa <= 0 else 0.5 * ctx + 0.5 * np.minimum(ctx, swa)
        if n_lin <= 0:
            # Softmax KV grows with effective context.
            kv_read[sl] += dec_rate[j] * (ov_dec / dt) * ctx_eff * kv_tok
        else:
            # Hybrid: softmax layers keep growing KV; linear/lightning layers
            # carry a constant recurrent state (~head_dim), not ctx. Provisional
            # work rate; the linear-attention fit term is downstream follow-up.
            n_lay = max(int(arch["n_layers"]), 1)
            soft_frac = (n_lay - n_lin) / n_lay
            lin_frac = n_lin / n_lay
            ctx_soft = ctx_eff * soft_frac
            ctx_lin = float(arch["head_dim"]) * lin_frac
            kv_read[sl] += dec_rate[j] * (ov_dec / dt) * (ctx_soft + ctx_lin) * kv_tok

    out = bin_work_rates(
        pre_tok, dec_tok, batch, pre_active, pre_iter, kv_read, arch, tp, nb
    )
    keep = valid & np.isfinite(power)
    n = int(keep.sum())
    if n == 0:
        return None
    out = {k: v[keep] for k, v in out.items()}
    out["power"] = power[keep]
    arch_scalar = {k: float(v) for k, v in arch.items() if k != "family"}
    return dict(out, n=n, arch=arch_scalar)


def reconstruct_bins_from_record(record, *, lambda_prefill, dt=1.0, trim_s=5.0):
    """Ledger view over a RunRecord: per-bin work rates + power.

    Feeds the record's requests table and TP-sum power through the same
    reconstruction engine the legacy builders use.
    """
    req = {
        "request_timestamps": record.request_timestamps,
        "ttfts": record.ttfts,
        "decode_times": record.decode_times,
        "input_lens": record.input_lens,
        "output_lens": record.output_lens,
        "has_timestamps": record.has_timestamps,
    }
    pw = {"timestamps": record.power_timestamps, "power": record.tp_sum_power()}
    return reconstruct_bins(
        req, pw, record.arch, record.tp, lambda_prefill, dt=dt, trim_s=trim_s,
        arrival_alignment=(
            "exact_epoch" if record.source_layout == "bundle" else "fold_1800"
        ),
    )
