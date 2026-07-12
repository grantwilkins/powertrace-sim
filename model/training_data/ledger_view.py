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


def exact_itl_mask(output_tokens, decode_itls):
    """Identify requests with one measured interval per post-first token."""
    return np.asarray([
        float(tokens).is_integer()
        and float(tokens) >= 0.0
        and len(intervals) == max(int(tokens) - 1, 0)
        for tokens, intervals in zip(output_tokens, decode_itls)
    ])


def schedule_work_rates(
    arrivals, prefill_starts, prefill_ends, decode_ends, input_tokens,
    output_tokens, edges, arch, tp, *, decode_itls=None,
):
    """Project one executed request schedule onto half-open time bins.

    ``decode_itls`` makes retrospective timing exact: every recorded interval
    carries one decode iteration's work. Omitting it retains the explicit
    throughput-modeled path used for arrival-only inference.
    """
    values = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (
            arrivals, prefill_starts, prefill_ends, decode_ends,
            input_tokens, output_tokens,
        )
    ]
    if len({value.size for value in values}) != 1:
        raise ValueError("Request schedule columns have different lengths")
    arr, pre_s, pre_e, dec_e, n_in, n_out = values
    edges = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("At least two finite bin edges are required")
    widths = np.diff(edges)
    if np.any(widths <= 0.0) or not np.allclose(widths, widths[0]):
        raise ValueError("Ledger bins must be a uniform increasing grid")
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("Request schedule values must be finite")
    if np.any(n_in < 0.0) or np.any(n_out < 0.0):
        raise ValueError("Request token counts must be non-negative")
    if np.any(pre_s < arr) or np.any(pre_e < pre_s) or np.any(dec_e < pre_e):
        raise ValueError("Request schedule intervals are not ordered")
    if decode_itls is not None and len(decode_itls) != arr.size:
        raise ValueError("Measured decode interval rows have different lengths")
    if decode_itls is not None:
        expected = np.rint(n_out).astype(np.int64) - 1
        if np.any(n_out != np.rint(n_out)) or np.any(expected < -1):
            raise ValueError("Measured output token counts must be non-negative integers")
        expected = np.maximum(expected, 0)
        if any(len(row) != count for row, count in zip(decode_itls, expected)):
            raise ValueError("Measured decode interval count must equal output tokens minus one")

    dt, nb = float(widths[0]), edges.size - 1
    bin_lo, bin_hi = edges[:-1], edges[1:]
    pre_tok = np.zeros(nb)
    dec_tok = np.zeros(nb)
    batch = np.zeros(nb)
    pre_active = np.zeros(nb)
    pre_iter = np.zeros(nb)
    kv_read = np.zeros(nb)
    measured_event_bins = []
    measured_event_contexts = []
    kv_tok = 2.0 * arch["n_layers"] * arch["n_kv"] * arch["head_dim"] * KV_ELEM_BYTES
    swa = float(arch.get("swa_window", 0.0))
    n_lin = int(arch.get("n_linear_layers", 0) or 0)

    for j in range(arr.size):
        pre_dur = pre_e[j] - pre_s[j]
        dec_dur = dec_e[j] - pre_e[j]
        ov_pre = np.clip(np.minimum(pre_e[j], bin_hi) - np.maximum(pre_s[j], bin_lo), 0.0, None)
        ov_dec = np.clip(np.minimum(dec_e[j], bin_hi) - np.maximum(pre_e[j], bin_lo), 0.0, None)
        if pre_dur > 0.0:
            pre_tok += (n_in[j] / pre_dur) * ov_pre / dt
            pre_active += ov_pre / dt
            pre_iter += ov_pre / pre_dur / dt
        if decode_itls is not None:
            intervals = np.asarray(decode_itls[j], dtype=np.float64).reshape(-1)
            if intervals.size == 0:
                if dec_dur != 0.0:
                    raise ValueError("Empty measured decode intervals require zero duration")
                continue
            if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0.0):
                raise ValueError("Measured decode intervals must be finite and positive")
            if not np.isclose(intervals.sum(), dec_dur, rtol=1e-9, atol=1e-9):
                raise ValueError("Measured decode intervals do not match decode duration")
            events = pre_e[j] + np.cumsum(intervals)
            event_bins = np.searchsorted(edges, events, side="right") - 1
            in_grid = (event_bins >= 0) & (event_bins < nb)
            measured_event_bins.append(event_bins[in_grid])
            measured_event_contexts.append(n_in[j] + np.arange(1, intervals.size + 1)[in_grid])
            batch += ov_dec / dt
        elif dec_dur > 0.0:
            dec_rate = n_out[j] / dec_dur
            dec_tok += dec_rate * ov_dec / dt
            batch += ov_dec / dt
            progress = np.clip(((bin_lo + bin_hi) / 2.0 - pre_e[j]) / dec_dur, 0.0, 1.0)
            context = n_in[j] + progress * n_out[j]
            context_effective = context if swa <= 0.0 else 0.5 * context + 0.5 * np.minimum(context, swa)
            if n_lin > 0:
                linear_fraction = n_lin / max(int(arch["n_layers"]), 1)
                context_effective = (
                    context_effective * (1.0 - linear_fraction)
                    + float(arch["head_dim"]) * linear_fraction
                )
            kv_read += dec_rate * (ov_dec / dt) * context_effective * kv_tok

    if measured_event_bins:
        event_bins = np.concatenate(measured_event_bins)
        contexts = np.concatenate(measured_event_contexts)
        if swa > 0.0:
            contexts = 0.5 * contexts + 0.5 * np.minimum(contexts, swa)
        if n_lin > 0:
            linear_fraction = n_lin / max(int(arch["n_layers"]), 1)
            contexts = (
                contexts * (1.0 - linear_fraction)
                + float(arch["head_dim"]) * linear_fraction
            )
        dec_tok += np.bincount(event_bins, minlength=nb) / dt
        kv_read += np.bincount(event_bins, weights=contexts, minlength=nb) * kv_tok / dt

    out = bin_work_rates(
        pre_tok, dec_tok, batch, pre_active, pre_iter, kv_read, arch, tp, nb
    )
    arrival_bin = np.searchsorted(edges, arr, side="right") - 1
    in_grid = (arrival_bin >= 0) & (arrival_bin < nb)
    out["arrivals"] = np.bincount(arrival_bin[in_grid], minlength=nb) / dt
    out["input_tokens_arriving"] = np.bincount(
        arrival_bin[in_grid], weights=n_in[in_grid], minlength=nb
    ) / dt
    out["output_tokens_requested"] = np.bincount(
        arrival_bin[in_grid], weights=n_out[in_grid], minlength=nb
    ) / dt
    arrived = arr[:, None] < bin_hi
    unfinished = arrived & (dec_e[:, None] > bin_hi)
    running = unfinished & (pre_s[:, None] <= bin_hi)
    out["A_t"] = unfinished.sum(axis=0).astype(np.float64)
    out["running_requests"] = running.sum(axis=0).astype(np.float64)
    out["waiting_requests"] = out["A_t"] - out["running_requests"]
    out["delta_A_t"] = np.r_[0.0, np.diff(out["A_t"])]
    return out


def bin_work_rates(pre_tok, dec_tok, batch, pre_active, pre_iter, kv_read,
                   arch, tp, nb):
    """Convert reconstructed activity into conserved physical work rates."""
    dec_iter = dec_tok / np.maximum(batch, 1e-9)
    iters = dec_iter + pre_iter
    if arch["moe_frac"] > 0:
        touched = 1.0 - (1.0 - arch["top_k"] / arch["n_experts"]) ** batch
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
    arrival_alignment="fold_1800", include_time=False,
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
    decode_itls = req.get("itls")
    if decode_itls is not None:
        exact = exact_itl_mask(n_out, decode_itls)
        arr, ttft, dec, n_in, n_out = (
            np.asarray(values)[exact] for values in (arr, ttft, dec, n_in, n_out)
        )
        decode_itls = np.asarray(decode_itls, dtype=object)[exact]
        if n_out.size == 0:
            return None

    # TTFT includes queueing; the actual prefill burst ends when the first
    # token is emitted and lasts ~ n_in / lambda_prefill. Place it at the end
    # of the TTFT window (queue wait contributes no work).
    pre_e = arr + ttft
    pre_dur = np.minimum(np.maximum(n_in / max(lambda_prefill, 1e-3), 1e-3), ttft)
    pre_s = pre_e - pre_dur
    dec_e = pre_e + dec
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

    out = schedule_work_rates(
        arr, pre_s, pre_e, dec_e, n_in, n_out, edges, arch, tp,
        decode_itls=decode_itls,
    )
    keep = valid & np.isfinite(power)
    n = int(keep.sum())
    if n == 0:
        return None
    out = {k: v[keep] for k, v in out.items()}
    out["power"] = power[keep]
    if include_time:
        out["time_epoch_s"] = (t0 + edges[1:])[keep]
    arch_scalar = {k: float(v) for k, v in arch.items() if k != "family"}
    return dict(out, n=n, arch=arch_scalar)


def reconstruct_bins_from_record(
    record, *, lambda_prefill, dt=1.0, trim_s=5.0, include_time=False
):
    """Ledger view over a RunRecord: per-bin work rates + power.

    Feeds the record's requests table and TP-sum power through the same
    reconstruction engine the legacy builders use.
    """
    req = {
        "request_timestamps": record.request_timestamps,
        "ttfts": record.ttfts,
        "itls": record.itls,
        "decode_times": record.decode_times,
        "input_lens": record.input_lens,
        "output_lens": record.output_lens,
        "has_timestamps": record.has_timestamps,
    }
    pw = {"timestamps": record.power_timestamps, "power": record.tp_sum_power()}
    return reconstruct_bins(
        req, pw, record.arch, record.tp, lambda_prefill, dt=dt, trim_s=trim_s,
        include_time=include_time,
        arrival_alignment=(
            "exact_epoch" if record.source_layout == "bundle" else "fold_1800"
        ),
    )
