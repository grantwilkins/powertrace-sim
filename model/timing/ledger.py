"""Project simulated engine iterations onto the selected 250 ms work ledger."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from model.timing.iteration import iteration_work
from model.training_data.moe_routing import RoutingLaw, expected_iteration_weight_bytes
from model.training_data.ledger_view import bin_work_rates, effective_context, kv_bytes_per_token

NATIVE_DT_S = 0.25

ALL_ROLES = ("train", "test_indomain", "holdout_model", "holdout_twin",
             "holdout_rate", "dtype_calibration")


def selected_roles(manifest: dict, requested: str) -> tuple[str, ...]:
    if requested == "all":
        return tuple(sorted(set(manifest["roles"].values())))
    return tuple(requested.split(","))


def _overlap_add(out, edges, t0, t1, weight):
    """out[b] += weight_j * overlap([t0_j, t1_j), bin b) for every interval j."""
    dt = float(edges[1] - edges[0])
    nb = edges.size - 1
    span = int(np.max(np.ceil((t1 - t0) / dt))) + 1 if t0.size else 0
    b_lo = np.floor((t0 - edges[0]) / dt).astype(np.int64)
    for k in range(span + 1):
        b = b_lo + k
        lo = edges[0] + b * dt
        ov = np.clip(np.minimum(t1, lo + dt) - np.maximum(t0, lo), 0.0, None)
        keep = (b >= 0) & (b < nb) & (ov > 0.0)
        np.add.at(out, b[keep], (weight * ov)[keep])


def emit_bins(
    trace, per_request, *, arch, tp, dt=NATIVE_DT_S,
    routing_law: RoutingLaw | None = None,
    horizon_s: float | None = None,
):
    """Simulated iteration trace + per-request timing -> ledger channels.

    ``trace`` is the ``iteration_trace`` list from ``simulate_requests``;
    ``per_request`` its return value. Returns the measured ledger's per-bin
    channel dict (power absent) plus ``busy``, with ``n`` and ``arch``.
    """
    trace = np.asarray(trace, dtype=np.float64)
    if trace.ndim != 2 or trace.shape[1] not in (7, 9, 12, 18):
        raise ValueError("Iteration traces must use 7, 9, 12, or 18 fields")
    t0, t1, dec_batch, context_mean, chunk_tokens, chunk_context, n_chunks = (
        trace[:, :7].T
    )
    prefill_logits = trace[:, 7] if trace.shape[1] >= 9 else np.zeros_like(t0)
    recorded_weight_bytes = trace[:, 8] if trace.shape[1] >= 9 else None
    if trace.shape[1] == 18:
        gemm_flops, attn_flops, attn_bytes = trace[:, 9:12].T
        (prefill_gemm_flops, decode_gemm_flops,
         prefill_attn_flops, decode_attn_flops,
         prefill_attn_bytes, decode_attn_bytes) = trace[:, 12:18].T
    else:
        work = [
            iteration_work(
                arch, decode_batch=decode, context_mean=context,
                prefill_chunks=[(prefill, prior)] if prefill > 0 else [],
                prefill_logits=logits, routing_law=routing_law,
            )
            for decode, context, prefill, prior, logits in zip(
                dec_batch, context_mean, chunk_tokens, chunk_context,
                prefill_logits,
            )
        ]
        if trace.shape[1] == 12:
            gemm_flops, attn_flops, attn_bytes = trace[:, 9:12].T
        else:
            gemm_flops = np.asarray([row["gemm_flops"] for row in work])
            attn_flops = np.asarray([row["attn_flops"] for row in work])
            attn_bytes = np.asarray([row["attn_bytes"] for row in work])
        phase_keys = (
            "prefill_gemm_flops", "decode_gemm_flops",
            "prefill_attn_flops", "decode_attn_flops",
            "prefill_attn_bytes", "decode_attn_bytes",
        )
        (prefill_gemm_flops, decode_gemm_flops,
         prefill_attn_flops, decode_attn_flops,
         prefill_attn_bytes, decode_attn_bytes) = (
            np.asarray([row[key] for row in work]) for key in phase_keys
        )
    arr = np.asarray([r["arrival_s"] for r in per_request], dtype=np.float64)
    adm = np.asarray([r["admitted_s"] for r in per_request], dtype=np.float64)
    n_in = np.asarray([r["n_in"] for r in per_request], dtype=np.float64)
    n_out = np.asarray([r["n_out"] for r in per_request], dtype=np.float64)
    dec_e = arr + np.asarray([r["e2e_s"] for r in per_request], dtype=np.float64)

    predicted_end = max(float(t1.max()) if t1.size else 0.0,
                        float(dec_e.max()) if dec_e.size else 0.0)
    t_max = predicted_end if horizon_s is None else float(horizon_s)
    if not np.isfinite(t_max) or t_max <= 0.0:
        raise ValueError("Ledger horizon must be positive and finite")
    # One bin past t_max: completion events use right-open bins, so a token
    # landing exactly on the final edge must still fall inside the grid.
    nb = int(np.floor(t_max / dt)) + 1
    edges = np.arange(nb + 1, dtype=np.float64) * dt
    wall = t1 - t0
    if np.any(wall <= 0.0):
        raise ValueError("Iteration trace records must have positive duration")
    iteration_tokens = dec_batch + chunk_tokens
    if np.any(iteration_tokens <= 0.0):
        raise ValueError("Iteration trace records must schedule at least one token")

    pre_tok = np.zeros(nb)
    batch = np.zeros(nb)
    pre_active = np.zeros(nb)
    pre_iter = np.zeros(nb)
    busy = np.zeros(nb)
    prefill_duty = np.zeros(nb)
    decode_duty = np.zeros(nb)
    w_read = np.zeros(nb)
    w_read_pre = np.zeros(nb)
    w_read_dec = np.zeros(nb)
    gemm_flops_rate = np.zeros(nb)
    attn_flops_rate = np.zeros(nb)
    attn_bytes_rate = np.zeros(nb)
    prefill_gemm_flops_rate = np.zeros(nb)
    decode_gemm_flops_rate = np.zeros(nb)
    prefill_attn_flops_rate = np.zeros(nb)
    decode_attn_flops_rate = np.zeros(nb)
    prefill_attn_bytes_rate = np.zeros(nb)
    decode_attn_bytes_rate = np.zeros(nb)
    _overlap_add(pre_tok, edges, t0, t1, chunk_tokens / wall / dt)
    _overlap_add(batch, edges, t0, t1, dec_batch / dt)
    _overlap_add(pre_active, edges, t0, t1, n_chunks / dt)
    _overlap_add(pre_iter, edges, t0, t1, (chunk_tokens > 0) / wall / dt)
    _overlap_add(busy, edges, t0, t1, np.ones_like(wall) / dt)
    _overlap_add(prefill_duty, edges, t0, t1, (chunk_tokens > 0) / dt)
    _overlap_add(decode_duty, edges, t0, t1, (dec_batch > 0) / dt)
    weight_bytes = (
        recorded_weight_bytes
        if recorded_weight_bytes is not None
        else np.asarray([
            expected_iteration_weight_bytes(
                arch, decode_tokens=decode, prefill_tokens=prefill,
                prefill_groups=groups, routing_law=routing_law)
            for decode, prefill, groups in zip(dec_batch, chunk_tokens, n_chunks)
        ])
    )
    weight_rate = weight_bytes / wall / dt
    _overlap_add(w_read, edges, t0, t1, weight_rate)
    _overlap_add(
        w_read_pre, edges, t0, t1, weight_rate * chunk_tokens / iteration_tokens)
    _overlap_add(
        w_read_dec, edges, t0, t1, weight_rate * dec_batch / iteration_tokens)
    _overlap_add(gemm_flops_rate, edges, t0, t1, gemm_flops / wall / dt)
    _overlap_add(attn_flops_rate, edges, t0, t1, attn_flops / wall / dt)
    _overlap_add(attn_bytes_rate, edges, t0, t1, attn_bytes / wall / dt)
    for output, values in (
        (prefill_gemm_flops_rate, prefill_gemm_flops),
        (decode_gemm_flops_rate, decode_gemm_flops),
        (prefill_attn_flops_rate, prefill_attn_flops),
        (decode_attn_flops_rate, decode_attn_flops),
        (prefill_attn_bytes_rate, prefill_attn_bytes),
        (decode_attn_bytes_rate, decode_attn_bytes),
    ):
        _overlap_add(output, edges, t0, t1, values / wall / dt)

    # Decode tokens complete at the iteration end, one per decoding sequence,
    # each reading its (mean) context from the KV cache.
    event_bins = np.searchsorted(edges, t1, side="right") - 1
    iteration_in_grid = (event_bins >= 0) & (event_bins < nb)
    in_grid = iteration_in_grid & (dec_batch > 0)
    dec_tok = np.bincount(event_bins[in_grid], weights=dec_batch[in_grid],
                          minlength=nb) / dt
    kv_read = np.bincount(
        event_bins[in_grid],
        weights=(dec_batch * effective_context(context_mean, arch))[in_grid],
        minlength=nb,
    ) * kv_bytes_per_token(arch) / dt

    out = bin_work_rates(
        pre_tok, dec_tok, batch, pre_active, pre_iter, kv_read, arch, tp, nb
    )
    out["w_read"] = w_read
    out["w_read_pre"] = w_read_pre
    out["w_read_dec"] = w_read_dec
    out["gemm_flops_rate"] = gemm_flops_rate
    out["attn_flops_rate"] = attn_flops_rate
    out["attn_bytes_rate"] = attn_bytes_rate
    out["prefill_gemm_flops_rate"] = prefill_gemm_flops_rate
    out["decode_gemm_flops_rate"] = decode_gemm_flops_rate
    out["prefill_attn_flops_rate"] = prefill_attn_flops_rate
    out["decode_attn_flops_rate"] = decode_attn_flops_rate
    out["prefill_attn_bytes_rate"] = prefill_attn_bytes_rate
    out["decode_attn_bytes_rate"] = decode_attn_bytes_rate
    out["busy"] = busy
    out["prefill_duty"] = prefill_duty
    out["decode_duty"] = decode_duty
    out["engine_iterations_rate"] = np.bincount(
        event_bins[iteration_in_grid], minlength=nb).astype(np.float64) / dt
    out["engine_iteration_tokens_rate"] = np.bincount(
        event_bins[iteration_in_grid],
        weights=iteration_tokens[iteration_in_grid], minlength=nb) / dt
    out["logit_tokens_rate"] = np.bincount(
        event_bins[iteration_in_grid],
        weights=(dec_batch + prefill_logits)[iteration_in_grid], minlength=nb
    ) / dt
    out["engine_tokens_per_iteration"] = np.divide(
        out["engine_iteration_tokens_rate"], out["engine_iterations_rate"],
        out=np.zeros(nb, dtype=np.float64),
        where=out["engine_iterations_rate"] > 0.0,
    )

    bin_hi = edges[1:]
    arrival_bin = np.searchsorted(edges, arr, side="right") - 1
    ok = (arrival_bin >= 0) & (arrival_bin < nb)
    out["arrivals"] = np.bincount(arrival_bin[ok], minlength=nb) / dt
    out["input_tokens_arriving"] = np.bincount(
        arrival_bin[ok], weights=n_in[ok], minlength=nb) / dt
    out["output_tokens_requested"] = np.bincount(
        arrival_bin[ok], weights=n_out[ok], minlength=nb) / dt
    arrived = arr[:, None] < bin_hi
    unfinished = arrived & (dec_e[:, None] > bin_hi)
    running = unfinished & (adm[:, None] <= bin_hi)
    out["A_t"] = unfinished.sum(axis=0).astype(np.float64)
    out["running_requests"] = running.sum(axis=0).astype(np.float64)
    out["waiting_requests"] = out["A_t"] - out["running_requests"]
    out["delta_A_t"] = np.r_[0.0, np.diff(out["A_t"])]
    arch_scalar = {k: float(v) for k, v in arch.items() if k != "family"}
    return dict(out, n=nb, arch=arch_scalar)


def iter_bins(
    trace, per_request, *, arch, tp, dt=NATIVE_DT_S,
    horizon_s: float | None = None,
) -> Iterator[dict[str, float]]:
    """Yield selected-model ledger bins without retaining output-sized arrays."""
    records = np.asarray(trace, dtype=np.float64)
    if records.ndim != 2 or records.shape[1] != 18:
        raise ValueError("Streaming ledger requires the selected 18-field trace")
    if not np.isfinite(records).all():
        raise ValueError("Iteration trace records must be finite")
    (t0, t1, dec_batch, context_mean, chunk_tokens, _chunk_context,
     n_chunks, prefill_logits, weight_bytes, gemm_flops, attn_flops,
     attn_bytes, prefill_gemm, decode_gemm, prefill_attn,
     decode_attn, prefill_attn_bytes, decode_attn_bytes) = records.T
    wall = t1 - t0
    if np.any(wall <= 0.0) or np.any(np.diff(t0) < 0.0):
        raise ValueError("Iteration trace must be ordered with positive durations")
    iteration_tokens = dec_batch + chunk_tokens
    if np.any(iteration_tokens <= 0.0):
        raise ValueError("Iteration trace records must schedule at least one token")

    arrivals = np.asarray([row["arrival_s"] for row in per_request], dtype=float)
    admitted = np.asarray([row["admitted_s"] for row in per_request], dtype=float)
    inputs = np.asarray([row["n_in"] for row in per_request], dtype=float)
    outputs = np.asarray([row["n_out"] for row in per_request], dtype=float)
    completed = arrivals + np.asarray(
        [row["e2e_s"] for row in per_request], dtype=float
    )
    predicted_end = max(float(t1[-1]) if t1.size else 0.0,
                        float(completed.max()) if completed.size else 0.0)
    limit = predicted_end if horizon_s is None else float(horizon_s)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Ledger horizon must be positive and finite")
    count = int(np.floor(limit / dt)) + 1
    arrival_order = np.argsort(arrivals, kind="stable")
    sorted_arrivals = arrivals[arrival_order]
    sorted_inputs = inputs[arrival_order]
    sorted_outputs = outputs[arrival_order]
    admitted_order = np.argsort(admitted, kind="stable")
    sorted_admitted = admitted[admitted_order]
    admitted_arrivals = arrivals[admitted_order]
    sorted_completed = np.sort(completed)
    kv_token_bytes = kv_bytes_per_token(arch)
    previous_active = 0.0
    overlap_cursor = 0

    for index in range(count):
        lo, hi = index * dt, (index + 1) * dt
        while overlap_cursor < t1.size and t1[overlap_cursor] <= lo:
            overlap_cursor += 1
        end = overlap_cursor
        while end < t0.size and t0[end] < hi:
            end += 1
        section = slice(overlap_cursor, end)
        overlap = np.clip(
            np.minimum(t1[section], hi) - np.maximum(t0[section], lo),
            0.0, None,
        )
        def duty(values) -> float:
            return float(np.sum(np.asarray(values)[section] * overlap) / dt)

        pre_tok = duty(chunk_tokens / wall)
        batch = duty(dec_batch)
        pre_active = duty(n_chunks)
        pre_iter = duty((chunk_tokens > 0.0) / wall)
        busy = float(overlap.sum() / dt)
        prefill_duty = duty(chunk_tokens > 0.0)
        decode_duty = duty(dec_batch > 0.0)
        weight_rate = weight_bytes / wall
        w_read = duty(weight_rate)
        w_read_pre = duty(weight_rate * chunk_tokens / iteration_tokens)
        w_read_dec = duty(weight_rate * dec_batch / iteration_tokens)

        event_lo = int(np.searchsorted(t1, lo, side="left"))
        event_hi = int(np.searchsorted(t1, hi, side="left"))
        events = slice(event_lo, event_hi)
        decode_events = dec_batch[events]
        dec_tok = float(decode_events.sum() / dt)
        kv_read = float(np.sum(
            decode_events * effective_context(context_mean[events], arch)
        ) * kv_token_bytes / dt)
        engine_iterations = float((event_hi - event_lo) / dt)
        engine_tokens = float(iteration_tokens[events].sum() / dt)

        base = bin_work_rates(
            np.asarray([pre_tok]), np.asarray([dec_tok]), np.asarray([batch]),
            np.asarray([pre_active]), np.asarray([pre_iter]),
            np.asarray([kv_read]), arch, tp, 1,
        )
        row = {key: float(np.asarray(value)[0]) for key, value in base.items()}
        for key, values in (
            ("w_read", weight_rate),
            ("w_read_pre", weight_rate * chunk_tokens / iteration_tokens),
            ("w_read_dec", weight_rate * dec_batch / iteration_tokens),
            ("gemm_flops_rate", gemm_flops / wall),
            ("attn_flops_rate", attn_flops / wall),
            ("attn_bytes_rate", attn_bytes / wall),
            ("prefill_gemm_flops_rate", prefill_gemm / wall),
            ("decode_gemm_flops_rate", decode_gemm / wall),
            ("prefill_attn_flops_rate", prefill_attn / wall),
            ("decode_attn_flops_rate", decode_attn / wall),
            ("prefill_attn_bytes_rate", prefill_attn_bytes / wall),
            ("decode_attn_bytes_rate", decode_attn_bytes / wall),
        ):
            row[key] = duty(values)
        row.update({
            "busy": busy,
            "prefill_duty": prefill_duty,
            "decode_duty": decode_duty,
            "engine_iterations_rate": engine_iterations,
            "engine_iteration_tokens_rate": engine_tokens,
            "logit_tokens_rate": float(
                (dec_batch[events] + prefill_logits[events]).sum() / dt
            ),
            "engine_tokens_per_iteration": (
                engine_tokens / engine_iterations if engine_iterations else 0.0
            ),
        })
        arrival_lo = int(np.searchsorted(sorted_arrivals, lo, side="left"))
        arrival_hi = int(np.searchsorted(sorted_arrivals, hi, side="left"))
        arrival_slice = slice(arrival_lo, arrival_hi)
        active = float(
            np.searchsorted(sorted_arrivals, hi, side="left")
            - np.searchsorted(sorted_completed, hi, side="right")
        )
        admitted_left = int(np.searchsorted(sorted_admitted, hi, side="left"))
        admitted_right = int(np.searchsorted(sorted_admitted, hi, side="right"))
        admitted_at_edge = np.count_nonzero(
            admitted_arrivals[admitted_left:admitted_right] < hi
        )
        running = float(
            admitted_left + admitted_at_edge
            - np.searchsorted(sorted_completed, hi, side="right")
        )
        row.update({
            "arrivals": float((arrival_hi - arrival_lo) / dt),
            "input_tokens_arriving": float(sorted_inputs[arrival_slice].sum() / dt),
            "output_tokens_requested": float(sorted_outputs[arrival_slice].sum() / dt),
            "A_t": active,
            "running_requests": running,
            "waiting_requests": active - running,
            "delta_A_t": active - previous_active if index else 0.0,
        })
        previous_active = active
        yield row
