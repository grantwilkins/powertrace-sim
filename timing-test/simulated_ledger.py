"""Simulated-ledger emitter: iteration trace -> measured-ledger 250 ms channels.

Projects the scheduler simulation's per-iteration trace onto a uniform time
grid and emits the exact channel names and semantics of the measured ledger
cache (feature-test/ledger_cache_250ms.npz), so the power stage's physics
code runs unchanged in arrival-only mode. Activity arrays feed the shared
``bin_work_rates`` from model/training_data/ledger_view.py, then simulated
weight traffic is replaced by the exact per-iteration weight bytes used by
the timing model.

Channel semantics mirror the measured exact-ITL reconstruction path:
- decode tokens and their KV reads are events placed at the iteration end
  (the measured path bins token-completion events the same way);
- coverage channels (batch, pre_active, pre_iter, prefill token rate) are
  interval overlaps of each iteration with the bins;
- weight bytes are spread over each iteration and conserved across bins;
  mixed-iteration bytes are split between phase channels by scheduled tokens;
- ``running_requests``/``waiting_requests`` use the simulator's true seat
  admission time instead of the measured builder's TTFT-based heuristic.

Times are the raw simulator clock (arrivals at their recorded offsets); the
meter onset delay and reading average belong to the power response chain,
not to this work clock.

The emitter also carries simulator-only engine coordinates that the legacy
request reconstruction could not observe: ``busy``, phase duty fractions,
exact iteration and scheduled-token rates, tokens per iteration, and the
GEMM/attention work used by the timing roofline. The exact iteration channels
come from iteration-completion events; they never reuse the reconstruction-only
``dec_tok / batch`` approximation.

Build the simulated cache over the timing dataset:
    uv run python timing-test/simulated_ledger.py \
        [--dt 0.25] [--roles all] [--out feature-test/ledger_cache_sim_250ms.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_timing import MAX_NUM_SEQS  # noqa: E402
from iteration_time import (  # noqa: E402
    iteration_work,
    launch_overhead_s,
    transformer_bw_scale,
)
from scheduler_sim import EngineConfig, simulate_requests  # noqa: E402
from model.training_data.arch import ARCH, get_arch  # noqa: E402
from model.training_data.moe_routing import (  # noqa: E402
    RoutingLaw,
    expected_iteration_weight_bytes,
    load_routing_laws,
)
from model.training_data.ledger_view import (  # noqa: E402
    bin_work_rates,
    effective_context,
    kv_bytes_per_token,
)

BASE = Path(__file__).resolve().parent
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
    trace, per_request, *, arch, tp, dt=0.25,
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


def simulate_run(data, rid, fitted, routing_laws=None):
    """Replay one timing-dataset run with the frozen efficiencies."""
    hardware = str(data["run_hardware"][rid])
    model = str(data["run_model"][rid])
    tp = int(data["run_tp"][rid])
    arch = dict(get_arch(model))
    routing_law = (routing_laws or {}).get(model)
    params = fitted[hardware]
    idx = np.flatnonzero(data["req_run_id"] == rid)
    order = idx[np.argsort(data["arrival_time_s"][idx], kind="stable")]
    requests = [(float(data["arrival_time_s"][i]),
                 int(data["input_tokens"][i]),
                 int(data["output_tokens"][i])) for i in order]
    trace: list = []
    per_request = simulate_requests(
        requests, arch=arch, hardware=hardware, tp=tp,
        eff_flops=params["eff_flops"], eff_bw=params["eff_bw"],
        t_launch_s=launch_overhead_s(
            arch, base_s=params["base_overhead_s"],
            per_message_s=params["per_message_s"][str(tp)]),
        t_sample_s=params.get("per_token_sample_s", 0.0),
        transformer_bw_scale=transformer_bw_scale(arch, params, hardware),
        engine=EngineConfig(max_num_seqs=MAX_NUM_SEQS.get(model, 256)),
        iteration_trace=trace, routing_law=routing_law)
    return trace, per_request, arch, tp, routing_law


HARDWARE_INDEX = {"A100": 0, "H100": 1}
PER_BIN_CHANNELS = (
    "pre_tok", "dec_tok", "batch", "pre_active", "iters",
    "w_read", "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm",
    "arrivals", "input_tokens_arriving", "output_tokens_requested",
    "A_t", "delta_A_t", "running_requests", "waiting_requests", "busy",
    "prefill_duty", "decode_duty", "engine_iteration_tokens_rate",
    "engine_iterations_rate", "engine_tokens_per_iteration",
    "logit_tokens_rate", "gemm_flops_rate", "attn_flops_rate",
    "attn_bytes_rate", "prefill_gemm_flops_rate",
    "decode_gemm_flops_rate", "prefill_attn_flops_rate",
    "decode_attn_flops_rate", "prefill_attn_bytes_rate",
    "decode_attn_bytes_rate",
)


def measured_run_horizon(data: dict, run_id: int) -> float:
    selected = np.asarray(data["req_run_id"]) == run_id
    if not selected.any():
        raise ValueError(f"Run {run_id} has no requests")
    completion = (
        np.asarray(data["arrival_time_s"], float)[selected]
        + np.asarray(data["ttft_s"], float)[selected]
        + np.asarray(data["decode_duration_s"], float)[selected]
    )
    return float(np.max(completion))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--roles", default="all",
                        help="comma-separated manifest roles, or 'all'")
    parser.add_argument("--manifest", default=str(BASE / "split_manifest_fp8.json"))
    parser.add_argument("--out", default="feature-test/ledger_cache_sim_250ms.npz")
    parser.add_argument(
        "--fitted", default=str(BASE / "fitted_efficiencies.json")
    )
    parser.add_argument(
        "--moe-routing", choices=("uniform", "measured"), default="uniform")
    args = parser.parse_args()

    data = dict(np.load(BASE / "timing_dataset.npz", allow_pickle=False))
    manifest = json.loads(Path(args.manifest).read_text())
    fitted = json.loads(Path(args.fitted).read_text())
    routing_laws = load_routing_laws() if args.moe_routing == "measured" else {}
    wanted = selected_roles(manifest, args.roles)
    roles = {int(k): v for k, v in manifest["roles"].items() if v in wanted}
    role_names = sorted(set(roles.values()))
    family_names = sorted({value["family"] for value in ARCH.values()})

    cols = defaultdict(list)
    counts = defaultdict(int)
    for rid, role in sorted(roles.items()):
        trace, per_request, arch, tp, routing_law = simulate_run(
            data, rid, fitted, routing_laws)
        bins = emit_bins(
            trace, per_request, arch=arch, tp=tp, dt=args.dt,
            routing_law=routing_law,
            horizon_s=measured_run_horizon(data, rid))
        n = bins["n"]
        for key in PER_BIN_CHANNELS:
            cols[key].append(bins[key])
        cols["power"].append(np.full(n, np.nan))
        arch_scalar = bins["arch"]
        cols["n_active"].append(np.full(n, arch_scalar["n_active"]))
        cols["transformer_active_params"].append(np.full(
            n, arch_scalar.get("transformer_active_params", arch_scalar["n_active"])
        ))
        cols["output_head_params"].append(np.full(
            n, arch_scalar.get("output_head_params", 0.0)
        ))
        cols["w_bytes"].append(np.full(n, arch_scalar["w_bytes"]))
        cols["fp8"].append(np.full(n, arch_scalar["fp8"]))
        cols["fp8_flop_frac"].append(np.full(n, float(arch_scalar.get(
            "fp8_flop_frac", 1.0 if arch_scalar["fp8"] else 0.0))))
        cols["tp"].append(np.full(n, float(tp)))
        cols["rate"].append(np.full(n, float(data["run_rate"][rid])))
        cols["run_id"].append(np.full(n, rid, dtype=np.int32))
        model = str(data["run_model"][rid])
        cols["model_idx"].append(np.full(n, list(ARCH).index(model), dtype=np.int32))
        cols["hw_idx"].append(np.full(
            n, HARDWARE_INDEX[str(data["run_hardware"][rid])], dtype=np.int32))
        cols["family_idx"].append(np.full(
            n, family_names.index(ARCH[model]["family"]), dtype=np.int32))
        cols["role_idx"].append(np.full(n, role_names.index(role), dtype=np.int32))
        counts[role] += 1

    out = {key: np.concatenate(values) for key, values in cols.items()}
    out["model_names"] = np.array(list(ARCH))
    out["model_arch_json"] = np.asarray(
        [json.dumps(ARCH[name], sort_keys=True) for name in ARCH])
    out["family_names"] = np.array(family_names)
    out["hw_names"] = np.array(["A100", "H100"])
    out["role_names"] = np.array(role_names)
    out["dt_s"] = np.asarray(float(args.dt))
    out["moe_routing_mode"] = np.asarray(args.moe_routing)
    np.savez_compressed(args.out, **out)
    summary = ", ".join(f"{role}={n}" for role, n in sorted(counts.items()))
    print(f"Simulated {sum(counts.values())} runs ({summary}) "
          f"-> {out['power'].size} bins -> {args.out}")


if __name__ == "__main__":
    main()
