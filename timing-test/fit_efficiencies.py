"""Fit per-hardware roofline efficiencies and per-GPU-count iteration overhead.

Frozen procedure (timing-test/DESIGN.md section 7): least squares on log
ratios over (a) probe calibration rows (direct iteration-time observations)
and (b) solo requests from TRAINING runs only — requests whose lifetime
overlaps no other request, so their inter-token latency samples the batch-1
iteration time and their time to first token samples queue-free prefill.
Holdout roles never contribute a point. No per-model constants: efficiencies
are shared across every model on a hardware; architecture enters only
through the work calculator.

Output: timing-test/fitted_efficiencies.json
  {hardware: {eff_flops, eff_bw, t_launch_s: {tp: value}, points, rmse_log}}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iteration_time import iteration_time_s, iteration_work, launch_overhead_s  # noqa: E402
from model.training_data.arch import ARCH, get_arch  # noqa: E402

BASE = Path(__file__).resolve().parent
CHUNK_BUDGET_TOKENS = 2048
EFF_BOUNDS = (0.05, 1.0)
BASE_OVERHEAD_BOUNDS_S = (0.0, 0.05)
PER_MESSAGE_BOUNDS_S = (0.0, 0.002)
FIRST_TOKEN_OVERHEAD_BOUNDS_S = (0.0, 0.2)
PER_TOKEN_SAMPLE_BOUNDS_S = (0.0, 0.0005)


def chunked_prefill_time_s(arch, n_in, *, hardware, tp, eff_flops, eff_bw,
                           t_launch_s, chunk_budget=CHUNK_BUDGET_TOKENS,
                           transformer_bw_scale=1.0):
    """Queue-free prompt processing time: one iteration per prefill chunk."""
    total = 0.0
    done = 0
    while done < n_in:
        chunk = min(chunk_budget, n_in - done)
        work = iteration_work(
            arch, prefill_chunk=chunk, prefill_context=done,
            prefill_logits=float(done + chunk == n_in),
        )
        total += iteration_time_s(work, hardware=hardware, tp=tp,
                                  eff_flops=eff_flops, eff_bw=eff_bw,
                                  t_launch_s=t_launch_s,
                                  transformer_bw_scale=transformer_bw_scale)
        done += chunk
    return total


def _decode_time_s(arch, batch, context, *, hardware, tp, eff_flops, eff_bw,
                   t_launch_s, t_sample_s=0.0, transformer_bw_scale=1.0):
    work = iteration_work(arch, decode_batch=batch, context_mean=context)
    return iteration_time_s(work, hardware=hardware, tp=tp,
                            eff_flops=eff_flops, eff_bw=eff_bw,
                            t_launch_s=t_launch_s, t_sample_s=t_sample_s,
                            transformer_bw_scale=transformer_bw_scale)


def probe_points(calibration) -> list[dict]:
    """One fitting point per probe level (llama-3-70b probes only exist)."""
    points = []
    for row in calibration["rows"]:
        counters = row.get("engine_counters") or {}
        if counters.get("preemptions", 0.0) > 0.0:
            continue
        measured = row["measured"]
        model = row.get("model", "llama-3-70b")
        if model == "gpt-oss-20b":
            # Engine-configuration mismatch: the legacy gpt-oss serving runs
            # used --async-scheduling (their serve script adds it) while the
            # iteration probes ran without it, so absolute probe latencies
            # (6.9 ms vs 3.9 ms at batch 1) describe a different engine mode
            # than the runs this model predicts. The llama probes match
            # their legacy configuration (29.35 vs 28.93 ms) and stay. The
            # gpt-oss probes are used only for configuration-independent
            # batch-shape deltas (fit_moe_routing.py).
            continue
        common = {"hardware": row["hardware"], "tp": int(row["tp"]),
                  "model": model,
                  "source": f"probe:{model}:{row['probe']}:{row['label']}"}
        if row["probe"] == "prefill_staircase":
            points.append({**common, "kind": "prefill",
                           "n_in": int(row["prompt_tokens"]),
                           "seconds": measured["median_ttft_ms"] / 1e3})
        else:
            batch = float(row.get("effective_decode_batch") or row["batch"])
            points.append({**common, "kind": "decode", "batch": batch,
                           "context": float(row["context_tokens_mean"]),
                           "seconds": measured["median_itl_ms"] / 1e3})
    return points


def solo_request_points(data, roles) -> list[dict]:
    """Batch-1 decode and queue-free prefill points from training runs."""
    run_ids = np.asarray(
        [rid for rid, role in roles.items() if role == "train"], dtype=int)
    points = []
    per_config = defaultdict(lambda: {"itl": [], "ttft": [], "n_in": [],
                                      "context": []})
    offsets = data["itl_offsets"]
    for rid in run_ids:
        mask = data["req_run_id"] == rid
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        arrival = data["arrival_time_s"][idx]
        finish = arrival + data["ttft_s"][idx] + data["decode_duration_s"][idx]
        order = np.argsort(arrival)
        arrival, finish, idx = arrival[order], finish[order], idx[order]
        # Solo: starts after every earlier request finished and ends before
        # the next arrival.
        prev_max_finish = np.r_[-np.inf, np.maximum.accumulate(finish)[:-1]]
        next_arrival = np.r_[arrival[1:], np.inf]
        solo = (arrival >= prev_max_finish) & (finish <= next_arrival)
        key = (str(data["run_hardware"][rid]), int(data["run_tp"][rid]),
               str(data["run_model"][rid]))
        for i in idx[solo]:
            n_out = int(data["output_tokens"][i])
            if n_out < 8:
                continue  # too few intervals for a stable median
            itls = data["itl_values"][offsets[i]:offsets[i + 1]]
            per_config[key]["itl"].append(float(np.median(itls)))
            per_config[key]["ttft"].append(float(data["ttft_s"][i]))
            per_config[key]["n_in"].append(int(data["input_tokens"][i]))
            per_config[key]["context"].append(
                int(data["input_tokens"][i]) + n_out / 2.0)
    for (hardware, tp, model), values in sorted(per_config.items()):
        if len(values["itl"]) < 5:
            continue
        points.append({"hardware": hardware, "tp": tp, "model": model,
                       "kind": "decode", "batch": 1.0,
                       "context": float(np.median(values["context"])),
                       "seconds": float(np.median(values["itl"])),
                       "source": f"solo:{model}:tp{tp}:n{len(values['itl'])}"})
        points.append({"hardware": hardware, "tp": tp, "model": model,
                       "kind": "prefill",
                       "n_in": int(np.median(values["n_in"])),
                       "seconds": float(np.median(values["ttft"])),
                       "source": f"solo_ttft:{model}:tp{tp}"})
    return points


LOADED_BATCH_BINS = (1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
                     192, 256)


def loaded_request_points(data, roles) -> list[dict]:
    """Inter-token-latency points at reconstructed concurrency from loaded
    TRAINING runs.

    Each request's decode window is [arrival+ttft, arrival+ttft+decode];
    its mean concurrency is the average number of simultaneously decoding
    requests over that window (interval-overlap integral, self included).
    Grouped per configuration and concurrency bin, these give dense
    (batch, context) -> inter-token-latency coverage across every training
    model and GPU count — the same signal a per-operator profiler would
    collect, recovered from the serving logs.
    """
    run_ids = np.asarray(
        [rid for rid, role in roles.items() if role == "train"], dtype=int)
    groups = defaultdict(lambda: {"itl": [], "batch": [], "context": []})
    offsets = data["itl_offsets"]
    for rid in run_ids:
        idx = np.flatnonzero(data["req_run_id"] == rid)
        if idx.size == 0:
            continue
        start = data["arrival_time_s"][idx] + data["ttft_s"][idx]
        end = start + data["decode_duration_s"][idx]
        keep = data["output_tokens"][idx] >= 16
        overlap = (np.minimum(end[keep, None], end[None, :])
                   - np.maximum(start[keep, None], start[None, :]))
        concurrency = 1.0 + (
            (np.clip(overlap, 0.0, None).sum(axis=1)
             - (end[keep] - start[keep])) / np.maximum(end[keep] - start[keep], 1e-9))
        key = (str(data["run_hardware"][rid]), int(data["run_tp"][rid]),
               str(data["run_model"][rid]))
        for j, i in enumerate(idx[keep]):
            itls = data["itl_values"][offsets[i]:offsets[i + 1]]
            bin_index = int(np.argmin(
                np.abs(np.log(np.asarray(LOADED_BATCH_BINS))
                       - np.log(max(concurrency[j], 1.0)))))
            groups[key + (bin_index,)]["itl"].append(float(np.median(itls)))
            groups[key + (bin_index,)]["batch"].append(float(concurrency[j]))
            groups[key + (bin_index,)]["context"].append(
                float(data["input_tokens"][i]) + float(data["output_tokens"][i]) / 2.0)
    points = []
    for (hardware, tp, model, bin_index), values in sorted(groups.items()):
        if len(values["itl"]) < 10:
            continue
        points.append({"hardware": hardware, "tp": tp, "model": model,
                       "kind": "decode",
                       "batch": float(np.median(values["batch"])),
                       "context": float(np.median(values["context"])),
                       "seconds": float(np.median(values["itl"])),
                       "source": (f"loaded:{model}:tp{tp}:"
                                  f"B~{LOADED_BATCH_BINS[bin_index]}:"
                                  f"n{len(values['itl'])}")})
    return points


def fit_hardware(points, hardware) -> dict:
    rows = [p for p in points if p["hardware"] == hardware]
    tps = sorted({p["tp"] for p in rows})
    arch_by_model = {name: get_arch(name) for name in ARCH}

    def predict(theta, point):
        eff_flops, eff_bw, base_s = theta[0], theta[1], theta[2]
        per_message_s = theta[3 + tps.index(point["tp"])]
        t_first_s = theta[3 + len(tps)]
        t_sample_s = theta[4 + len(tps)]
        arch = arch_by_model[point["model"]]
        t_launch = launch_overhead_s(arch, base_s=base_s,
                                     per_message_s=per_message_s)
        if point["kind"] == "decode":
            return _decode_time_s(arch, point["batch"], point["context"],
                                  hardware=hardware, tp=point["tp"],
                                  eff_flops=eff_flops, eff_bw=eff_bw,
                                  t_launch_s=t_launch, t_sample_s=t_sample_s)
        return t_first_s + chunked_prefill_time_s(
            arch, point["n_in"], hardware=hardware, tp=point["tp"],
            eff_flops=eff_flops, eff_bw=eff_bw, t_launch_s=t_launch)

    def residuals(theta):
        return np.asarray(
            [np.log(predict(theta, p)) - np.log(p["seconds"]) for p in rows])

    x0 = np.asarray([0.5, 0.6, 0.002] + [0.0001] * len(tps) + [0.02, 3e-5])
    lower = ([EFF_BOUNDS[0]] * 2 + [BASE_OVERHEAD_BOUNDS_S[0]]
             + [PER_MESSAGE_BOUNDS_S[0]] * len(tps)
             + [FIRST_TOKEN_OVERHEAD_BOUNDS_S[0], PER_TOKEN_SAMPLE_BOUNDS_S[0]])
    upper = ([EFF_BOUNDS[1]] * 2 + [BASE_OVERHEAD_BOUNDS_S[1]]
             + [PER_MESSAGE_BOUNDS_S[1]] * len(tps)
             + [FIRST_TOKEN_OVERHEAD_BOUNDS_S[1], PER_TOKEN_SAMPLE_BOUNDS_S[1]])
    result = least_squares(residuals, x0, bounds=(lower, upper), xtol=1e-12)
    fitted = {"eff_flops": float(result.x[0]), "eff_bw": float(result.x[1]),
              "base_overhead_s": float(result.x[2]),
              "per_message_s": {str(tp): float(result.x[3 + i])
                                for i, tp in enumerate(tps)},
              "first_token_overhead_s": float(result.x[3 + len(tps)]),
              "per_token_sample_s": float(result.x[4 + len(tps)]),
              "points": len(rows),
              "rmse_log": float(np.sqrt(np.mean(result.fun ** 2))),
              "per_point": [{"source": p["source"], "tp": p["tp"],
                             "kind": p["kind"], "measured_s": p["seconds"],
                             "predicted_s": float(predict(result.x, p))}
                            for p in rows]}
    return fitted


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BASE / "fitted_efficiencies.json"))
    parser.add_argument("--dataset", default=str(BASE / "timing_dataset.npz"))
    parser.add_argument("--split-manifest", default=str(BASE / "split_manifest.json"))
    parser.add_argument("--calibration", default=str(BASE / "probe_calibration.json"))
    args = parser.parse_args(argv)
    calibration = json.loads(Path(args.calibration).read_text())
    data = dict(np.load(args.dataset, allow_pickle=False))
    manifest = json.loads(Path(args.split_manifest).read_text())
    roles = {int(k): v for k, v in manifest["roles"].items()}
    points = (probe_points(calibration) + solo_request_points(data, roles)
              + loaded_request_points(data, roles))
    held_pairs = {(hardware, model)
                  for source in (manifest["holdout_model"], manifest["holdout_twin"])
                  for hardware, model in source.items()}
    assert not any((p["hardware"], p["model"]) in held_pairs for p in points), \
        "holdout model leaked into fitting points"
    output = {"procedure": "timing-test/DESIGN.md section 7",
              "chunk_budget_tokens": CHUNK_BUDGET_TOKENS}
    for hardware in ("A100", "H100"):
        output[hardware] = fit_hardware(points, hardware)
        summary = {k: output[hardware][k] for k in
                   ("eff_flops", "eff_bw", "base_overhead_s", "per_message_s",
                    "first_token_overhead_s", "per_token_sample_s",
                    "points", "rmse_log")}
        print(hardware, json.dumps(summary))
    Path(args.out).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
