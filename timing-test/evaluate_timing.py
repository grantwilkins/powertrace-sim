"""Score the timing model on the frozen split matrix (DESIGN.md sections 3-6).

For every non-training run, replay its recorded arrivals through the
scheduler simulation with the frozen per-hardware efficiencies and compare
per-request time to first token, decode duration, and end-to-end latency
against the measured values. The per-config log-fit baseline (median rates
from model/throughput_database.json) is scored on the same requests; it has
per-model measured medians — an advantage the principled model refuses —
and it represents what per-deployment calibration alone gives you.

Usage: uv run python timing-test/evaluate_timing.py \
           [--out-dir results/timing_test_v1] [--roles all|quick]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iteration_time import launch_overhead_s, transformer_bw_scale  # noqa: E402
from scheduler_sim import EngineConfig, simulate_requests  # noqa: E402
from model.training_data.arch import get_arch  # noqa: E402
from model.training_data.moe_routing import load_routing_laws  # noqa: E402

BASE = Path(__file__).resolve().parent
EVAL_ROLES = (
    "test_indomain", "holdout_model", "holdout_twin", "holdout_rate",
    "heldout_model", "heldout_tp", "heldout_rate",
)
# Recorded serving flags (profiling/server/serve-*.sh), not fitted values.
MAX_NUM_SEQS = {"llama-3-405b": 64}


def _baseline(db, config_id, n_in, n_out):
    entry = db.get(config_id)
    if entry is None:
        return None
    prefill = float(entry.get("prefill_rate_median_toks_per_s")
                    or entry.get("lambda_prefill") or 0.0)
    decode = float(entry.get("decode_rate_median_toks_per_s") or 0.0)
    if prefill <= 0 or decode <= 0:
        return None
    ttft = n_in / prefill
    dec = np.maximum(n_out - 1, 0) / decode
    return {"ttft_s": ttft, "decode_duration_s": dec, "e2e_s": ttft + dec}


def evaluate_run(data, rid, fitted, db, routing_laws=None):
    hardware = str(data["run_hardware"][rid])
    model = str(data["run_model"][rid])
    tp = int(data["run_tp"][rid])
    arch = dict(get_arch(model))
    routing_law = (routing_laws or {}).get(model)
    params = fitted[hardware]
    t_launch = launch_overhead_s(
        arch, base_s=params["base_overhead_s"],
        per_message_s=params["per_message_s"][str(tp)])
    idx = np.flatnonzero(data["req_run_id"] == rid)
    order = idx[np.argsort(data["arrival_time_s"][idx], kind="stable")]
    requests = [(float(data["arrival_time_s"][i]),
                 int(data["input_tokens"][i]),
                 int(data["output_tokens"][i])) for i in order]
    engine = EngineConfig(max_num_seqs=MAX_NUM_SEQS.get(model, 256))
    simulated = simulate_requests(
        requests, arch=arch, hardware=hardware, tp=tp,
        eff_flops=params["eff_flops"], eff_bw=params["eff_bw"],
        t_launch_s=t_launch, t_sample_s=params.get("per_token_sample_s", 0.0),
        transformer_bw_scale=transformer_bw_scale(arch, params, hardware),
        engine=engine, routing_law=routing_law)
    t_first = params["first_token_overhead_s"]
    rows = []
    config_id = f"{model}_{hardware}_tp{tp}"
    for sim, i in zip(simulated, order):
        measured = {"ttft_s": float(data["ttft_s"][i]),
                    "decode_duration_s": float(data["decode_duration_s"][i])}
        measured["e2e_s"] = measured["ttft_s"] + measured["decode_duration_s"]
        predicted = {"ttft_s": sim["ttft_s"] + t_first,
                     "decode_duration_s": sim["decode_duration_s"],
                     "e2e_s": sim["e2e_s"] + t_first}
        base = _baseline(db, config_id, int(data["input_tokens"][i]),
                         int(data["output_tokens"][i]))
        row = {"n_in": int(data["input_tokens"][i]),
               "n_out": int(data["output_tokens"][i])}
        for phase in ("ttft_s", "decode_duration_s", "e2e_s"):
            row[f"meas_{phase}"] = measured[phase]
            row[f"pred_{phase}"] = predicted[phase]
            row[f"base_{phase}"] = base[phase] if base else float("nan")
        rows.append(row)
    return rows


def _cell_summary(rows, which):
    out = {}
    for phase in ("ttft_s", "decode_duration_s", "e2e_s"):
        meas = np.asarray([r[f"meas_{phase}"] for r in rows])
        pred = np.asarray([r[f"{which}_{phase}"] for r in rows])
        err = pred - meas
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = 100.0 * err / np.where(meas > 0, meas, np.nan)
        prefix = phase.replace("_s", "")
        out[f"{prefix}_medabs_s"] = float(np.nanmedian(np.abs(err)))
        out[f"{prefix}_p90abs_s"] = float(np.nanpercentile(np.abs(err), 90))
        out[f"{prefix}_rmse_s"] = float(np.sqrt(np.nanmean(err ** 2)))
        out[f"{prefix}_medabs_pct"] = float(np.nanmedian(np.abs(pct)))
        out[f"{prefix}_med_signed_pct"] = float(np.nanmedian(pct))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/timing_test_v1")
    parser.add_argument("--roles", default="all", choices=("all", "quick"))
    parser.add_argument("--manifest", default="split_manifest.json",
                        help="manifest file under timing-test/ (the fp8 "
                             "amendment uses split_manifest_fp8.json)")
    parser.add_argument(
        "--fitted", default=str(BASE / "fitted_efficiencies.json")
    )
    parser.add_argument(
        "--moe-routing", choices=("uniform", "measured"), default="uniform")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = dict(np.load(BASE / "timing_dataset.npz", allow_pickle=False))
    manifest = json.loads((BASE / args.manifest).read_text())
    fitted = json.loads(Path(args.fitted).read_text())
    routing_laws = load_routing_laws() if args.moe_routing == "measured" else {}
    db_raw = json.loads(
        (BASE.parent / "model" / "throughput_database.json").read_text())
    db = db_raw.get("configs", db_raw)
    roles = {int(k): v for k, v in manifest["roles"].items()}

    cells = defaultdict(list)
    started = time.time()
    run_ids = [rid for rid, role in sorted(roles.items()) if role in EVAL_ROLES]
    if args.roles == "quick":
        run_ids = run_ids[::10]
    for n, rid in enumerate(run_ids):
        rows = evaluate_run(data, rid, fitted, db, routing_laws)
        key = (roles[rid], str(data["run_hardware"][rid]),
               str(data["run_model"][rid]), int(data["run_tp"][rid]),
               float(data["run_rate"][rid]))
        cells[key].extend(rows)
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(run_ids)} runs, {time.time() - started:.0f}s",
                  flush=True)

    summary = []
    for (role, hardware, model, tp, rate), rows in sorted(cells.items()):
        record = {"role": role, "hardware": hardware, "model": model,
                  "tp": tp, "rate": rate, "requests": len(rows)}
        record.update({f"model_{k}": v for k, v in
                       _cell_summary(rows, "pred").items()})
        record.update({f"baseline_{k}": v for k, v in
                       _cell_summary(rows, "base").items()})
        summary.append(record)
    with (out / "cell_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out / "fitted_efficiencies.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_point"}
                    if isinstance(v, dict) else v
                    for k, v in fitted.items()}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}/cell_metrics.csv with {len(summary)} cells")


if __name__ == "__main__":
    main()
