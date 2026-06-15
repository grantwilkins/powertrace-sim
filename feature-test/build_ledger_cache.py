"""Build a cached per-bin work ledger from all matched profiling runs.

Parses power CSVs + benchmark JSONs once and stores per-second bins with raw
physical quantities (token rates, batch, iteration rate, traffic terms) plus
run metadata, so model-fitting iterations are fast.

Run from repo root:
    uv run python feature-test/build_ledger_cache.py --max-per-cell 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.pipeline.artifact_resolution import resolve_throughput  # noqa: E402
from model.training_data.power_parsing import parse_power_csv, parse_request_json  # noqa: E402
from model.utils.io import load_json, resolve_existing_path  # noqa: E402

GIB = 1024.0**3

# Architecture descriptors (as served; see profiling/server/*.sh for dtypes).
ARCH = {
    "llama-3-8b": dict(
        family="dense-8b", n_active=8.03e9, w_bytes=2 * 8.03e9, d_model=4096,
        n_layers=32, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
    ),
    "deepseek-r1-distill-8b": dict(
        family="dense-8b", n_active=8.03e9, w_bytes=2 * 8.03e9, d_model=4096,
        n_layers=32, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
    ),
    "llama-3-70b": dict(
        family="dense-70b", n_active=70.55e9, w_bytes=2 * 70.55e9, d_model=8192,
        n_layers=80, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
    ),
    "deepseek-r1-distill-70b": dict(
        family="dense-70b", n_active=70.55e9, w_bytes=2 * 70.55e9, d_model=8192,
        n_layers=80, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=0,
    ),
    "llama-3-405b": dict(
        family="dense-405b", n_active=405.85e9, w_bytes=1 * 405.85e9, d_model=16384,
        n_layers=126, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1,
        swa_window=0, fp8=1,
    ),
    "gpt-oss-120b": dict(
        family="moe-120b", n_active=5.1e9, w_bytes=60.8 * GIB, d_model=2880,
        n_layers=36, n_kv=8, head_dim=64, moe_frac=0.9, n_experts=128, top_k=4,
        swa_window=128, fp8=0,
    ),
    "gpt-oss-20b": dict(
        family="moe-20b", n_active=3.6e9, w_bytes=12.8 * GIB, d_model=2880,
        n_layers=24, n_kv=8, head_dim=64, moe_frac=0.9, n_experts=32, top_k=4,
        swa_window=128, fp8=0,
    ),
}

KV_ELEM_BYTES = 2.0


def build_run_bins(json_path, csv_path, model_name, tp, lambda_prefill, dt=1.0, trim_s=5.0):
    arch = ARCH[model_name]
    req = parse_request_json(json_path)
    pw = parse_power_csv(csv_path, tensor_parallelism=tp)
    if req is None or pw is None or not req["has_timestamps"]:
        return None

    p_ts, p_w = pw["timestamps"], pw["power"]
    t0 = float(p_ts[0])
    arr = req["request_timestamps"] - t0
    arr = arr - round(float(np.min(arr)) / 1800.0) * 1800.0
    if float(np.min(arr)) < -2.0 or float(np.min(arr)) > 600.0:
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
    edges = np.arange(run_start, run_end, dt)
    if edges.size < 11:
        return None
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

    pre_tok = np.zeros(nb)
    dec_tok = np.zeros(nb)
    batch = np.zeros(nb)
    pre_active = np.zeros(nb)  # concurrent prefills
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
        kv_read[sl] += dec_rate[j] * (ov_dec / dt) * ctx_eff * kv_tok

    dec_iter = dec_tok / np.maximum(batch, 1e-9)
    iters = dec_iter + pre_iter
    if arch["moe_frac"] > 0:
        # Decode: experts touched grows with batch. Prefill: a chunk carries
        # thousands of tokens, so effectively all experts are read.
        touched = np.minimum(1.0, batch * arch["top_k"] / arch["n_experts"])
        w_eff_dec = arch["w_bytes"] * ((1 - arch["moe_frac"]) + arch["moe_frac"] * touched)
    else:
        w_eff_dec = np.full(nb, arch["w_bytes"])
    w_read_dec = w_eff_dec * dec_iter
    w_read_pre = arch["w_bytes"] * pre_iter
    w_read = w_read_dec + w_read_pre
    kv_write = (pre_tok + dec_tok) * kv_tok

    # TP all-reduce traffic: 2 all-reduces/layer, each moves 2*(TP-1)/TP of the
    # activation (d_model * 2 bytes) per token through NVLink.
    tok_rate = pre_tok + dec_tok
    comm = (
        tok_rate * arch["n_layers"] * 2.0
        * 2.0 * arch["d_model"] * 2.0 * (tp - 1.0) / max(tp, 1)
    )

    keep = valid & np.isfinite(power)
    n = int(keep.sum())
    if n == 0:
        return None
    arch_scalar = {k: float(v) for k, v in arch.items() if k != "family"}
    return dict(
        power=power[keep], pre_tok=pre_tok[keep], dec_tok=dec_tok[keep],
        batch=batch[keep], pre_active=pre_active[keep], iters=iters[keep],
        w_read=w_read[keep], w_read_pre=w_read_pre[keep], w_read_dec=w_read_dec[keep],
        kv_read=kv_read[keep], kv_write=kv_write[keep], comm=comm[keep],
        n=n, arch=arch_scalar,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    ap.add_argument("--max-per-cell", type=int, default=3)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--out", default="feature-test/ledger_cache.npz")
    args = ap.parse_args()

    throughput_db = load_json("model/throughput_database.json")
    base = Path(args.pair_manifest_csv).resolve().parent
    cell_count = defaultdict(int)
    runs = []
    with open(args.pair_manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "").strip() != "matched":
                continue
            model = row["model_name"].strip()
            if model not in ARCH:
                continue
            cell = (model, row["hardware"].strip(), row["tensor_parallelism"].strip(), row["rate"].strip())
            if cell_count[cell] >= args.max_per_cell:
                continue
            jp = resolve_existing_path(row["json_path"].strip(), str(base))
            cp = resolve_existing_path(row["power_csv_path"].strip(), str(base))
            if jp is None or cp is None:
                continue
            cell_count[cell] += 1
            runs.append(dict(model=model, hardware=row["hardware"].strip(),
                             tp=int(row["tensor_parallelism"]), rate=float(row["rate"]),
                             json_path=jp, csv_path=cp))
    print(f"Selected {len(runs)} runs")

    cols = defaultdict(list)
    n_fail = 0
    lam_cache = {}
    for i, r in enumerate(runs):
        cfg_id = f"{r['model']}_{r['hardware']}_tp{r['tp']}"
        if cfg_id not in lam_cache:
            try:
                th = resolve_throughput(throughput_db, cfg_id)
                lam_cache[cfg_id] = float(
                    th.get("lambda_prefill", th.get("prefill_rate_median_toks_per_s"))
                )
            except Exception:
                lam_cache[cfg_id] = 5000.0
        b = build_run_bins(r["json_path"], r["csv_path"], r["model"], r["tp"],
                           lam_cache[cfg_id], dt=args.dt)
        if b is None:
            n_fail += 1
            continue
        n = b["n"]
        for key in ("power", "pre_tok", "dec_tok", "batch", "pre_active", "iters",
                    "w_read", "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm"):
            cols[key].append(b[key])
        a = b["arch"]
        cols["n_active"].append(np.full(n, a["n_active"]))
        cols["w_bytes"].append(np.full(n, a["w_bytes"]))
        cols["fp8"].append(np.full(n, a["fp8"]))
        cols["tp"].append(np.full(n, float(r["tp"])))
        cols["rate"].append(np.full(n, r["rate"]))
        cols["run_id"].append(np.full(n, i, dtype=np.int32))
        cols["model_idx"].append(np.full(n, list(ARCH).index(r["model"]), dtype=np.int32))
        cols["hw_idx"].append(np.full(n, 0 if r["hardware"] == "A100" else 1, dtype=np.int32))
        fam = ARCH[r["model"]]["family"]
        fams = sorted({v["family"] for v in ARCH.values()})
        cols["family_idx"].append(np.full(n, fams.index(fam), dtype=np.int32))
        if (i + 1) % 50 == 0:
            print(f"  parsed {i+1}/{len(runs)} ({n_fail} failed)")

    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["model_names"] = np.array(list(ARCH))
    out["family_names"] = np.array(sorted({v["family"] for v in ARCH.values()}))
    out["hw_names"] = np.array(["A100", "H100"])
    np.savez_compressed(args.out, **out)
    total = out["power"].size
    print(f"Parsed {len(runs)-n_fail}/{len(runs)} runs OK -> {total} bins -> {args.out}")


if __name__ == "__main__":
    main()
