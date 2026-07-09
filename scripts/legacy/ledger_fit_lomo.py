"""Leave-one-model-out roofline-ledger fit on measured power traces.

Tests whether node power is explained by hardware-level energy coefficients
applied to model-independent work rates (FLOP/s, DRAM bytes/s) computed from
per-request timing (arrival, TTFT, decode time) and architecture descriptors.

For each hardware platform, fits
    P_node(t) = TP * p_idle + e_f * FLOPs/s + e_w * weight_bytes/s + e_kv * kv_bytes/s
with non-negative least squares, then evaluates leave-one-architecture-out:
fit on all other model families, predict the held-out family's power bins.

Run:
    uv run -m scripts.eval.ledger_fit_lomo
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

from model.training_data.power_parsing import parse_power_csv, parse_request_json
from model.utils.io import resolve_existing_path

GIB = 1024.0**3

# Architecture descriptors. w_bytes = weights as served (dtype-aware).
# kv_bytes_tok = 2(K+V) * n_layers * n_kv_heads * head_dim * bytes_per_elem.
# MoE: weight traffic per iteration scales with experts touched ~ min(1, B*top_k/E).
ARCH: Dict[str, Dict[str, float]] = {
    "llama-3-8b": dict(
        family="dense-8b", n_active=8.03e9, w_bytes=2 * 8.03e9,
        n_layers=32, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1, swa_window=0,
    ),
    "deepseek-r1-distill-8b": dict(
        family="dense-8b", n_active=8.03e9, w_bytes=2 * 8.03e9,
        n_layers=32, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1, swa_window=0,
    ),
    "llama-3-70b": dict(
        family="dense-70b", n_active=70.55e9, w_bytes=2 * 70.55e9,
        n_layers=80, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1, swa_window=0,
    ),
    "deepseek-r1-distill-70b": dict(
        family="dense-70b", n_active=70.55e9, w_bytes=2 * 70.55e9,
        n_layers=80, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1, swa_window=0,
    ),
    "llama-3-405b": dict(  # served FP8
        family="dense-405b", n_active=405.85e9, w_bytes=1 * 405.85e9,
        n_layers=126, n_kv=8, head_dim=128, moe_frac=0.0, n_experts=1, top_k=1, swa_window=0,
    ),
    "gpt-oss-120b": dict(  # MXFP4 MoE, ~60.8 GiB checkpoint, 5.1B active
        family="moe-120b", n_active=5.1e9, w_bytes=60.8 * GIB,
        n_layers=36, n_kv=8, head_dim=64, moe_frac=0.9, n_experts=128, top_k=4, swa_window=128,
    ),
    "gpt-oss-20b": dict(  # MXFP4 MoE, ~12.8 GiB checkpoint, 3.6B active
        family="moe-20b", n_active=3.6e9, w_bytes=12.8 * GIB,
        n_layers=24, n_kv=8, head_dim=64, moe_frac=0.9, n_experts=32, top_k=4, swa_window=128,
    ),
}

KV_ELEM_BYTES = 2.0  # bf16 KV cache


def _fold_offset(raw_offset_s: float) -> float:
    """Remove timezone-shaped offset (multiple of 30 min) from clock difference."""
    return raw_offset_s - round(raw_offset_s / 1800.0) * 1800.0


def build_run_bins(
    json_path: str,
    csv_path: str,
    model_name: str,
    tp: int,
    dt: float = 1.0,
    trim_s: float = 5.0,
) -> Optional[Dict[str, np.ndarray]]:
    arch = ARCH[model_name]
    req = parse_request_json(json_path)
    pw = parse_power_csv(csv_path, tensor_parallelism=tp)
    if req is None or pw is None or not req["has_timestamps"]:
        return None

    p_ts, p_w = pw["timestamps"], pw["power"]
    t0 = float(p_ts[0])

    arr = req["request_timestamps"] - t0
    arr = arr - round(float(np.min(arr)) / 1800.0) * 1800.0  # fold timezone offset
    if float(np.min(arr)) < -2.0 or float(np.min(arr)) > 600.0:
        return None  # alignment failed
    ttft, dec = req["ttfts"], req["decode_times"]
    n_in, n_out = req["input_lens"], req["output_lens"]

    pre_s, pre_e = arr, arr + ttft
    dec_s, dec_e = pre_e, pre_e + dec
    run_end = min(float(p_ts[-1] - t0), float(np.max(dec_e)))
    run_start = max(trim_s, float(np.min(arr)))
    if run_end - run_start < 10 * dt:
        return None

    edges = np.arange(run_start, run_end, dt)
    if edges.size < 11:
        return None
    nb = edges.size - 1

    # Bin power by mean of samples in each bin
    p_rel = p_ts - t0
    idx = np.searchsorted(edges, p_rel) - 1
    ok = (idx >= 0) & (idx < nb) & np.isfinite(p_w)
    pow_sum = np.bincount(idx[ok], weights=p_w[ok], minlength=nb)
    pow_cnt = np.bincount(idx[ok], minlength=nb)
    valid = pow_cnt > 0
    power = np.where(valid, pow_sum / np.maximum(pow_cnt, 1), np.nan)

    pre_rate = n_in / np.maximum(ttft, 1e-3)   # prefill tokens/s while prefilling
    dec_rate = n_out / np.maximum(dec, 1e-3)   # decode tokens/s while decoding
    kv_tok = 2.0 * arch["n_layers"] * arch["n_kv"] * arch["head_dim"] * KV_ELEM_BYTES
    swa = float(arch["swa_window"])

    pre_tok = np.zeros(nb)
    dec_tok = np.zeros(nb)
    batch = np.zeros(nb)
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
        pre_iter[sl] += ov_pre / max(float(ttft[j]), 1e-3) / dt
        # context length at bin midpoint; KV read traffic = rate * ctx * kv_tok
        mid = (b_lo + b_hi) / 2.0
        prog = np.clip((mid - dec_s[j]) / max(float(dec[j]), 1e-3), 0.0, 1.0)
        ctx = n_in[j] + prog * n_out[j]
        ctx_eff = ctx if swa <= 0 else 0.5 * ctx + 0.5 * np.minimum(ctx, swa)
        kv_read[sl] += dec_rate[j] * (ov_dec / dt) * ctx_eff * kv_tok

    flops = 2.0 * arch["n_active"] * (pre_tok + dec_tok)
    iters = dec_tok / np.maximum(batch, 1e-9) + pre_iter
    if arch["moe_frac"] > 0:
        eff_b = np.maximum(batch, (pre_tok > 0).astype(float))
        touched = np.minimum(1.0, eff_b * arch["top_k"] / arch["n_experts"])
        w_eff = arch["w_bytes"] * ((1 - arch["moe_frac"]) + arch["moe_frac"] * touched)
    else:
        w_eff = np.full(nb, arch["w_bytes"])
    w_read = w_eff * iters

    keep = valid & np.isfinite(power)
    return {
        "power": power[keep],
        "tp": np.full(int(keep.sum()), float(tp)),
        "flops": flops[keep],
        "w_read": w_read[keep],
        "kv_read": kv_read[keep],
        "batch": batch[keep],
    }


def load_manifest_runs(pair_manifest_csv: str, max_per_cell: int) -> List[Dict[str, str]]:
    base = Path(pair_manifest_csv).resolve().parent
    cell_count: Dict[tuple, int] = defaultdict(int)
    runs: List[Dict[str, str]] = []
    with open(pair_manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "").strip() != "matched":
                continue
            model = row["model_name"].strip()
            if model not in ARCH:
                continue
            cell = (model, row["hardware"].strip(), row["tensor_parallelism"].strip(), row["rate"].strip())
            if cell_count[cell] >= max_per_cell:
                continue
            jp = resolve_existing_path(row["json_path"].strip(), str(base))
            cp = resolve_existing_path(row["power_csv_path"].strip(), str(base))
            if jp is None or cp is None:
                continue
            cell_count[cell] += 1
            runs.append(
                dict(model=model, hardware=row["hardware"].strip(),
                     tp=row["tensor_parallelism"].strip(), rate=row["rate"].strip(),
                     json_path=str(jp), csv_path=str(cp))
            )
    return runs


def fit_nnls(X: np.ndarray, y: np.ndarray):
    coef, _ = nnls(X, y)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    mape = float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0))) * 100
    return coef, r2, mape


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair-manifest-csv", default="results/stage0/pair_manifest.csv")
    ap.add_argument("--max-per-cell", type=int, default=2, help="runs per (model,hw,tp,rate)")
    ap.add_argument("--out-dir", default="results/ledger_fit")
    args = ap.parse_args()

    runs = load_manifest_runs(args.pair_manifest_csv, args.max_per_cell)
    print(f"Selected {len(runs)} runs")

    # hardware -> family -> list of per-run dicts (with metadata)
    data: Dict[str, List[Dict]] = defaultdict(list)
    n_fail = 0
    for i, r in enumerate(runs):
        bins = build_run_bins(r["json_path"], r["csv_path"], r["model"], int(r["tp"]))
        if bins is None:
            n_fail += 1
            continue
        bins["meta"] = r
        bins["family"] = ARCH[r["model"]]["family"]
        data[r["hardware"]].append(bins)
        if (i + 1) % 50 == 0:
            print(f"  parsed {i+1}/{len(runs)} ({n_fail} failed)")
    print(f"Parsed {len(runs) - n_fail}/{len(runs)} runs OK")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coeff_rows, run_rows = [], []

    def design(bin_list):
        X = np.column_stack([
            np.concatenate([b["tp"] for b in bin_list]),
            np.concatenate([b["flops"] for b in bin_list]),
            np.concatenate([b["w_read"] for b in bin_list]),
            np.concatenate([b["kv_read"] for b in bin_list]),
        ])
        y = np.concatenate([b["power"] for b in bin_list])
        return X, y

    for hw, bin_list in sorted(data.items()):
        fams = sorted({b["family"] for b in bin_list})
        X, y = design(bin_list)
        coef, r2, mape = fit_nnls(X, y)
        p_idle, e_f, e_w, e_kv = coef
        print(f"\n=== {hw}: pooled fit on {len(bin_list)} runs, {y.size} bins, families={fams}")
        print(f"  p_idle={p_idle:.1f} W/GPU  e_f={e_f*1e12:.3f} pJ/FLOP  "
              f"e_w={e_w*1e12:.1f} pJ/B  e_kv={e_kv*1e12:.1f} pJ/B  R2={r2:.3f} MAPE={mape:.1f}%")
        coeff_rows.append(dict(hardware=hw, scope="pooled", heldout="", n_bins=y.size,
                               p_idle_w=p_idle, e_f_pj=e_f * 1e12, e_w_pj=e_w * 1e12,
                               e_kv_pj=e_kv * 1e12, r2=r2, mape_pct=mape))

        if len(fams) < 2:
            continue
        for held in fams:
            train = [b for b in bin_list if b["family"] != held]
            test = [b for b in bin_list if b["family"] == held]
            tr_fams = {b["family"] for b in train}
            if len(tr_fams) < 2 or not test:
                continue
            Xtr, ytr = design(train)
            coef, r2_tr, _ = fit_nnls(Xtr, ytr)
            for b in test:
                Xte = np.column_stack([b["tp"], b["flops"], b["w_read"], b["kv_read"]])
                pred = Xte @ coef
                m = b["meta"]
                run_rows.append(dict(
                    hardware=hw, heldout_family=held, model=m["model"], tp=m["tp"],
                    rate=m["rate"], measured_mean_w=float(np.mean(b["power"])),
                    predicted_mean_w=float(np.mean(pred)),
                    bin_mape_pct=float(np.mean(np.abs(pred - b["power"]) / np.maximum(b["power"], 1.0))) * 100,
                ))
            sub = [r for r in run_rows if r["hardware"] == hw and r["heldout_family"] == held]
            errs = [abs(r["predicted_mean_w"] - r["measured_mean_w"]) / r["measured_mean_w"] * 100 for r in sub]
            print(f"  LOMO hold out {held:>11}: {len(sub):3d} runs, "
                  f"mean-power err median {np.median(errs):5.1f}%  p90 {np.percentile(errs, 90):5.1f}%")
            coeff_rows.append(dict(hardware=hw, scope="lomo", heldout=held, n_bins=Xtr.shape[0],
                                   p_idle_w=coef[0], e_f_pj=coef[1] * 1e12, e_w_pj=coef[2] * 1e12,
                                   e_kv_pj=coef[3] * 1e12, r2=r2_tr, mape_pct=float(np.median(errs))))

    import pandas as pd
    pd.DataFrame(coeff_rows).to_csv(out_dir / "coefficients.csv", index=False)
    pd.DataFrame(run_rows).to_csv(out_dir / "lomo_runs.csv", index=False)
    print(f"\nWrote {out_dir}/coefficients.csv and lomo_runs.csv")

    df = pd.DataFrame(run_rows)
    if not df.empty:
        hws = sorted(df["hardware"].unique())
        fig, axes = plt.subplots(1, len(hws), figsize=(5.5 * len(hws), 5.0), squeeze=False)
        for ax, hw in zip(axes[0], hws):
            sub = df[df["hardware"] == hw]
            for fam, g in sub.groupby("heldout_family"):
                ax.scatter(g["measured_mean_w"], g["predicted_mean_w"], s=18, alpha=0.7, label=fam)
            lim = [0, max(sub["measured_mean_w"].max(), sub["predicted_mean_w"].max()) * 1.05]
            ax.plot(lim, lim, "k--", lw=1)
            ax.plot(lim, [x * 1.1 for x in lim], "k:", lw=0.7)
            ax.plot(lim, [x * 0.9 for x in lim], "k:", lw=0.7)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel("Measured mean node power (W)")
            ax.set_ylabel("LOMO-predicted mean node power (W)")
            ax.set_title(f"{hw} (held-out family, ±10% dotted)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"lomo_pred_vs_measured.{ext}")
        print(f"Wrote {out_dir}/lomo_pred_vs_measured.pdf/.png")


if __name__ == "__main__":
    main()
