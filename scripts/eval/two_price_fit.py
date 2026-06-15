"""Fit the two-price power model P = c0 + c1*f + c2*g on raw measured traces.

Pipeline (per model x hardware x TP config, from raw data only):
  1. Window each run into W-second bins: mean node power (sum over the TP-group
     GPUs from nvidia-smi samples), prefill token rate f (input tokens spread
     over [arrival, arrival+TTFT]), decode token rate g (token events at
     arrival+TTFT+cumsum(ITL)), and per-window latency stats (mean ITL, mean
     TTFT per prefill token).
  2. OLS regression P = c0 + c1 f + c2 g  -> idle floor, J/prefill-token,
     J/decode-token, R^2, c1/c2 ratio, corr(f, g).
  3. Behavioral knee: per-load-bin median ITL vs decode rate (and normalized
     TTFT vs prefill rate); knee = first bin where latency exceeds 1.25x the
     low-load baseline.  F, G = sustained per-phase throughput at the knee.
  4. Validation figure: power vs l = f/F + g/G colored by prefill share, plus
     residual-vs-load and latency-vs-load panels.

Inputs are ONLY the raw measured files in data/sharegpt-benchmark-*/:
  *.csv  : timestamp, power.draw [W], utilization.gpu [%], memory.used [MiB]
  *.json : vLLM benchmark output (request_timestamps, ttfts, itls, input_lens,
           output_lens, request_rate, tensor_parallel_size, ...)

Usage:
  uv run python scripts/eval/two_price_fit.py [--window 5] [--configs llama-3-8b-h100 ...]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "two_price_fit")
FIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "two_price_fit")

CSV_NAME_RE = re.compile(r"^(?P<model>.+)_tp(?P<tp>\d+)_p(?P<rate>[\d.]+)_d(?P<stamp>[\d-]+)\.csv$")
ACTIVE_MEM_MIB = 5000.0  # GPUs in the TP group hold model weights (tens of GiB)
LATENCY_KNEE_FACTOR = 1.25


def parse_power_csv(path, tp):
    """Return (sample_times_epoch, node_power_w) for the TP-group GPUs.

    nvidia-smi rows interleave all GPUs on the host; the TP group is the set
    with model weights resident (memory.used > ACTIVE_MEM_MIB).  Node power at
    a window is mean(active-row power) * tp, which equals the average over
    sample frames of the summed TP-group power as long as each frame samples
    every GPU once (it does; nvidia-smi dmon-style loop).
    """
    times, powers = [], []
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                mem = float(row[3].strip().split()[0])
                if mem < ACTIVE_MEM_MIB:
                    continue
                p = float(row[1].strip().split()[0])
                ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                ts = ts.replace(tzinfo=timezone.utc).timestamp()
            except (ValueError, IndexError):
                continue
            times.append(ts)
            powers.append(p)
    return np.asarray(times), np.asarray(powers) * tp


def find_json_for_csv(cfg_dir, m):
    """Match power CSV to its vLLM JSON by tp, rate, and datetime stamp."""
    pats = glob.glob(os.path.join(cfg_dir, f"*{m['stamp']}.json"))
    pats = [p for p in pats if f"tp{m['tp']}" in os.path.basename(p)]
    return pats[0] if len(pats) == 1 else None


def window_run(power_t, power_w, bench, window):
    """Bin one run into W-second windows.  Returns dict of per-window arrays."""
    arrivals = np.asarray(bench["request_timestamps"], dtype=float)
    ttfts = np.asarray(bench["ttfts"], dtype=float)
    in_lens = np.asarray(bench["input_lens"], dtype=float)
    itls = bench["itls"]
    n_req = len(arrivals)
    if n_req == 0 or len(power_t) == 0:
        return None

    # Decode token event times: first token at arrival+ttft, rest via ITLs.
    dec_times, dec_itl = [], []
    for i in range(n_req):
        t0 = arrivals[i] + ttfts[i]
        deltas = np.asarray(itls[i], dtype=float)
        dec_times.append(t0 + np.concatenate(([0.0], np.cumsum(deltas))))
        dec_itl.append(np.concatenate(([np.nan], deltas)))
    dec_times = np.concatenate(dec_times)
    dec_itl = np.concatenate(dec_itl)

    t_start = arrivals.min()
    t_end = dec_times.max()
    # Keep only windows fully inside the active benchmark interval.
    first_edge = np.ceil(t_start / window) * window
    edges = np.arange(first_edge, np.floor(t_end / window) * window + window / 2, window)
    if len(edges) < 3:
        return None
    n_win = len(edges) - 1

    # Mean node power per window.
    p_idx = np.searchsorted(edges, power_t) - 1
    valid = (p_idx >= 0) & (p_idx < n_win)
    p_sum = np.bincount(p_idx[valid], weights=power_w[valid], minlength=n_win)
    p_cnt = np.bincount(p_idx[valid], minlength=n_win)

    # Prefill tokens/s: spread input_lens[i] uniformly over [arrival, arrival+ttft],
    # accumulated by overlap with each window.
    f_tok = np.zeros(n_win)
    ttft_per_tok_sum = np.zeros(n_win)
    ttft_cnt = np.zeros(n_win)
    for i in range(n_req):
        a, b = arrivals[i], arrivals[i] + max(ttfts[i], 1e-6)
        lo = max(int((a - edges[0]) // window), 0)
        hi = min(int((b - edges[0]) // window), n_win - 1)
        if hi < 0 or lo > n_win - 1:
            continue
        dur = b - a
        for w in range(lo, hi + 1):
            ov = min(b, edges[w + 1]) - max(a, edges[w])
            if ov > 0:
                f_tok[w] += in_lens[i] * ov / dur
        wa = int((a - edges[0]) // window)
        if 0 <= wa < n_win and in_lens[i] > 0:
            ttft_per_tok_sum[wa] += ttfts[i] / in_lens[i]
            ttft_cnt[wa] += 1

    # Decode tokens/s and mean ITL per window.
    d_idx = np.searchsorted(edges, dec_times) - 1
    dvalid = (d_idx >= 0) & (d_idx < n_win)
    g_tok = np.bincount(d_idx[dvalid], minlength=n_win).astype(float)
    itl_ok = dvalid & np.isfinite(dec_itl)
    itl_sum = np.bincount(d_idx[itl_ok], weights=dec_itl[itl_ok], minlength=n_win)
    itl_cnt = np.bincount(d_idx[itl_ok], minlength=n_win)

    keep = p_cnt > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "power_w": (p_sum / np.maximum(p_cnt, 1))[keep],
            "f_tps": (f_tok / window)[keep],
            "g_tps": (g_tok / window)[keep],
            "itl_ms": np.where(itl_cnt > 0, itl_sum / np.maximum(itl_cnt, 1), np.nan)[keep] * 1e3,
            "ttft_per_tok_ms": np.where(ttft_cnt > 0, ttft_per_tok_sum / np.maximum(ttft_cnt, 1), np.nan)[keep] * 1e3,
        }


def knee_from_latency(load, lat, n_bins=20):
    """Behavioral knee: bin windows by load, take median latency per bin.
    Baseline = minimum binned median (the flat efficient region; per-token
    TTFT is overhead-dominated at low load so it falls before it rises).
    Knee = first bin past the minimum whose median exceeds
    LATENCY_KNEE_FACTOR x baseline.  Returns (knee_load, baseline) or
    (None, baseline) if latency never departs."""
    ok = np.isfinite(load) & np.isfinite(lat) & (load > 0)
    if ok.sum() < 50:
        return None, np.nan
    load, lat = load[ok], lat[ok]
    qs = np.unique(np.quantile(load, np.linspace(0, 1, n_bins + 1)))
    centers, medians = [], []
    for i in range(len(qs) - 1):
        sel = (load >= qs[i]) & (load <= qs[i + 1])
        if sel.sum() >= 5:
            centers.append(np.median(load[sel]))
            medians.append(np.median(lat[sel]))
    if len(centers) < 4:
        return None, np.nan
    centers, medians = np.asarray(centers), np.asarray(medians)
    i_min = int(np.argmin(medians))
    base = medians[i_min]
    above = np.where(medians[i_min:] > LATENCY_KNEE_FACTOR * base)[0]
    if len(above) == 0:
        return None, base
    return centers[i_min + above[0]], base


def collect_runs(cfg_dir, window):
    """Parse all run pairs in a config dir, grouped by tensor parallelism."""
    by_tp = {}
    for path in sorted(glob.glob(os.path.join(cfg_dir, "*.csv"))):
        m = CSV_NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        jpath = find_json_for_csv(cfg_dir, m)
        if jpath is None:
            continue
        try:
            with open(jpath) as fh:
                bench = json.load(fh)
            tp = int(m["tp"])
            pt, pw = parse_power_csv(path, tp)
            win = window_run(pt, pw, bench, window)
        except Exception as exc:  # noqa: BLE001 - skip malformed runs, report
            print(f"  ! skipped {os.path.basename(path)}: {exc}", file=sys.stderr)
            continue
        if win is None:
            continue
        win["rate"] = np.full(len(win["power_w"]), float(m["rate"]))
        by_tp.setdefault(tp, []).append(win)
    return by_tp


def fit_tp_group(cfg_name, tp, runs):
    P = np.concatenate([r["power_w"] for r in runs])
    f = np.concatenate([r["f_tps"] for r in runs])
    g = np.concatenate([r["g_tps"] for r in runs])
    itl = np.concatenate([r["itl_ms"] for r in runs])
    ttft_pt = np.concatenate([r["ttft_per_tok_ms"] for r in runs])

    # 2. OLS P = c0 + c1 f + c2 g
    X = np.column_stack([np.ones_like(f), f, g])
    coef, _, _, _ = np.linalg.lstsq(X, P, rcond=None)
    pred = X @ coef
    resid = P - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((P - P.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr_fg = float(np.corrcoef(f, g)[0, 1])

    # 3. Behavioral knee in TOTAL fitted load L = c1 f + c2 g (joint, because
    # corr(f,g) is high and per-phase marginal latencies are confounded by
    # queueing).  The knee defines the iso-load capacity line c1 F = c2 G = L*,
    # so F = L*/c1, G = L*/c2 and ell = L/L*.
    # ITL is the saturation signal; per-token TTFT departs at trivial load
    # from scheduler queueing, so it is reported but not used for L*.
    load_proxy = coef[1] * f + coef[2] * g
    knee_itl, itl_base = knee_from_latency(load_proxy, itl)
    knee_ttft, _ = knee_from_latency(load_proxy, ttft_pt)
    knee_found = knee_itl is not None
    L_star = knee_itl if knee_found else float(np.quantile(load_proxy, 0.99))
    F = L_star / coef[1] if coef[1] > 0 else np.nan
    G = L_star / coef[2] if coef[2] > 0 else np.nan
    ell = load_proxy / L_star
    f_share = np.where(f + g > 0, f / np.maximum(f + g, 1e-9), 0.0)

    # Residual curvature vs load: mean residual in top vs middle load tercile.
    t1, t2 = np.quantile(load_proxy, [1 / 3, 2 / 3])
    resid_mid = float(resid[(load_proxy > t1) & (load_proxy <= t2)].mean())
    resid_hi = float(resid[load_proxy > t2].mean())

    summary = {
        "config": cfg_name,
        "tp": tp,
        "n_runs": len(runs),
        "n_windows": len(P),
        "c0_idle_w": float(coef[0]),
        "c1_j_per_prefill_tok": float(coef[1]),
        "c2_j_per_decode_tok": float(coef[2]),
        "c1_over_c2": float(coef[1] / coef[2]) if coef[2] != 0 else np.nan,
        "r2": float(r2),
        "rmse_w": float(np.sqrt(ss_res / len(P))),
        "mean_power_w": float(P.mean()),
        "corr_f_g": corr_fg,
        "L_knee_w": float(L_star),
        "knee_found": knee_found,
        "knee_metric": "itl" if knee_found else "p99-fallback",
        "ttft_knee_w": float(knee_ttft) if knee_ttft is not None else np.nan,
        "F_prefill_tps": float(F),
        "G_decode_tps": float(G),
        "itl_baseline_ms": float(itl_base) if np.isfinite(itl_base) else np.nan,
        "resid_mean_mid_w": resid_mid,
        "resid_mean_hi_w": resid_hi,
    }

    arrays = {"P": P, "f": f, "g": g, "itl": itl, "ttft_pt": ttft_pt,
              "ell": ell, "f_share": f_share, "pred": pred, "resid": resid,
              "load_proxy": load_proxy}
    return summary, arrays


def make_figure(cfg_name, s, a, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    sc = ax.scatter(a["ell"], a["P"], c=a["f_share"], s=5, alpha=0.4,
                    cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
    ax.set_xlabel(r"modeled load  $\ell = f/F + g/G$")
    ax.set_ylabel("measured node power [W]")
    ax.set_title(f"{cfg_name}\n$R^2$={s['r2']:.3f}, c1/c2={s['c1_over_c2']:.2f}")
    fig.colorbar(sc, ax=ax, label="prefill share of tokens")

    ax = axes[1]
    ax.scatter(a["load_proxy"], a["resid"], s=4, alpha=0.3, rasterized=True)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"fitted load  $c_1 f + c_2 g$  [W]")
    ax.set_ylabel("residual [W]")
    ax.set_title(f"residuals (mid {s['resid_mean_mid_w']:+.1f} W, "
                 f"hi {s['resid_mean_hi_w']:+.1f} W)")

    ax = axes[2]
    ok = np.isfinite(a["itl"])
    ax.scatter(a["load_proxy"][ok], a["itl"][ok], s=4, alpha=0.3,
               color="tab:green", label="mean ITL", rasterized=True)
    ok2 = np.isfinite(a["ttft_pt"])
    ax.scatter(a["load_proxy"][ok2], a["ttft_pt"][ok2], s=4, alpha=0.3,
               color="tab:orange", label="TTFT / prefill tok", rasterized=True)
    if s["knee_found"]:
        ax.axvline(s["L_knee_w"], color="r", ls="--",
                   label=f"knee L*={s['L_knee_w']:.0f} W ({s['knee_metric']})")
    ax.legend(markerscale=3)
    ax.set_xlabel(r"fitted load  $c_1 f + c_2 g$  [W]")
    ax.set_ylabel("latency [ms]")
    ax.set_yscale("log")
    ax.set_title("behavioral saturation")

    fig.tight_layout()
    path = os.path.join(fig_dir, f"{cfg_name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=5.0, help="window size [s]")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="suffixes of sharegpt-benchmark-* dirs to include")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(FIG_ROOT, exist_ok=True)

    cfg_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "sharegpt-benchmark-*")))
    if args.configs:
        cfg_dirs = [d for d in cfg_dirs
                    if any(d.endswith(c) for c in args.configs)]

    summaries = []
    for cfg_dir in cfg_dirs:
        base_name = os.path.basename(cfg_dir).replace("sharegpt-benchmark-", "")
        print(f"== {base_name}")
        by_tp = collect_runs(cfg_dir, args.window)
        for tp in sorted(by_tp):
            cfg_name = f"{base_name}_tp{tp}"
            s, arrays = fit_tp_group(cfg_name, tp, by_tp[tp])
            summaries.append(s)
            np.savez_compressed(
                os.path.join(OUT_ROOT, f"windows_{cfg_name}.npz"), **arrays)
            fig_path = make_figure(cfg_name, s, arrays, FIG_ROOT)
            print(f"   tp{tp}: {s['n_runs']} runs, {s['n_windows']} windows | "
                  f"P = {s['c0_idle_w']:.0f} + {s['c1_j_per_prefill_tok']:.4f} f "
                  f"+ {s['c2_j_per_decode_tok']:.4f} g  (R2={s['r2']:.3f}, "
                  f"c1/c2={s['c1_over_c2']:.2f}, corr(f,g)={s['corr_f_g']:.2f})")
            print(f"        L*={s['L_knee_w']:.0f} W ({s['knee_metric']}) -> "
                  f"F={s['F_prefill_tps']:.0f}, G={s['G_decode_tps']:.0f} tok/s "
                  f"| {fig_path}")

    if summaries:
        keys = list(summaries[0].keys())
        with open(os.path.join(OUT_ROOT, "summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(summaries)
        print(f"\nwrote {os.path.join(OUT_ROOT, 'summary.csv')}")


if __name__ == "__main__":
    main()
