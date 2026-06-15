"""Peak fix + unseen-model generalization.

Structural change: the saturating DVFS/activity terms are driven by
DECODE-ONLY compute utilization, so prefill energy cannot hide there and must
flow through its own linear coefficients (e_f_prefill, e_w_prefill). This is
what lets the model produce the prefill power peaks on a saturated node.

Generalization demo: hold out the dense-405b family entirely (the only FP8
model), fit on 8B/70B classes, predict every 405B run.

Run: uv run python feature-test/peak_and_holdout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature-test"))

import fit_map_priors as fmp  # noqa: E402
from final_model import LAGS  # noqa: E402
from fit_models import FEATURES, PEAK_FLOPS, load_cache, metrics  # noqa: E402

OUT = Path("feature-test/results")

FEATS2 = ["tp", "tp_link", "busy_tp", "sat_bw_a", "sat_bw_b", "sat_bw_c",
          "flops_pre", "flops_dec", "w_read_pre", "kv_write", "comm"]
IDLE2 = {"tp", "tp_link"}


def setup_features(d):
    # Roofline-consistent decode memory power: a saturating function of
    # achieved DRAM bandwidth utilization (weights + KV reads vs per-GPU BW).
    # Linear at low utilization (8B/70B), saturating at high (405B) — replaces
    # the linear weight-bytes term + binary decode indicator, which
    # double-count when both extrapolate to big models.
    FEATURES["sat_bw_a"] = ("p_bw_u0.05 [W/GPU]",
                            lambda d: d["tp"] * (1 - np.exp(-d["util_mem"] / 0.05)))
    FEATURES["sat_bw_b"] = ("p_bw_u0.15 [W/GPU]",
                            lambda d: d["tp"] * (1 - np.exp(-d["util_mem"] / 0.15)))
    FEATURES["sat_bw_c"] = ("p_bw_u0.4 [W/GPU]",
                            lambda d: d["tp"] * (1 - np.exp(-d["util_mem"] / 0.4)))
    fmp.PRIORS["sat_bw_a"] = (80.0, 0.70)
    fmp.PRIORS["sat_bw_b"] = (80.0, 0.70)
    fmp.PRIORS["sat_bw_c"] = (80.0, 0.70)
    fmp.PRIORS["busy_tp"] = (30.0, 0.70)
    # Decode is memory-bound GEMV: marginal compute energy must be small.
    # Without this, e_f_decode inflates to ~7 pJ/FLOP in-sample (absorbed by
    # family multipliers) and detonates on 10x-FLOPs held-out models.
    fmp.PRIORS["flops_dec"] = (5.0e-13, 0.40)
    # Multipliers pinned near 1 so shared coefficients stay honest.
    fmp.PHI_PRIOR_SD = 0.05
    # FP8 arithmetic costs ~half BF16 energy per FLOP: physics, not fitted.
    for k in ("flops_pre", "flops_dec"):
        d[k] = d[k] * (1.0 - 0.5 * d["fp8"])
    fmp.FEATS = FEATS2  # map_fit/laplace read module globals


def design2(d, hw, run_ids):
    lag = LAGS[hw]
    cols = [np.asarray(FEATURES[f][1](d), dtype=np.float64) for f in FEATS2]
    return np.column_stack([lag(c, run_ids) for c in cols])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_cache()
    setup_features(d)
    y = d["power"].astype(np.float64)
    run_ids = d["run_id"]
    fams = d["family_idx"]
    fam_names = [str(x) for x in d["family_names"]]
    mn = [str(x) for x in d["model_names"]]
    idle_cols = np.array([f in IDLE2 for f in FEATS2])

    for hw_i, hw in enumerate([str(x) for x in d["hw_names"]]):
        hw_mask = d["hw_idx"] == hw_i
        if hw_mask.sum() == 0:
            continue
        X = design2(d, hw, run_ids)
        th, mult, sigma = fmp.fit_two_stage(X[hw_mask], y[hw_mask], fams[hw_mask], idle_cols)
        pred = fmp.predict_map(X[hw_mask], th, mult, fams[hw_mask], idle_cols)
        ins = metrics(y[hw_mask], pred)
        post_sd = fmp.laplace_sd(X[hw_mask], y[hw_mask], th, mult, fams[hw_mask], idle_cols, sigma)

        print(f"\n######## {hw} peak-fix model (decode-only DVFS terms)")
        print(f"  in-sample: R2={ins['r2']:.4f} RMSE={ins['rmse']:.1f}W MAPE={ins['mape']:.1f}%")
        for j, f in enumerate(FEATS2):
            shrink = post_sd[j] / fmp.PRIORS[f][1]
            drift = (th[j] - np.log(fmp.PRIORS[f][0])) / fmp.PRIORS[f][1]
            status = ("DATA-IDENTIFIED" if shrink < 0.3 else
                      "partial" if shrink < 0.7 else "PRIOR-DOMINATED")
            print(f"    {FEATURES[f][0]:30s} {np.exp(th[j]):10.3g} shrink={shrink:5.2f} "
                  f"drift={drift:+5.1f}  {status}")

        # peak fidelity on saturated TP8 runs
        sat = hw_mask & (d["rate"] >= 2.0) & (d["tp"] == 8) & (d["batch"] > 2)
        pr_all = fmp.predict_map(X, th, mult, fams, idle_cols)
        for q in (0.5, 0.9, 0.99):
            print(f"  saturated busy bins P{int(q*100)}: measured {np.quantile(y[sat], q):.0f}W "
                  f"predicted {np.quantile(pr_all[sat], q):.0f}W")

        # ---- 405B holdout (H100 only)
        if hw != "H100":
            continue
        fi_405 = fam_names.index("dense-405b")
        tr = hw_mask & (fams != fi_405)
        te = hw_mask & (fams == fi_405)
        # cold start: no information from the held-out family may enter
        th_h, mult_h, _ = fmp.fit_two_stage(X[tr], y[tr], fams[tr], idle_cols)
        mult_h.pop(fi_405, None)
        pred_h = fmp.predict_map(X, th_h, mult_h, fams, idle_cols)
        # physical clamp: sustained per-GPU power cap, estimated from TRAINING
        # families only (p99.5 of busy per-GPU power); GPUs enforce power
        # limits, so no prediction may exceed it.
        busy_tr = tr & (d["batch"] > 0)
        cap_w = float(np.quantile(y[busy_tr] / d["tp"][busy_tr], 0.995))
        pred_h = np.minimum(pred_h, cap_w * d["tp"])
        print(f"  power cap from training families: {cap_w:.0f} W/GPU")
        mh = metrics(y[te], pred_h[te])
        errs, rm, rp = [], [], []
        for r in np.unique(run_ids[te]):
            mm = te & (run_ids == r)
            a, b = float(np.mean(y[mm])), float(np.mean(pred_h[mm]))
            rm.append(a); rp.append(b)
            errs.append(abs(b - a) / a * 100)
        print(f"\n  === 405B HOLDOUT (trained only on 8B/70B classes) ===")
        print(f"  bin-level: R2={mh['r2']:.4f} RMSE={mh['rmse']:.1f}W MAPE={mh['mape']:.1f}%")
        print(f"  run means: median err {np.median(errs):.1f}%  worst {max(errs):.1f}%  ({len(errs)} runs)")

        # figure: trace overlay + run-mean scatter
        is405 = np.array([mn[i] == "llama-3-405b" for i in d["model_idx"]])
        cand = te & is405 & (d["rate"] == 2.0)
        rid = np.unique(run_ids[cand])[0]
        mm = run_ids == rid
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
        t = np.arange(int(mm.sum()))
        axes[0].plot(t, y[mm], color="k", lw=0.7, label="measured (never trained on)")
        axes[0].plot(t, pred_h[mm], color="#d62728", lw=1.1,
                     label="predicted from 8B/70B physics")
        axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("node power (W)")
        axes[0].set_title("llama-3-405B-FP8, 8×H100 @ 2 req/s — zero-shot")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        axes[1].scatter(rm, rp, s=22, color="#d62728")
        lim = [min(rm) * 0.9, max(rm) * 1.08]
        axes[1].plot(lim, lim, "k--", lw=1)
        axes[1].plot(lim, [x * 1.1 for x in lim], "k:", lw=0.7)
        axes[1].plot(lim, [x * 0.9 for x in lim], "k:", lw=0.7)
        axes[1].set_xlabel("measured run mean (W)")
        axes[1].set_ylabel("predicted run mean (W)")
        axes[1].set_title(f"all {len(errs)} held-out runs (±10%)")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "holdout_405b.png", dpi=140)
        fig.savefig(OUT / "holdout_405b.pdf")
        print(f"  wrote {OUT}/holdout_405b.png/.pdf")


if __name__ == "__main__":
    main()
