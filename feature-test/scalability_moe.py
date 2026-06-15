"""MoE scalability test: fit on gpt-oss-20b only, predict 20b AND 120b.

Extrapolates across model size (20B->120B), expert count (32->128), and GPU
count (TP1/2 -> TP4/8) within the mixture-of-experts architecture family.

Run: uv run python feature-test/scalability_moe.py
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
from peak_and_holdout import FEATS2, IDLE2, design2, setup_features  # noqa: E402
from fit_models import load_cache, metrics  # noqa: E402

OUT = Path("feature-test/results")


def main():
    d = load_cache()
    setup_features(d)
    y = d["power"].astype(np.float64)
    run_ids = d["run_id"]
    fams = d["family_idx"]
    mn = [str(x) for x in d["model_names"]]
    fam_names = [str(x) for x in d["family_names"]]
    idle_cols = np.array([f in IDLE2 for f in FEATS2])

    is20 = np.array([mn[i] == "gpt-oss-20b" for i in d["model_idx"]])
    is120 = np.array([mn[i] == "gpt-oss-120b" for i in d["model_idx"]])

    X = design2(d, "A100", run_ids)
    tr = is20  # FIT ON 20B ONLY, cold start
    th, mult, _ = fmp.fit_two_stage(X[tr], y[tr], fams[tr], idle_cols)
    cap = float(np.quantile(y[tr & (d["batch"] > 0)] / d["tp"][tr & (d["batch"] > 0)], 0.995))
    fi20 = fam_names.index("moe-20b")

    def predict(msk, use_mult):
        m = {fi20: mult.get(fi20, 1.0)} if use_mult else {}
        p = fmp.predict_map(X[msk], th, m, fams[msk], idle_cols)
        return np.minimum(p, cap * d["tp"][msk])

    def curve(is_model, tp, use_mult):
        rates, meas, pred = [], [], []
        for rate in [0.125, 0.25, 0.5, 1, 2, 4]:
            msk = is_model & (d["tp"] == tp) & (d["rate"] == rate)
            if not msk.sum():
                continue
            rates.append(rate)
            meas.append(float(y[msk].mean()))
            pred.append(float(predict(msk, use_mult).mean()))
        return np.array(rates), np.array(meas), np.array(pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    colors = {("gpt-oss-20b", 1): "#1f77b4", ("gpt-oss-20b", 2): "#2ca02c",
              ("gpt-oss-120b", 4): "#ff7f0e", ("gpt-oss-120b", 8): "#d62728"}
    for (is_m, name, tp, mlt) in [(is20, "gpt-oss-20b", 1, True), (is20, "gpt-oss-20b", 2, True),
                                  (is120, "gpt-oss-120b", 4, False), (is120, "gpt-oss-120b", 8, False)]:
        r, meas, pred = curve(is_m, tp, mlt)
        col = colors[(name, tp)]
        tag = "FIT" if is_m is is20 else "PREDICTED"
        ax.plot(r, meas, "o-", color=col, label=f"{name} TP{tp} measured")
        ax.plot(r, pred, "x--", color=col, label=f"{name} TP{tp} model ({tag})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("request rate (req/s)")
    ax.set_ylabel("mean node power (W)")
    ax.set_title("Fit on 20B only -> predict 20B and 120B power-vs-load")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    cand = is120 & (d["tp"] == 8) & (d["rate"] == 2.0)
    rid = np.unique(run_ids[cand])[0]
    mm = run_ids == rid
    p = predict(mm, False)
    t = np.arange(int(mm.sum()))
    ax.plot(t, y[mm], color="k", lw=0.7, label="measured (never trained on)")
    ax.plot(t, p, color="#d62728", lw=1.1, label="zero-shot from 20B physics")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("node power (W)")
    ax.set_title("gpt-oss-120B, 8xA100 @ 2 req/s -- zero-shot trace")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "scalability_moe.png", dpi=140)
    fig.savefig(OUT / "scalability_moe.pdf")

    e20 = []
    for r in np.unique(run_ids[is20]):
        s = run_ids[is20] == r
        e20.append(abs(predict(is20, True)[s].mean() / y[is20][s].mean() - 1) * 100)
    e120 = []
    for r in np.unique(run_ids[is120]):
        s = run_ids[is120] == r
        e120.append(abs(predict(is120, False)[s].mean() / y[is120][s].mean() - 1) * 100)
    print(f"20B fit-family run-mean err: median {np.median(e20):.1f}%")
    print(f"120B zero-shot run-mean err: median {np.median(e120):.1f}%  bin R2 "
          f"{metrics(y[is120], predict(is120, False))['r2']:.3f}")
    print(f"Wrote {OUT}/scalability_moe.png/.pdf")


if __name__ == "__main__":
    main()
