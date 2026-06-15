"""Plot an example of the final model: measured power vs the prediction,
decomposed into physical components (idle / NVLink / clocks / prefill / decode).

Run: uv run python feature-test/plot_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature-test"))

from final_model import FEATS, IDLE_FEATS, LAGS, design, fit_hw, predict  # noqa: E402
from fit_models import FEATURES, load_cache  # noqa: E402

GROUPS = {
    "idle": ["tp"],
    "NVLink standing": ["tp_link"],
    "clocks/DVFS overhead": ["busy_tp", "sat_cmp_lo", "sat_cmp_hi", "sat_mem_lo", "sat_mem_hi"],
    "prefill": ["flops_pre", "w_read_pre"],
    "decode": ["flops_dec", "w_read_dec", "kv_read", "kv_write", "comm"],
}
COLORS = {
    "idle": "#bdbdbd", "NVLink standing": "#969696",
    "clocks/DVFS overhead": "#fdd0a2", "prefill": "#e6550d", "decode": "#3182bd",
}


def main():
    d = load_cache()
    y = d["power"].astype(np.float64)
    run_ids = d["run_id"]
    fams = d["family_idx"]
    idle_cols = np.array([f in IDLE_FEATS for f in FEATS])
    mn = [str(x) for x in d["model_names"]]

    hw_i, hw = 1, "H100"
    hw_mask = d["hw_idx"] == hw_i
    X = design(d, hw, run_ids)
    coef, mult = fit_hw(X, y, hw_mask, fams, idle_cols)

    # pick a llama-3-70b H100 TP8 run at rate 0.5 (visible bursts + gaps)
    is70 = np.array([mn[i] == "llama-3-70b" for i in d["model_idx"]])
    cand = hw_mask & is70 & (d["tp"] == 8) & (d["rate"] == 0.5)
    rid = np.unique(run_ids[cand])[0]
    m = run_ids == rid
    fam = int(fams[m][0])
    mfac = mult.get(fam, 1.0)

    # per-group contributions (lagged features x coefs; multiplier on dynamics)
    contrib = {}
    for gname, gfeats in GROUPS.items():
        cols = np.array([f in gfeats for f in FEATS])
        c = X[m][:, cols] @ coef[cols]
        if gname not in ("idle", "NVLink standing"):
            c = c * mfac
        contrib[gname] = c
    pred = predict(X, coef, mult, fams, idle_cols)[m]

    t = np.arange(int(m.sum()))
    w0, w1 = 0, min(360, t.size)  # first 6 minutes

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4),
                             gridspec_kw={"height_ratios": [1, 1.4]})
    ax = axes[0]
    ax.plot(t, y[m], color="k", lw=0.7, alpha=0.8, label="measured")
    ax.plot(t, pred, color="#d62728", lw=1.1, label="model")
    ax.axvspan(w0, w1, color="gold", alpha=0.15)
    ax.set_ylabel("node power (W)")
    ax.set_title("llama-3-70b on 8×H100, ShareGPT @ 0.5 req/s — measured vs first-principles model")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sl = slice(w0, w1)
    bottom = np.zeros(w1 - w0)
    for gname in GROUPS:
        c = contrib[gname][sl]
        ax.fill_between(t[sl], bottom, bottom + c, color=COLORS[gname],
                        label=gname, lw=0)
        bottom += c
    ax.plot(t[sl], y[m][sl], color="k", lw=0.9, label="measured")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("node power (W)")
    ax.set_title("prediction decomposed into physical components (first 6 min)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(w0, w1 - 1)

    fig.tight_layout()
    out = Path("feature-test/results/example_decomposition.png")
    fig.savefig(out, dpi=140)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
