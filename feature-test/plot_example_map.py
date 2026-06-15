"""Example plot of the MAP-with-priors model: measured vs predicted power,
decomposed into physical components.

Run: uv run python feature-test/plot_example_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature-test"))

from final_model import FEATS, IDLE_FEATS, design  # noqa: E402
from fit_map_priors import fit_two_stage, predict_map  # noqa: E402
from fit_models import load_cache  # noqa: E402

GROUPS = {
    "idle + NVLink": ["tp", "tp_link"],
    "clocks/DVFS overhead": ["busy_tp", "sat_cmp_lo", "sat_cmp_hi", "sat_mem_lo", "sat_mem_hi"],
    "prefill": ["flops_pre", "w_read_pre"],
    "decode": ["flops_dec", "w_read_dec", "kv_read", "kv_write", "comm"],
}
COLORS = {
    "idle + NVLink": "#bdbdbd",
    "clocks/DVFS overhead": "#fdd0a2",
    "prefill": "#e6550d",
    "decode": "#3182bd",
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
    th, mult, _ = fit_two_stage(X[hw_mask], y[hw_mask], fams[hw_mask], idle_cols)
    c = np.exp(th)

    is70 = np.array([mn[i] == "llama-3-70b" for i in d["model_idx"]])
    cand = hw_mask & is70 & (d["tp"] == 8) & (d["rate"] == 0.5)
    rid = np.unique(run_ids[cand])[0]
    m = run_ids == rid
    mfac = mult.get(int(fams[m][0]), 1.0)

    contrib = {}
    for gname, gfeats in GROUPS.items():
        cols = np.array([f in gfeats for f in FEATS])
        contr = X[m][:, cols] @ c[cols]
        if gname != "idle + NVLink":
            contr = contr * mfac
        contrib[gname] = contr
    pred = predict_map(X[m], th, mult, fams[m], idle_cols)

    t = np.arange(int(m.sum()))
    w0, w1 = 0, min(360, t.size)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4),
                             gridspec_kw={"height_ratios": [1, 1.4]})
    ax = axes[0]
    ax.plot(t, y[m], color="k", lw=0.7, alpha=0.8, label="measured")
    ax.plot(t, pred, color="#d62728", lw=1.1, label="MAP model")
    ax.axvspan(w0, w1, color="gold", alpha=0.15)
    ax.set_ylabel("node power (W)")
    ax.set_title("llama-3-70b on 8×H100, ShareGPT @ 0.5 req/s — MAP model (physical priors, Student-t)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sl = slice(w0, w1)
    bottom = np.zeros(w1 - w0)
    for gname in GROUPS:
        cc = contrib[gname][sl]
        ax.fill_between(t[sl], bottom, bottom + cc, color=COLORS[gname], label=gname, lw=0)
        bottom += cc
    ax.plot(t[sl], y[m][sl], color="k", lw=0.9, label="measured")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("node power (W)")
    ax.set_title("MAP prediction decomposed into physical components (first 6 min)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(w0, w1 - 1)

    fig.tight_layout()
    out = Path("feature-test/results/example_decomposition_map.png")
    fig.savefig(out, dpi=140)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
