"""Physical-component decomposition for the MoE models (gpt-oss 20B & 120B).

Uses the model fit on gpt-oss-20B ONLY. 20B is in-family; 120B is zero-shot.
Shows how the energy breakdown shifts with scale under identical physics.

Run: uv run python feature-test/plot_decomposition_moe.py
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
from fit_models import FEATURES, load_cache  # noqa: E402

OUT = Path("feature-test/results")

GROUPS = {
    "idle + NVLink": ["tp", "tp_link"],
    "clock floor": ["busy_tp"],
    "decode memory (saturating)": ["sat_bw_a", "sat_bw_b", "sat_bw_c"],
    "prefill": ["flops_pre", "w_read_pre"],
    "decode compute + traffic": ["flops_dec", "kv_write", "comm"],
}
COLORS = {
    "idle + NVLink": "#bdbdbd",
    "clock floor": "#969696",
    "decode memory (saturating)": "#3182bd",
    "prefill": "#e6550d",
    "decode compute + traffic": "#74c476",
}


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
    X = design2(d, "A100", run_ids)
    th, mult, _ = fmp.fit_two_stage(X[is20], y[is20], fams[is20], idle_cols)
    c = np.exp(th)
    fi20 = fam_names.index("moe-20b")
    cap = float(np.quantile(y[is20 & (d["batch"] > 0)] / d["tp"][is20 & (d["batch"] > 0)], 0.995))

    panels = [
        ("gpt-oss-20b", 2, 2.0, mult.get(fi20, 1.0), "fit family"),
        ("gpt-oss-120b", 8, 2.0, 1.0, "ZERO-SHOT"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2))
    for ax, (name, tp, rate, mfac, tag) in zip(axes, panels):
        ism = np.array([mn[i] == name for i in d["model_idx"]])
        cand = ism & (d["tp"] == tp) & (d["rate"] == rate)
        rid = np.unique(run_ids[cand])[0]
        m = run_ids == rid
        w1 = min(360, int(m.sum()))
        t = np.arange(w1)

        bottom = np.zeros(w1)
        for gname, gfeats in GROUPS.items():
            cols = np.array([f in gfeats for f in FEATS2])
            contr = (X[m][:w1, cols] @ c[cols])
            if gname not in ("idle + NVLink",):
                contr = contr * mfac
            ax.fill_between(t, bottom, bottom + contr, color=COLORS[gname], label=gname, lw=0)
            bottom += contr
        ax.plot(t, y[m][:w1], color="k", lw=0.9, label="measured")
        ax.set_xlim(0, w1 - 1)
        ax.set_ylabel("node power (W)")
        ax.set_title(f"{name}, {tp}xA100 @ {rate:g} req/s  ({tag}) "
                     f"-- model fit on 20B only")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=8, ncol=3, loc="upper right")
    axes[-1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(OUT / "decomposition_moe.png", dpi=140)
    fig.savefig(OUT / "decomposition_moe.pdf")
    print(f"Wrote {OUT}/decomposition_moe.png/.pdf")

    # print the energy split for each panel (full run)
    for name, tp, rate, mfac, tag in panels:
        ism = np.array([mn[i] == name for i in d["model_idx"]])
        msk = ism & (d["tp"] == tp)
        tot = 0.0
        parts = {}
        for gname, gfeats in GROUPS.items():
            cols = np.array([f in gfeats for f in FEATS2])
            contr = (X[msk][:, cols] @ c[cols])
            if gname not in ("idle + NVLink",):
                contr = contr * (mfac if name == "gpt-oss-20b" else 1.0)
            parts[gname] = float(contr.sum())
            tot += parts[gname]
        print(f"\n{name} TP{tp} energy split ({tag}):")
        for g, v in parts.items():
            print(f"  {g:30s} {100*v/tot:5.1f}%")


if __name__ == "__main__":
    main()
