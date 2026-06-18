"""Publication figure: recommended F1_phase model overlaid on the held-out
validate workloads (never used for fitting).

Run:
    srun -p dev --time=00:10:00 --mem=4GB $SCRATCH/runpy.sh \
        new-profiling-model/plot_heldout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataio
import physics
from fit_eval import (RUNS, STAIRCASE, VALIDATE, fit_with_multiplier, predict,
                      metrics, window_metrics, run_mean_errors, r2)

MODEL = "F1_phase"
TITLES = {
    "google/gemma-4-31B-it": "gemma-4-31B  (dense, 30 B active)",
    "google/gemma-4-26B-A4B-it": "gemma-4-26B-A4B  (MoE, 128 experts / top-8, 4 B active)",
}


def build_fit():
    idle_pts = []
    for r in ["a100_decode_staircase_tp2_1781723125",
              "a100_decode_staircase_tp2_1781723242"] + VALIDATE:
        ip = dataio.idle_point(RUNS / r)
        if ip is not None and 120 < ip["power"] < 260:
            idle_pts.append(ip)
    probe_pts = []
    for r in STAIRCASE:
        probe_pts += dataio.load_staircase(RUNS / r)
    return fit_with_multiplier(idle_pts + probe_pts, physics.MODELS[MODEL])


def main():
    fit = build_fit()
    val = {r: dataio.load_validate_bins(RUNS / r) for r in VALIDATE}
    val = {r: p for r, p in val.items() if p}

    fig, axes = plt.subplots(len(val), 1, figsize=(11, 3.5 * len(val)),
                             squeeze=False)
    axes = axes[:, 0]
    for ax, (r, pts) in zip(axes, val.items()):
        y = np.array([p["power"] for p in pts])
        pred = predict(fit, pts)
        t = np.arange(y.size)
        m = metrics(y, pred)
        w30 = window_metrics(y, pred, 30)
        ee = list(run_mean_errors(pts, pred).values())[0]

        ax.fill_between(t, 0, y, color="0.85", zorder=0)
        ax.plot(t, y, color="0.25", lw=1.4, label="measured (nvidia-smi)", zorder=2)
        ax.plot(t, pred, color="#d1495b", lw=1.8, alpha=0.9,
                label=f"{MODEL} prediction", zorder=3)
        # 30 s window means (the energy-fidelity view)
        n = (y.size // 30) * 30
        if n >= 30:
            wt = np.arange(15, n, 30)
            ax.plot(wt, y[:n].reshape(-1, 30).mean(1), "o", color="0.25", ms=6, zorder=4)
            ax.plot(wt, pred[:n].reshape(-1, 30).mean(1), "D", color="#d1495b",
                    ms=6, zorder=5, label="30 s window mean")

        ax.set_title(TITLES.get(pts[0]["model"], pts[0]["model"]), fontsize=11)
        ax.set_ylabel("node power (W)")
        ax.set_ylim(0, max(y.max(), pred.max()) * 1.18)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        ax.text(0.012, 0.94,
                f"held out from fitting   |   energy error {ee:.1f}%   "
                f"|   30 s-window R² {w30['r2']:.2f}, MAPE {w30['mape']:.1f}%   "
                f"|   1 s R² {m['r2']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("First-principles power model on held-out validate workloads "
                 "(A100, TP=2)\nfit only on staircase probes + idle anchor",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = HERE / "results" / "heldout_overlay.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
