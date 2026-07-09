"""Operator-facing power-model parameters per node type.

Every value has a one-sentence definition, computed directly from the measured
windows (results/two_price_fit/windows_*.npz) with no fitting:

  F        : the fastest prefill rate the node ever sustained over a 5 s
             window [tok/s].
  G        : the fastest decode rate the node ever sustained over a 5 s
             window [tok/s].
  ell      : f/F + g/G -- how full the node is, as a fraction.
  rho*     : the highest ell at which inter-token latency is still within 25%
             of its best value (binned medians).  If the sweep never pushed
             latency up, rho* = 0.8 by convention (flagged).
  P_idle   : median measured power of windows serving zero tokens [W].
  P_busy   : median measured power of windows running at or above rho* [W].
  p_bar    : P_busy / rho* -- the pool-level price of load [W per node-unit].

The dispatch score is Delta P_j = p_bar * ell_j.  Nothing else is needed.

Usage:  uv run python scripts/eval/operator_table.py
"""

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "two_price_fit")
FIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "two_price_fit")

RHO_STAR_FALLBACK = 0.8
LATENCY_TOLERANCE = 1.25  # "within 25% of its best value"


def rho_star_from_itl(ell, itl, n_bins=20):
    """Highest ell whose binned median ITL is within 25% of the best median."""
    ok = np.isfinite(ell) & np.isfinite(itl) & (ell > 0)
    if ok.sum() < 50:
        return None
    ell, itl = ell[ok], itl[ok]
    edges = np.unique(np.quantile(ell, np.linspace(0, 1, n_bins + 1)))
    centers, medians = [], []
    for i in range(len(edges) - 1):
        sel = (ell >= edges[i]) & (ell <= edges[i + 1])
        if sel.sum() >= 5:
            centers.append(np.median(ell[sel]))
            medians.append(np.median(itl[sel]))
    if len(centers) < 4:
        return None
    centers, medians = np.asarray(centers), np.asarray(medians)
    best = medians.min()
    bad = np.where(medians > LATENCY_TOLERANCE * best)[0]
    bad = bad[bad > np.argmin(medians)]  # only departures past the best point
    if len(bad) == 0:
        return None  # latency never rose: sweep did not reach the knee
    return float(centers[bad[0] - 1]) if bad[0] > 0 else float(centers[0])


def fit_power_knee(ell, P, p_idle, p_busy):
    """Ramp-plateau node model: P = P_idle + (P_busy - P_idle) * min(ell/k, 1).
    The two levels are fixed by their plain definitions; only the knee k is
    chosen, by least squares over a grid.  Returns (k, r2)."""
    candidates = np.quantile(ell[ell > 0], np.linspace(0.01, 1.0, 200))
    best_k, best_sse = None, np.inf
    for k in np.unique(candidates):
        pred = p_idle + (p_busy - p_idle) * np.clip(ell / k, 0, 1)
        sse = ((P - pred) ** 2).sum()
        if sse < best_sse:
            best_k, best_sse = float(k), sse
    r2 = 1 - best_sse / ((P - P.mean()) ** 2).sum()
    return best_k, float(r2)


def plot_panel(ax, name, ell, P, r):
    """Measured windows vs the ramp-plateau node model."""
    ax.scatter(ell, P, s=3, alpha=0.25, color="steelblue", rasterized=True)
    xmax = max(float(ell.max()) * 1.05, r["rho_star"] * 1.3)
    ax.axhline(r["P_idle_w"], color="gray", ls=":", lw=1.2)
    ax.axhline(r["P_busy_w"], color="gray", ls=":", lw=1.2)
    ax.axvline(r["rho_star"], color="red", ls="--", lw=1.2)
    k = r["ell_power_knee"]
    p0, p1 = r["P_idle_w"], r["P_busy_w"]
    xs = np.linspace(0, xmax, 200)
    ax.plot(xs, p0 + (p1 - p0) * np.clip(xs / k, 0, 1), color="black", lw=1.8)
    ax.annotate(f"$R^2$={r['r2_ramp']:.2f}\nknee $\\ell$={k:.2f}",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    star = "" if r["rho_star_measured"] else "~"
    ax.set_title(f"{name}  ($\\rho^*$={star}{r['rho_star']:.2f})", fontsize=8.5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, max(float(P.max()), r["P_busy_w"]) * 1.1)
    ax.tick_params(labelsize=7)


def main():
    rows = []
    panels = []
    for path in sorted(glob.glob(os.path.join(IN_ROOT, "windows_*.npz"))):
        name = os.path.basename(path).replace("windows_", "").replace(".npz", "")
        d = np.load(path)
        P, f, g, itl = d["P"], d["f"], d["g"], d["itl"]

        F = float(f.max())
        G = float(g.max())
        ell = f / F + g / G

        knee = rho_star_from_itl(ell, itl)
        rho = knee if knee is not None else RHO_STAR_FALLBACK

        empty = (f + g) == 0
        p_idle = float(np.median(P[empty])) if empty.sum() >= 20 else float(P.min())

        at_setpoint = ell >= rho
        if at_setpoint.sum() < 30:  # sparse near the top: take the busiest 10%
            at_setpoint = ell >= np.quantile(ell, 0.9)
        p_busy = float(np.median(P[at_setpoint]))

        ell_pow, r2_ramp = fit_power_knee(ell, P, p_idle, p_busy)
        rows.append({
            "config": name,
            "P_idle_w": round(p_idle),
            "P_busy_w": round(p_busy),
            "rho_star": round(rho, 2),
            "rho_star_measured": knee is not None,
            "p_bar_w_per_node": round(p_busy / rho),
            "F_prefill_tps": round(F),
            "G_decode_tps": round(G),
            "ell_power_knee": round(ell_pow, 2),
            "r2_ramp": round(r2_ramp, 2),
        })
        panels.append((name, ell, P, rows[-1]))

    ncols = 5
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows))
    for ax, (name, ell, P, r) in zip(axes.flat, panels):
        plot_panel(ax, name, ell, P, r)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"node fraction $\ell$", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("power [W]", fontsize=8)
    fig.suptitle(r"Node power model per type: ramp from $P_{idle}$ to "
                 r"$P_{busy}$ at the power knee, then flat (black) — levels "
                 r"dotted, setpoint $\rho^*$ red, measured windows blue",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = os.path.join(FIG_ROOT, "operator_model_grid.png")
    fig.savefig(grid_path, dpi=130)
    plt.close(fig)

    out = os.path.join(IN_ROOT, "operator_table.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'config':38s} {'P_idle':>6s} {'P_busy':>6s} {'rho*':>5s} "
          f"{'p_bar':>6s} {'F':>7s} {'G':>5s}")
    for r in rows:
        star = " " if r["rho_star_measured"] else "~"
        print(f"{r['config']:38s} {r['P_idle_w']:6d} {r['P_busy_w']:6d} "
              f"{star}{r['rho_star']:4.2f} {r['p_bar_w_per_node']:6d} "
              f"{r['F_prefill_tps']:7d} {r['G_decode_tps']:5d}")
    print("\n(~ = sweep never pushed latency up; rho* = 0.8 by convention)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
