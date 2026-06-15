"""Saturating refit of node power on the windowed raw traces.

Consumes the per-window arrays produced by two_price_fit.py
(results/two_price_fit/windows_<config>.npz: P, f, g, itl, ttft_pt) and fits,
per node type:

    P(f, g) = P0 + dP * w / (1 + w),   w = a*f + b*g     (Michaelis-Menten)

so marginal per-token prices at low load are (dP*a, dP*b) and power saturates
at P0 + dP.  From the fit and the latency data it derives:

  - F, G       : per-phase capacities = max sustained windowed rates (p99.5),
                 so ell = f/F + g/G is a true node fraction in [0, ~1]
  - power knee : ell where fitted P reaches P0 + 0.8 dP (early knee)
  - latency knee: ell where binned median ITL departs 1.25x its minimum
                 (late knee); fallback rho* = 0.8 if the sweep never saturates
  - s_plat     : plateau slope dP/d(ell) at rho* (fixed-node marginal price)
  - p_amort    : pi(rho*)/rho* (autoscaler-amortized price per node-unit)
  - per-busy-second phase prices p_pre = dP*a*F, p_dec = dP*b*G

Outputs results/two_price_fit/saturating_summary.csv and a three-panel figure
per config in figures/two_price_fit/sat_<config>.png.

Usage:  uv run python scripts/eval/saturating_fit.py
"""

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

IN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "two_price_fit")
FIG_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "two_price_fit")

LATENCY_KNEE_FACTOR = 1.25
POWER_KNEE_FRAC = 0.8  # power knee = ell where P reaches P0 + this * dP
RHO_STAR_FALLBACK = 0.8


def linear_fit(P, f, g):
    X = np.column_stack([np.ones_like(f), f, g])
    coef, _, _, _ = np.linalg.lstsq(X, P, rcond=None)
    r = P - X @ coef
    r2 = 1 - (r**2).sum() / ((P - P.mean()) ** 2).sum()
    return coef, r2


def mm_fit(P, f, g):
    """P = P0 + dP * w/(1+w), w = a f + b g, all params >= 0."""
    coef, _ = linear_fit(P, f, g)
    P0_init = np.quantile(P, 0.02)
    dP_init = max(np.quantile(P, 0.99) - P0_init, 1.0)
    L = np.maximum(coef[1], 1e-6) * f + np.maximum(coef[2], 1e-6) * g
    k = max(np.median(L[L > 0]), 1.0)  # half-saturation at the median load
    x0 = np.array([P0_init, dP_init,
                   max(coef[1], 1e-6) / k, max(coef[2], 1e-6) / k])

    def model(p):
        w = p[2] * f + p[3] * g
        return p[0] + p[1] * w / (1 + w)

    res = least_squares(lambda p: model(p) - P, x0,
                        bounds=([0, 0, 0, 0], [np.inf] * 4),
                        x_scale=[100, 100, x0[2] or 1e-3, x0[3] or 1e-3])
    p = res.x
    pred = model(p)
    r2 = 1 - ((P - pred) ** 2).sum() / ((P - P.mean()) ** 2).sum()
    return p, pred, r2


def latency_knee(ell, lat, n_bins=20):
    ok = np.isfinite(ell) & np.isfinite(lat) & (ell > 0)
    if ok.sum() < 50:
        return None
    ell, lat = ell[ok], lat[ok]
    qs = np.unique(np.quantile(ell, np.linspace(0, 1, n_bins + 1)))
    centers, medians = [], []
    for i in range(len(qs) - 1):
        sel = (ell >= qs[i]) & (ell <= qs[i + 1])
        if sel.sum() >= 5:
            centers.append(np.median(ell[sel]))
            medians.append(np.median(lat[sel]))
    if len(centers) < 4:
        return None
    centers, medians = np.asarray(centers), np.asarray(medians)
    i_min = int(np.argmin(medians))
    above = np.where(medians[i_min:] > LATENCY_KNEE_FACTOR * medians[i_min])[0]
    return float(centers[i_min + above[0]]) if len(above) else None


def analyze(name, d):
    P, f, g = d["P"], d["f"], d["g"]
    itl = d["itl"]

    # Capacities from the traces' own max sustained windowed rates.
    F = float(np.quantile(f[f > 0], 0.995))
    G = float(np.quantile(g[g > 0], 0.995))
    ell = f / F + g / G
    f_share = np.where(f + g > 0, f / np.maximum(f + g, 1e-9), 0.0)

    lin_coef, r2_lin = linear_fit(P, f, g)
    (P0, dP, a, b), pred, r2_sat = mm_fit(P, f, g)
    # Token prices from the LINEAR fit: operating-range averages.  The MM
    # low-load marginals (dP*a, dP*b) encode step steepness, not energy, and
    # collinearity (corr(f,g)~0.8) drives a -> 0 in the MM fit.
    c1, c2 = float(lin_coef[1]), float(lin_coef[2])

    # Fitted P along the ray of mean phase mix, as a function of ell.
    mix_f = f.mean() / max(f.mean() / F + g.mean() / G, 1e-9)  # f at ell=1
    mix_g = g.mean() / max(f.mean() / F + g.mean() / G, 1e-9)

    def P_of_ell(x):
        w = a * mix_f * x + b * mix_g * x
        return P0 + dP * w / (1 + w)

    def slope_of_ell(x):
        wp = a * mix_f + b * mix_g
        w = wp * x
        return dP * wp / (1 + w) ** 2

    # Early knee: power reaches P0 + POWER_KNEE_FRAC * dP  ->  w = frac/(1-frac)
    w_knee = POWER_KNEE_FRAC / (1 - POWER_KNEE_FRAC)
    ell_pow = w_knee / max(a * mix_f + b * mix_g, 1e-12)

    # Late knee: ITL departure.  rho* sits at the latency knee (or fallback).
    ell_lat = latency_knee(ell, itl)
    knee_found = ell_lat is not None
    rho_star = ell_lat if knee_found else RHO_STAR_FALLBACK

    s_plat = float(slope_of_ell(rho_star))          # fixed-node price [W/ell]
    p_amort = float(P_of_ell(rho_star) / rho_star)  # autoscaled price [W/ell]

    summary = {
        "config": name,
        "n_windows": len(P),
        "F_prefill_tps": F,
        "G_decode_tps": G,
        "P0_w": float(P0),
        "P_max_w": float(P0 + dP),
        "c1_j_per_prefill_tok": c1,   # linear-fit operating-range averages
        "c2_j_per_decode_tok": c2,
        "c1_over_c2": c1 / c2 if c2 > 0 else np.nan,
        "p_pre_w_per_busy_s": c1 * F,
        "p_dec_w_per_busy_s": c2 * G,
        "r2_linear": float(r2_lin),
        "r2_saturating": float(r2_sat),
        "ell_power_knee": float(ell_pow),
        "ell_latency_knee": float(ell_lat) if knee_found else np.nan,
        "latency_knee_found": knee_found,
        "rho_star": float(rho_star),
        "s_plat_w_per_ell": s_plat,
        "p_amort_w_per_ell": p_amort,
        "amort_over_plat": float(p_amort / s_plat) if s_plat > 0 else np.inf,
    }
    arrays = {"ell": ell, "P": P, "pred": pred, "f_share": f_share, "itl": itl,
              "P_of_ell": P_of_ell, "rho_star": rho_star, "ell_pow": ell_pow,
              "ell_lat": ell_lat}
    return summary, arrays


def make_figure(name, s, a):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    sc = ax.scatter(a["ell"], a["P"], c=a["f_share"], s=5, alpha=0.35,
                    cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
    xs = np.linspace(0, max(a["ell"].max(), 1.05), 200)
    ax.plot(xs, a["P_of_ell"](xs), "k-", lw=2, label="saturating fit")
    ax.axvline(a["ell_pow"], color="purple", ls=":", lw=1.5,
               label=f"power knee $\\ell$={a['ell_pow']:.2f}")
    if a["ell_lat"] is not None:
        ax.axvline(a["ell_lat"], color="r", ls="--", lw=1.5,
                   label=f"latency knee $\\ell$={a['ell_lat']:.2f}")
    ax.annotate(
        f"$s_{{plat}}$={s['s_plat_w_per_ell']:.0f} W/$\\ell$\n"
        f"$\\bar p_{{amort}}$={s['p_amort_w_per_ell']:.0f} W/$\\ell$\n"
        f"ratio {s['amort_over_plat']:.1f}x",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_xlabel(r"node fraction  $\ell = f/F + g/G$")
    ax.set_ylabel("measured node power [W]")
    ax.set_title(f"{name}\n$R^2$: linear {s['r2_linear']:.2f} "
                 f"$\\to$ saturating {s['r2_saturating']:.2f}")
    ax.legend(fontsize=8, loc="center right")
    fig.colorbar(sc, ax=ax, label="prefill share of tokens")

    ax = axes[1]
    resid = a["P"] - a["pred"]
    ax.scatter(a["ell"], resid, s=4, alpha=0.3, rasterized=True)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("residual [W]")
    ax.set_title(f"saturating-fit residuals (RMSE "
                 f"{np.sqrt((resid**2).mean()):.0f} W)")

    ax = axes[2]
    ok = np.isfinite(a["itl"])
    ax.scatter(a["ell"][ok], a["itl"][ok], s=4, alpha=0.3, color="tab:green",
               rasterized=True)
    ax.axvline(a["ell_pow"], color="purple", ls=":", lw=1.5, label="power knee")
    if a["ell_lat"] is not None:
        ax.axvline(a["ell_lat"], color="r", ls="--", lw=1.5,
                   label="latency knee")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("mean ITL [ms]")
    ax.set_yscale("log")
    ax.set_title("latency vs node fraction")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(FIG_ROOT, f"sat_{name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main():
    summaries = []
    for path in sorted(glob.glob(os.path.join(IN_ROOT, "windows_*.npz"))):
        name = os.path.basename(path).replace("windows_", "").replace(".npz", "")
        d = np.load(path)
        s, arrays = analyze(name, d)
        summaries.append(s)
        fig_path = make_figure(name, s, arrays)
        print(f"{name:38s} R2 {s['r2_linear']:.2f}->{s['r2_saturating']:.2f} | "
              f"P0={s['P0_w']:.0f} Pmax={s['P_max_w']:.0f} W | "
              f"knees: pow {s['ell_power_knee']:.2f}, "
              f"lat {s['ell_latency_knee'] if s['latency_knee_found'] else float('nan'):.2f} | "
              f"s_plat={s['s_plat_w_per_ell']:.0f}, "
              f"p_amort={s['p_amort_w_per_ell']:.0f} W/ell "
              f"({s['amort_over_plat']:.1f}x) | "
              f"p_pre/s={s['p_pre_w_per_busy_s']:.0f}, "
              f"p_dec/s={s['p_dec_w_per_busy_s']:.0f} W")
    if summaries:
        out = os.path.join(IN_ROOT, "saturating_summary.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
