"""MAP fit with physical priors + robust heteroscedastic likelihood.

Coefficients are parameterized as c = exp(theta) (positive by construction)
with log-normal priors centered on physically expected values (datasheet
compute efficiency, HBM energy/byte, idle power). Likelihood is Student-t
(nu=4) with state-dependent scale sigma_i = a + b * P_pred, so saturated
high-power bins don't dominate and transition spikes don't drag coefficients.
Per-family efficiency multipliers m = exp(phi) carry a tight prior toward 1.

The Laplace approximation at the MAP gives a posterior sd per coefficient;
comparing it to the prior sd labels each constant DATA-IDENTIFIED (data
narrowed the prior) or PRIOR-DOMINATED (the data cannot pin it down — e.g.
e_f_prefill on ShareGPT, e_kv at short context).

Run: uv run python feature-test/fit_map_priors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature-test"))

from final_model import FEATS, IDLE_FEATS, design  # noqa: E402
from fit_models import FEATURES, load_cache, metrics  # noqa: E402

OUT = Path("feature-test/results")
NU = 4.0
SIGMA_FLOOR = 15.0
PHI_PRIOR_SD = 0.10  # family multipliers ~ within +/-10% of 1

# (prior mean, prior sd in log space). Sources: GPU datasheets (idle, TDP/peak
# FLOPs), HBM ~60-200 pJ/B system level, NVLink ~80-240 pJ/B.
PRIORS = {
    "tp": (40.0, 0.30),
    "tp_link": (30.0, 0.70),
    "busy_tp": (30.0, 0.70),
    "sat_cmp_lo": (50.0, 1.00),
    "sat_cmp_hi": (50.0, 1.00),
    "sat_mem_lo": (50.0, 1.00),
    "sat_mem_hi": (50.0, 1.00),
    "flops_pre": (1.0e-12, 0.25),
    "flops_dec": (1.0e-12, 0.70),
    "w_read_pre": (1.5e-10, 0.50),
    "w_read_dec": (1.5e-10, 0.50),
    "kv_read": (1.5e-10, 0.30),
    "kv_write": (1.5e-10, 0.50),
    "comm": (2.0e-10, 0.50),
}


def map_fit(X, y, fams_local, idle_cols, sigma, theta0=None, phi_fams=None):
    """L-BFGS MAP fit. Returns (theta, phi_dict)."""
    p = X.shape[1]
    mu = np.array([np.log(PRIORS[f][0]) for f in FEATS])
    sd = np.array([PRIORS[f][1] for f in FEATS])
    ufams = sorted(set(phi_fams or []))
    nf = len(ufams)
    fam_pos = {f: i for i, f in enumerate(ufams)}
    fam_idx = np.array([fam_pos.get(f, -1) for f in fams_local])

    Xi, Xd = X[:, idle_cols], X[:, ~idle_cols]

    def unpack(z):
        return z[:p], z[p:]

    def obj(z):
        th, ph = unpack(z)
        c = np.exp(th)
        m = np.ones(y.size)
        if nf:
            mm = np.concatenate([np.exp(ph), [1.0]])
            m = mm[fam_idx]
        dyn = Xd @ c[~idle_cols]
        pred = Xi @ c[idle_cols] + m * dyn
        r = y - pred
        q = NU * sigma**2 + r**2
        nll = float(np.sum(0.5 * (NU + 1) * np.log1p(r**2 / (NU * sigma**2))))
        gpred = -(NU + 1) * r / q
        g_th = np.empty(p)
        g_th[idle_cols] = (gpred @ Xi) * c[idle_cols]
        g_th[~idle_cols] = ((gpred * m) @ Xd) * c[~idle_cols]
        g_ph = np.zeros(nf)
        for f, i in fam_pos.items():
            sel = fam_idx == i
            g_ph[i] = float(np.sum(gpred[sel] * dyn[sel])) * np.exp(ph[i])
        # priors
        nll += float(np.sum((th - mu) ** 2 / (2 * sd**2)))
        g_th += (th - mu) / sd**2
        if nf:
            nll += float(np.sum(ph**2 / (2 * PHI_PRIOR_SD**2)))
            g_ph += ph / PHI_PRIOR_SD**2
        return nll, np.concatenate([g_th, g_ph])

    z0 = np.concatenate([mu if theta0 is None else theta0, np.zeros(nf)])
    res = minimize(obj, z0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=500, ftol=1e-10))
    th, ph = unpack(res.x)
    return th, {f: float(np.exp(ph[i])) for f, i in fam_pos.items()}


def predict_map(X, theta, mult, fams_local, idle_cols):
    c = np.exp(theta)
    mvec = np.array([mult.get(f, 1.0) for f in fams_local])
    return X[:, idle_cols] @ c[idle_cols] + mvec * (X[:, ~idle_cols] @ c[~idle_cols])


def het_sigma(y, pred):
    """sigma_i = a + b*pred from robust regression of |resid| on pred."""
    r = np.abs(y - pred)
    A = np.column_stack([np.ones_like(pred), pred])
    w = 1.0 / np.maximum(pred, 100.0)
    coef, *_ = np.linalg.lstsq(A * w[:, None], r * w * 1.2533, rcond=None)
    return np.maximum(SIGMA_FLOOR, A @ coef)


def fit_two_stage(X, y, fams_local, idle_cols, theta0=None):
    sigma = np.full(y.size, 100.0)
    th, mult = map_fit(X, y, fams_local, idle_cols, sigma, theta0=theta0,
                       phi_fams=list(set(fams_local)))
    for _ in range(2):
        pred = predict_map(X, th, mult, fams_local, idle_cols)
        sigma = het_sigma(y, pred)
        th, mult = map_fit(X, y, fams_local, idle_cols, sigma, theta0=th,
                           phi_fams=list(set(fams_local)))
    return th, mult, sigma


def laplace_sd(X, y, theta, mult, fams_local, idle_cols, sigma):
    """Posterior sd of theta via Gauss-Newton Laplace approximation."""
    c = np.exp(theta)
    mvec = np.array([mult.get(f, 1.0) for f in fams_local])
    J = X.copy()
    J[:, ~idle_cols] *= mvec[:, None]
    J = J * c[None, :]  # d pred / d theta
    w = (NU + 1) / (NU * sigma**2)  # GN weight of t-likelihood near mode
    H = (J * w[:, None]).T @ J
    H += np.diag([1.0 / PRIORS[f][1] ** 2 for f in FEATS])
    cov = np.linalg.inv(H)
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_cache()
    y = d["power"].astype(np.float64)
    run_ids = d["run_id"]
    fams = d["family_idx"]
    fam_names = [str(x) for x in d["family_names"]]
    idle_cols = np.array([f in IDLE_FEATS for f in FEATS])

    out_json = {}
    for hw_i, hw in enumerate([str(x) for x in d["hw_names"]]):
        hw_mask = d["hw_idx"] == hw_i
        if hw_mask.sum() == 0:
            continue
        X = design(d, hw, run_ids)
        Xh, yh, fh = X[hw_mask], y[hw_mask], fams[hw_mask]

        # ---- full fit + identifiability
        th, mult, sigma = fit_two_stage(Xh, yh, fh, idle_cols)
        post_sd = laplace_sd(Xh, yh, th, mult, fh, idle_cols, sigma)
        pred = predict_map(Xh, th, mult, fh, idle_cols)
        ins = metrics(yh, pred)

        print(f"\n######## {hw} MAP fit (Student-t nu={NU:g}, heteroscedastic)")
        print(f"  in-sample: R2={ins['r2']:.4f} RMSE={ins['rmse']:.1f}W MAPE={ins['mape']:.1f}%")
        print(f"  {'coefficient':30s} {'MAP':>10s} {'prior':>10s} {'shrink':>7s} {'drift':>6s}  status")
        id_rows = []
        for j, f in enumerate(FEATS):
            shrink = post_sd[j] / PRIORS[f][1]
            drift = (th[j] - np.log(PRIORS[f][0])) / PRIORS[f][1]
            status = ("DATA-IDENTIFIED" if shrink < 0.3
                      else "partial" if shrink < 0.7 else "PRIOR-DOMINATED")
            print(f"  {FEATURES[f][0]:30s} {np.exp(th[j]):10.3g} {PRIORS[f][0]:10.3g} "
                  f"{shrink:7.2f} {drift:+6.1f}  {status}")
            id_rows.append(dict(hardware=hw, feature=f, label=FEATURES[f][0],
                                map_value=float(np.exp(th[j])),
                                prior_mean=PRIORS[f][0],
                                posterior_sd_log=float(post_sd[j]),
                                prior_sd_log=PRIORS[f][1],
                                shrink=float(shrink), drift_sigma=float(drift),
                                status=status))
        print("  family multipliers: " + ", ".join(
            f"{fam_names[k]}={v:.3f}" for k, v in sorted(mult.items())))

        # ---- grouped 5-fold CV
        rng = np.random.default_rng(0)
        uruns = np.unique(run_ids[hw_mask])
        fold_of = dict(zip(uruns.tolist(), rng.integers(0, 5, size=uruns.size).tolist()))
        fold_arr = np.array([fold_of.get(r, -1) for r in run_ids])
        ys, ps = [], []
        for f in range(5):
            tr = hw_mask & (fold_arr != f)
            te = hw_mask & (fold_arr == f)
            tht, mt, _ = fit_two_stage(X[tr], y[tr], fams[tr], idle_cols, theta0=th)
            ys.append(y[te])
            ps.append(predict_map(X[te], tht, mt, fams[te], idle_cols))
        cv = metrics(np.concatenate(ys), np.concatenate(ps))
        print(f"  CV(5-fold by run): R2={cv['r2']:.4f} RMSE={cv['rmse']:.1f}W MAPE={cv['mape']:.1f}%")

        # ---- LOMO
        lomo = {}
        for fi in np.unique(fh):
            te = hw_mask & (fams == fi)
            tr = hw_mask & (fams != fi)
            if len(np.unique(fams[tr])) < 2:
                continue
            tht, mt, _ = fit_two_stage(X[tr], y[tr], fams[tr], idle_cols, theta0=th)
            mt.pop(int(fi), None)
            predt = predict_map(X[te], tht, mt, fams[te], idle_cols)
            errs = [abs(np.mean(predt[run_ids[te] == r]) - np.mean(y[te][run_ids[te] == r]))
                    / max(np.mean(y[te][run_ids[te] == r]), 1.0) * 100
                    for r in np.unique(run_ids[te])]
            lomo[fam_names[fi]] = float(np.median(errs))
        print("  LOMO run-mean err: " + ", ".join(f"{k}={v:.1f}%" for k, v in lomo.items()))

        # plateau/gap range check on example run
        mn = [str(x) for x in d["model_names"]]
        is70 = np.array([mn[i] == "llama-3-70b" for i in d["model_idx"]])
        sel = hw_mask & is70 & (d["tp"] == 8) & (d["rate"] == 0.5)
        if sel.sum():
            rid = np.unique(run_ids[sel])[0]
            mm = run_ids == rid
            pr = predict_map(X[mm], th, mult, fams[mm], idle_cols)
            busy = d["batch"][mm] > 2
            gap = d["batch"][mm] == 0
            print(f"  example run plateau: meas {y[mm][busy].mean():.0f}W pred {pr[busy].mean():.0f}W"
                  f" | gaps: meas {y[mm][gap].mean():.0f}W pred {pr[gap].mean():.0f}W")

        pd.DataFrame(id_rows).to_csv(OUT / f"map_identifiability_{hw}.csv", index=False)
        out_json[hw] = dict(
            coefficients={f: float(np.exp(th[j])) for j, f in enumerate(FEATS)},
            posterior_sd_log={f: float(post_sd[j]) for j, f in enumerate(FEATS)},
            family_multipliers={fam_names[k]: v for k, v in mult.items()},
            metrics=dict(cv=cv, in_sample=ins, lomo=lomo),
        )

    with open(OUT / "map_coefficients.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nWrote {OUT}/map_coefficients.json and map_identifiability_*.csv")


if __name__ == "__main__":
    main()
