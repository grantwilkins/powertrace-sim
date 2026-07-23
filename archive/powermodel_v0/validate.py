"""Calibrate on idle+staircases, validate on never-fit realistic runs.

Calibration uses ONLY the identification probes (idle anchor + prefill/decode
staircases). Validation is on runs never seen during the fit (validate + agentic).
Primary metric: held-out run-energy error % + 30 s-window R². Also prints the
per-coefficient identifiability table and writes coefficients + a trace plot.

Run on a compute node (not the login node):
    python -m powermodel.validate --ledger powermodel/ledger.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from powermodel import estimate as E
from powermodel import model as M
from powermodel.ingest import load_ledger
from powermodel.priors import FEATS, LABELS

CALIB_PROBES = ("idle", "prefill_staircase", "decode_staircase", "mixed_grid")
VAL_PROBES = ("validate", "agentic")


def _probe_of(ledger, rid):
    return str(ledger["probe_names"][ledger["probe_idx"][ledger["run_id"] == rid][0]])


def window_r2(y, pred, run_ids, win=30):
    keys = []
    for rid in np.unique(run_ids):
        sl = run_ids == rid
        idx = np.arange(sl.sum())
        keys.append(rid * 100000 + idx // win)
    key = np.concatenate(keys)
    order = np.argsort(run_ids, kind="stable")
    ys, ps = {}, {}
    for k, yi, pi in zip(key, y, pred):
        ys.setdefault(k, []).append(yi)
        ps.setdefault(k, []).append(pi)
    ym = np.array([np.mean(v) for v in ys.values()])
    pm = np.array([np.mean(ps[k]) for k in ys])
    return M.metrics(ym, pm)


def run_energy_errors(y, pred, run_ids, sel_runs):
    errs = {}
    for rid in sel_runs:
        sl = run_ids == rid
        if not sl.any():
            continue
        meas, prd = float(np.mean(y[sl])), float(np.mean(pred[sl]))
        errs[rid] = abs(prd - meas) / max(meas, 1.0) * 100
    return errs


def _lag1_acf(x):
    """Lag-1 autocorrelation of a 1-D series (residual-whiteness diagnostic)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    x = x - x.mean()
    denom = float(np.sum(x * x))
    return float(np.sum(x[1:] * x[:-1]) / denom) if denom > 0 else float("nan")


def honest_run_metrics(y, pred, standing, busy, total_sd, run_ids, sel_runs):
    """Per-run metrics that strip the memorized standing pedestal and idle padding
    (critic guidance): everything on the DYNAMIC residual over BUSY bins only.

    Returns per-run dict: energy_bias% (signed), energy_abs%, dyn_nrmse (1 Hz),
    std_ratio (pred/meas dynamic), resid_acf1 (whiteness), cover2 (fraction within
    +/-2 sigma). Headline these for ZERO-SHOT.
    """
    out = {}
    yd, pd = y - standing, pred - standing       # dynamic component
    for rid in sel_runs:
        sl = (run_ids == rid) & (np.asarray(busy) > 0)
        if sl.sum() < 5:
            continue
        meas, prd = float(np.mean(y[sl])), float(np.mean(pred[sl]))
        ydn, pdn = yd[sl], pd[sl]
        dyn_mean = max(float(np.mean(ydn)), 1.0)
        out[rid] = dict(
            energy_bias=(prd - meas) / max(meas, 1.0) * 100,
            energy_abs=abs(prd - meas) / max(meas, 1.0) * 100,
            dyn_nrmse=float(np.sqrt(np.mean((pdn - ydn) ** 2))) / dyn_mean * 100,
            std_ratio=float(np.std(pdn) / max(np.std(ydn), 1e-6)),
            resid_acf1=_lag1_acf(ydn - pdn),
            cover2=float(np.mean(np.abs(y[sl] - pred[sl]) <= 2 * total_sd[sl])),
            n=int(sl.sum()),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="powermodel/ledger.npz")
    ap.add_argument("--out", default="powermodel/results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    led = load_ledger(args.ledger)
    y = led["power"].astype(np.float64)
    run_ids = led["run_id"]
    probe_idx = led["probe_idx"]
    probe_names = [str(x) for x in led["probe_names"]]

    def is_calib(rid):
        p = _probe_of(led, rid)
        return any(p.startswith(c) or c in p for c in CALIB_PROBES)

    def is_val(rid):
        p = _probe_of(led, rid)
        return any(c in p for c in VAL_PROBES)

    by_hw = {}
    for hw_i, hw in enumerate([str(x) for x in led["hw_names"]]):
        hw_mask = led["hw_idx"] == hw_i
        if hw_mask.sum() == 0:
            continue
        runs = np.unique(run_ids[hw_mask])
        calib_runs = [r for r in runs if is_calib(r)]
        val_runs = [r for r in runs if is_val(r)]
        calib_mask = hw_mask & np.isin(run_ids, calib_runs)

        fit = M.calibrate(led, hw_i, train_mask=calib_mask)
        by_hw[hw] = fit
        X = fit["X"]
        theta, cov, sigma = fit["theta"], fit["cov"], fit["sigma"]
        pred = M.predict_full(fit)             # anchored standing + dynamic
        y_dyn = y - fit["standing"]            # dynamic residual (for sigma/attribution)

        print(f"\n######## {hw}: calib runs={len(calib_runs)} val runs={len(val_runs)}")
        print(f"  standing anchor: {fit['standing_per_gpu']:.1f} W/GPU")
        ins = M.metrics(y[calib_mask], pred[calib_mask])
        print(f"  calib in-sample: R2={ins['r2']:.4f} RMSE={ins['rmse']:.1f}W MAPE={ins['mape']:.1f}%")

        # identifiability
        print(f"  {'coefficient':22s} {'MAP':>11s} {'prior':>10s} {'shrink':>7s} {'drift':>6s}  status")
        for row in E.identifiability(theta, cov):
            print(f"  {LABELS[row['feature']]:22s} {row['value']:11.3g} "
                  f"{row['prior_mean']:10.3g} {row['shrink']:7.2f} "
                  f"{row['drift_sigma']:+6.1f}  {row['status']}")

        # Honest held-out metrics (critic guidance): on the DYNAMIC residual,
        # BUSY bins only (strip the memorized standing pedestal + idle padding),
        # split SAME-FAMILY vs ZERO-SHOT (cross-family). Per-run, named — never a
        # summary over 1-2 runs as if it were a distribution.
        sigma_full = sigma_for(X[hw_mask], y_dyn[hw_mask], theta)
        pvh = E.predictive(X, theta, cov, E.het_sigma(y - fit["standing"], pred - fit["standing"]))
        total_sd = pvh["total_sd"]
        busy = led["busy"]
        calib_fams = set(led["family_idx"][calib_mask].tolist())
        fam_of = {rid: int(led["family_idx"][run_ids == rid][0]) for rid in val_runs}
        for tag, sel in (("SAME-FAMILY", [r for r in val_runs if fam_of[r] in calib_fams]),
                         ("ZERO-SHOT", [r for r in val_runs if fam_of[r] not in calib_fams])):
            if not sel:
                continue
            hm = honest_run_metrics(y, pred, fit["standing"], busy, total_sd, run_ids, sel)
            print(f"  HELD-OUT [{tag}]  (dynamic residual, busy bins; per-run, named)")
            print(f"    {'run':40s} {'bias%':>6s} {'|e|%':>5s} {'dNRMSE%':>7s} {'std':>5s} {'racf1':>6s} {'cov2':>5s}")
            for rid in sorted(hm):
                m = hm[rid]
                nm = [mm['name'] for mm in led['meta'] if mm['run_id'] == rid][0]
                print(f"    {nm[:40]:40s} {m['energy_bias']:+6.0f} {m['energy_abs']:5.0f} "
                      f"{m['dyn_nrmse']:7.0f} {m['std_ratio']:5.2f} {m['resid_acf1']:+6.2f} {m['cover2']:5.2f}")

        # predictive intervals + factor attribution (on the dynamic residual)
        pv = E.predictive(X[hw_mask], theta, cov,
                          sigma_for(X[hw_mask], y_dyn[hw_mask], theta))
        share = pv["factor_var_mean"]
        tot = float(np.sum(share)) or 1.0
        print("  predictive variance share by factor:")
        for f, s in sorted(zip(FEATS, share), key=lambda kv: -kv[1])[:6]:
            print(f"    {LABELS[f]:22s} {100*s/tot:5.1f}%")

        _plot(led, hw_i, X, theta, pred, val_runs, out, hw)

    M.save_coefficients(out / "coefficients.json", by_hw)
    print(f"\nWrote {out}/coefficients.json")


def sigma_for(X, y, theta):
    return E.het_sigma(y, X @ np.exp(theta))


def _plot(led, hw_i, X, theta, pred, val_runs, out, hw):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not val_runs:
        return
    rid = val_runs[0]
    sl = led["run_id"] == rid
    y = led["power"][sl]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    t = np.arange(sl.sum())
    ax.plot(t, y, lw=0.8, label="measured")
    ax.plot(t, pred[sl], lw=1.2, label="predicted")
    ax.set_xlabel("time (s)"); ax.set_ylabel("node power (W)")
    name = [m["name"] for m in led["meta"] if m["run_id"] == rid][0]
    ax.set_title(f"{hw} held-out {name}"); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"trace_{hw}.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
