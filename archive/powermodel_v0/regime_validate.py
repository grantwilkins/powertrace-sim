"""Validate the power-per-regime model with the SAME honest metrics + split as
``validate.py`` so results are directly comparable to the energy-per-unit model.

Calibration uses ONLY the identification probes (CALIB_PROBES); held-out runs
(VAL_PROBES) are split SAME-FAMILY vs ZERO-SHOT. All held-out metrics are on the
DYNAMIC residual over BUSY bins only. Runs three regime variants per platform:
additive, additive+active-floor, and bottleneck.

Run on a compute node (not the login node):
    python -m powermodel.regime_validate --ledger $SCRATCH/ptsim/powermodel_ledger.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from powermodel import estimate as E
from powermodel import regime as R
from powermodel.ingest import load_ledger
from powermodel.validate import (CALIB_PROBES, VAL_PROBES, _probe_of,
                                 honest_run_metrics)

VARIANTS = (
    ("additive", False, "ADDITIVE  P=Pcmp*u_cmp + Pmem*u_mem"),
    ("additive", True, "ADDITIVE+FLOOR  + P_active*busy"),
    ("bottleneck", False, "BOTTLENECK  P=P_regime*max(u_cmp,u_mem)"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="powermodel/ledger.npz")
    args = ap.parse_args()

    led = load_ledger(args.ledger)
    y = led["power"].astype(np.float64)
    run_ids = led["run_id"]
    busy = led["busy"]

    def is_calib(rid):
        p = _probe_of(led, rid)
        return any(p.startswith(c) or c in p for c in CALIB_PROBES)

    def is_val(rid):
        p = _probe_of(led, rid)
        return any(c in p for c in VAL_PROBES)

    name_of = {m["run_id"]: m["name"] for m in led["meta"]}

    for hw_i, hw in enumerate([str(x) for x in led["hw_names"]]):
        hw_mask = led["hw_idx"] == hw_i
        if hw_mask.sum() == 0:
            continue
        runs = np.unique(run_ids[hw_mask])
        calib_runs = [r for r in runs if is_calib(r)]
        val_runs = [r for r in runs if is_val(r)]
        calib_mask = hw_mask & np.isin(run_ids, calib_runs)
        calib_fams = set(led["family_idx"][calib_mask].tolist())
        fam_of = {rid: int(led["family_idx"][run_ids == rid][0]) for rid in val_runs}

        print(f"\n{'='*78}\n{hw}: calib runs={len(calib_runs)} val runs={len(val_runs)} "
              f"calib_fams={sorted(calib_fams)}\n{'='*78}")

        for variant, floor, desc in VARIANTS:
            fit = R.calibrate(led, hw_i, train_mask=calib_mask,
                              variant=variant, floor=floor)
            X = fit["X"]
            theta, cov = fit["theta"], fit["cov"]
            pred = R.predict_full(fit)
            y_dyn = y - fit["standing"]

            tag_floor = "+floor" if floor else ""
            print(f"\n  ---- {variant}{tag_floor}: {desc}")
            print(f"       standing anchor: {fit['standing_per_gpu']:.1f} W/GPU")
            ins = R.metrics(y[calib_mask], pred[calib_mask])
            insd = R.metrics(y_dyn[calib_mask], (X[calib_mask] @ np.exp(theta)))
            print(f"       calib in-sample: R2(full)={ins['r2']:.4f} "
                  f"R2(dyn)={insd['r2']:.4f} RMSE={ins['rmse']:.1f}W")

            print(f"       {'coefficient':18s} {'MAP':>11s} {'prior':>10s} "
                  f"{'shrink':>7s} {'drift':>6s}  status")
            for row in E.identifiability(theta, cov, priors=fit["priors"]):
                print(f"       {R.LABELS[row['feature']]:18s} {row['value']:11.3g} "
                      f"{row['prior_mean']:10.3g} {row['shrink']:7.2f} "
                      f"{row['drift_sigma']:+6.1f}  {row['status']}")

            pvh = E.predictive(X, theta, cov,
                               E.het_sigma(y_dyn, pred - fit["standing"]))
            total_sd = pvh["total_sd"]
            for tag, sel in (("SAME-FAMILY",
                              [r for r in val_runs if fam_of[r] in calib_fams]),
                             ("ZERO-SHOT",
                              [r for r in val_runs if fam_of[r] not in calib_fams])):
                if not sel:
                    continue
                hm = honest_run_metrics(y, pred, fit["standing"], busy,
                                        total_sd, run_ids, sel)
                print(f"       HELD-OUT [{tag}] (dyn residual, busy bins; per-run)")
                print(f"         {'run':38s} {'bias%':>6s} {'|e|%':>5s} "
                      f"{'dNRMSE%':>7s} {'std':>5s} {'racf1':>6s} {'cov2':>5s}")
                for rid in sorted(hm):
                    m = hm[rid]
                    nm = str(name_of[rid])[:38]
                    print(f"         {nm:38s} {m['energy_bias']:+6.0f} "
                          f"{m['energy_abs']:5.0f} {m['dyn_nrmse']:7.0f} "
                          f"{m['std_ratio']:5.2f} {m['resid_acf1']:+6.2f} "
                          f"{m['cover2']:5.2f}")


if __name__ == "__main__":
    main()
