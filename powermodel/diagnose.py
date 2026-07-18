"""Per-run diagnostic: measured vs predicted power + per-term breakdown.

Reveals which regime (idle / light decode / saturated decode / prefill) the fit
diverges on, and how each physical term contributes. Run on a compute node:
    python -m powermodel.diagnose --ledger <ledger.npz> [--hw A100]
"""

from __future__ import annotations

import argparse

import numpy as np

from powermodel import model as M
from powermodel.ingest import load_ledger
from powermodel.priors import FEATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="powermodel/ledger.npz")
    ap.add_argument("--hw", default="A100")
    args = ap.parse_args()

    led = load_ledger(args.ledger)
    y = led["power"].astype(float)
    run_ids = led["run_id"]
    hw_i = list(map(str, led["hw_names"])).index(args.hw)

    def probe(rid):
        return str(led["probe_names"][led["probe_idx"][run_ids == rid][0]])

    runs = np.unique(run_ids[led["hw_idx"] == hw_i])
    calib = [r for r in runs if any(c in probe(r) for c in ("idle", "staircase"))]
    cm = (led["hw_idx"] == hw_i) & np.isin(run_ids, calib)

    fit = M.calibrate(led, hw_i, train_mask=cm)
    pred = M.predict_full(fit)
    c = np.exp(fit["theta"])
    X = fit["X"]
    st = fit["standing"]

    print("standing/gpu=%.1f W   coefficients:" % fit["standing_per_gpu"])
    for f, v in zip(FEATS, c):
        print("   %-8s = %.4g" % (f, v))
    hdr = "%-40s %-16s %7s %7s | per-term mean W" % ("run", "probe", "meas", "pred")
    print(hdr)
    for r in runs:
        sl = run_ids == r
        terms = {f: float(np.mean(X[sl, j] * c[j])) for j, f in enumerate(FEATS)}
        nm = [m["name"] for m in led["meta"] if m["run_id"] == r][0]
        used = "CALIB" if r in calib else "held "
        print("%-40s %-16s %7.0f %7.0f | st=%4.0f act=%4.0f flop=%5.0f hbm=%5.0f attn=%4.0f  [%s]"
              % (nm[:40], probe(r)[:16], np.mean(y[sl]), np.mean(pred[sl]),
                 np.mean(st[sl]), terms["active"], terms["flop"], terms["hbm"],
                 terms["attn"], used))


if __name__ == "__main__":
    main()
