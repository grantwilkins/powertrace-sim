"""Residual diagnostics for the best ledger model.

Where does the remaining error live (family / TP / rate / load state), and how
much of it is irreducible per-bin power noise (stochastic fluctuation within
steady state) that a mean model cannot explain?

Run: uv run python feature-test/diagnose_residuals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fit_models import MODELS, build_design_full, ema_by_run, load_cache, metrics  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

BEST = {"A100": ("M7_full", 0.7), "H100": ("M7_full", 0.5)}


def main():
    d = load_cache()
    y = np.asarray(d["power"], dtype=np.float64)
    run_ids = d["run_id"]
    fam_names = [str(x) for x in d["family_names"]]

    for hw_i, hw in enumerate([str(x) for x in d["hw_names"]]):
        hw_mask = d["hw_idx"] == hw_i
        if hw_mask.sum() == 0:
            continue
        mname, alpha = BEST[hw]
        X = build_design_full(d, MODELS[mname], alpha=alpha)
        coef, _ = nnls(X[hw_mask], y[hw_mask])
        pred = X @ coef
        resid = y - pred

        print(f"\n######## {hw}  ({mname}, lag alpha={alpha})")
        m = metrics(y[hw_mask], pred[hw_mask])
        print(f"  overall: R2={m['r2']:.4f} RMSE={m['rmse']:.1f}W MAPE={m['mape']:.1f}%")

        # error by family
        df = pd.DataFrame(dict(
            fam=[fam_names[i] for i in d["family_idx"][hw_mask]],
            tp=d["tp"][hw_mask].astype(int), rate=d["rate"][hw_mask],
            batch=d["batch"][hw_mask], y=y[hw_mask], r=resid[hw_mask],
        ))
        g = df.groupby("fam").agg(
            rmse=("r", lambda x: float(np.sqrt(np.mean(x**2)))),
            bias=("r", "mean"), p_mean=("y", "mean"), n=("r", "size"))
        print("  by family:\n" + g.round(1).to_string())
        g = df.groupby("tp").agg(
            rmse=("r", lambda x: float(np.sqrt(np.mean(x**2)))),
            bias=("r", "mean"), p_mean=("y", "mean"))
        print("  by TP:\n" + g.round(1).to_string())

        # irreducible noise floor: within-run std of power around its own
        # short EMA, in saturated bins (batch above per-run 75th pct)
        sm = ema_by_run(y, run_ids, 0.5)
        hf = y - sm
        sat = hw_mask & (d["batch"] > np.quantile(d["batch"][hw_mask], 0.75))
        print(f"  intrinsic HF power noise (steady, batch>q75): "
              f"std={float(np.std(hf[sat])):.1f}W  vs model RMSE {m['rmse']:.1f}W")

        # error at 5s aggregation (energy view)
        for win in (5, 30):
            kb = (np.arange(y.size) // win)
            key = kb * 10000 + run_ids  # unique per run+window
            dfa = pd.DataFrame(dict(k=key[hw_mask], y=y[hw_mask], p=pred[hw_mask]))
            a = dfa.groupby("k").mean()
            ma = metrics(a["y"].values, a["p"].values)
            print(f"  {win}s-mean: R2={ma['r2']:.4f} RMSE={ma['rmse']:.1f}W MAPE={ma['mape']:.1f}%")


if __name__ == "__main__":
    main()
