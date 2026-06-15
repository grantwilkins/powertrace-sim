"""Final first-principles power model: fit, evaluate, attribute, and export.

Model (per hardware platform, NNLS so every term is a non-negative power):
    P_node(t) = [idle]    p_idle * TP  +  p_link * TP * 1[TP>1]
              + [DVFS]    sum_k p_sat_k * TP * (1 - exp(-util/u0_k))  + p_active * TP * 1[busy]
              + [compute] e_f_pre * FLOPs_pre/s + e_f_dec * FLOPs_dec/s
              + [memory]  e_w_pre * W_bytes_pre/s + e_w_dec * W_bytes_dec/s
                          + e_kv_r * KV_read/s + e_kv_w * KV_write/s
              + [fabric]  e_comm * NVLink_bytes/s
    then a causal lag filter (meter averaging + power dynamics) and a
    per-model-family efficiency multiplier on the dynamic part (kernel/MFU
    differences; ~1.0 by construction, default 1.0 for unseen models).

Run: uv run python feature-test/final_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "feature-test"))

from fit_models import (  # noqa: E402
    FEATURES, HBM_BW, PEAK_FLOPS, ema_by_run, load_cache, metrics,
)

OUT = Path("feature-test/results")
FEATS = ["tp", "tp_link", "busy_tp", "sat_cmp_lo", "sat_cmp_hi", "sat_mem_lo",
         "sat_mem_hi", "flops_pre", "flops_dec", "w_read_pre", "w_read_dec",
         "kv_read", "kv_write", "comm"]
IDLE_FEATS = {"tp", "tp_link"}
PREFILL_KEYS = ("pre_tok", "flops_pre", "w_read_pre")
DECODE_KEYS = ("dec_tok", "flops_dec", "w_read_dec", "kv_read", "batch")


def ma_by_run(x, run_ids, k):
    out = np.empty_like(x, dtype=np.float64)
    bnd = np.flatnonzero(np.diff(run_ids)) + 1
    starts = np.concatenate([[0], bnd, [x.size]])
    kern = np.ones(k) / k
    for s, e in zip(starts[:-1], starts[1:]):
        seg = x[s:e]
        pad = np.concatenate([np.full(k - 1, seg[0]), seg])
        out[s:e] = np.convolve(pad, kern, mode="valid")
    return out


LAGS = {
    "A100": lambda c, rid: ema_by_run(c, rid, 0.6),
    "H100": lambda c, rid: ema_by_run(ma_by_run(c, rid, 2), rid, 0.7),
}


def recompute_derived(d):
    d = dict(d)
    d["flops_pre"] = 2.0 * d["n_active"] * d["pre_tok"]
    d["flops_dec"] = 2.0 * d["n_active"] * d["dec_tok"]
    d["busy"] = ((d["pre_tok"] + d["dec_tok"]) > 0).astype(np.float64)
    bw = np.where(d["hw_idx"] == 0, HBM_BW[0], HBM_BW[1])
    pf = np.where(d["hw_idx"] == 0, PEAK_FLOPS[0], PEAK_FLOPS[1])
    d["util_mem"] = np.clip((d["w_read"] + d["kv_read"]) / (d["tp"] * bw), 0.0, 1.5)
    d["util_cmp"] = np.clip((d["flops_pre"] + d["flops_dec"]) / (d["tp"] * pf), 0.0, 1.5)
    return d


def zero_phase(d, phase):
    """Return a copy of the cache with one phase's work removed."""
    d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    tok = d["pre_tok"] + d["dec_tok"]
    if phase == "prefill":
        share = np.divide(d["dec_tok"], tok, out=np.zeros_like(tok), where=tok > 0)
        for k in PREFILL_KEYS:
            d2[k] = np.zeros_like(d[k])
        d2["kv_write"] = d["dec_tok"] * np.divide(
            d["kv_write"], tok, out=np.zeros_like(tok), where=tok > 0)
    elif phase == "decode":
        share = np.divide(d["pre_tok"], tok, out=np.zeros_like(tok), where=tok > 0)
        for k in DECODE_KEYS:
            d2[k] = np.zeros_like(d[k])
        d2["kv_write"] = d["pre_tok"] * np.divide(
            d["kv_write"], tok, out=np.zeros_like(tok), where=tok > 0)
    else:  # both
        share = np.zeros_like(tok)
        for k in PREFILL_KEYS + DECODE_KEYS:
            d2[k] = np.zeros_like(d[k])
        d2["kv_write"] = np.zeros_like(d["kv_write"])
    d2["comm"] = d["comm"] * share
    d2["w_read"] = d2["w_read_pre"] + d2["w_read_dec"]
    return recompute_derived(d2)


def design(d, hw, run_ids):
    cols = [np.asarray(FEATURES[f][1](d), dtype=np.float64) for f in FEATS]
    lag = LAGS[hw]
    return np.column_stack([lag(c, run_ids) for c in cols])


def fit_hw(X, y, mask, fams, idle_cols):
    coef, _ = nnls(X[mask], y[mask])
    dyn = X[:, ~idle_cols] @ coef[~idle_cols]
    idle = X[:, idle_cols] @ coef[idle_cols]
    mult = {}
    for fi in np.unique(fams[mask]):
        m = mask & (fams == fi) & (dyn > 1.0)
        mult[int(fi)] = float(np.clip(
            np.sum(dyn[m] * (y[m] - idle[m])) / max(np.sum(dyn[m] ** 2), 1e-9), 0.5, 2.0))
    return coef, mult


def predict(X, coef, mult, fams, idle_cols):
    dyn = X[:, ~idle_cols] @ coef[~idle_cols]
    idle = X[:, idle_cols] @ coef[idle_cols]
    mvec = np.array([mult.get(int(fi), 1.0) for fi in fams])
    return idle + mvec * dyn


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_cache()
    y = d["power"].astype(np.float64)
    run_ids = d["run_id"]
    fams = d["family_idx"]
    fam_names = [str(x) for x in d["family_names"]]
    idle_cols = np.array([f in IDLE_FEATS for f in FEATS])

    summary, coef_out = [], {}
    for hw_i, hw in enumerate([str(x) for x in d["hw_names"]]):
        hw_mask = d["hw_idx"] == hw_i
        if hw_mask.sum() == 0:
            continue
        X = design(d, hw, run_ids)

        # ---- grouped 5-fold CV (multipliers learned inside folds)
        rng = np.random.default_rng(0)
        uruns = np.unique(run_ids[hw_mask])
        fold_of = dict(zip(uruns.tolist(), rng.integers(0, 5, size=uruns.size).tolist()))
        fold_arr = np.array([fold_of.get(r, -1) for r in run_ids])
        ys, ps, idxs = [], [], []
        for f in range(5):
            tr = hw_mask & (fold_arr != f)
            te = hw_mask & (fold_arr == f)
            coef, mult = fit_hw(X, y, tr, fams, idle_cols)
            pred = predict(X, coef, mult, fams, idle_cols)
            ys.append(y[te]); ps.append(pred[te]); idxs.append(np.flatnonzero(te))
        ycv, pcv = np.concatenate(ys), np.concatenate(ps)
        icv = np.concatenate(idxs)
        m1 = metrics(ycv, pcv)
        # aggregated windows (energy fidelity)
        aggs = {}
        for win in (5, 30):
            key = (icv // win) * 100000 + run_ids[icv]
            dfa = pd.DataFrame(dict(k=key, y=ycv, p=pcv)).groupby("k").mean()
            aggs[win] = metrics(dfa["y"].values, dfa["p"].values)

        # ---- LOMO (held-out family, multiplier defaults to 1)
        lomo = {}
        for fi in np.unique(fams[hw_mask]):
            te = hw_mask & (fams == fi)
            tr = hw_mask & (fams != fi)
            if len(np.unique(fams[tr])) < 2:
                continue
            coef, mult = fit_hw(X, y, tr, fams, idle_cols)
            mult.pop(int(fi), None)
            pred = predict(X, coef, mult, fams, idle_cols)
            errs = []
            for r in np.unique(run_ids[te]):
                mm = te & (run_ids == r)
                errs.append(abs(np.mean(pred[mm]) - np.mean(y[mm])) / max(np.mean(y[mm]), 1.0) * 100)
            lomo[fam_names[fi]] = float(np.median(errs))

        # ---- LOTO (held-out TP)
        loto = {}
        for tp in np.unique(d["tp"][hw_mask]):
            te = hw_mask & (d["tp"] == tp)
            tr = hw_mask & (d["tp"] != tp)
            if tr.sum() == 0 or len(np.unique(d["tp"][tr])) < 2:
                continue
            coef, mult = fit_hw(X, y, tr, fams, idle_cols)
            pred = predict(X, coef, mult, fams, idle_cols)
            errs = []
            for r in np.unique(run_ids[te]):
                mm = te & (run_ids == r)
                errs.append(abs(np.mean(pred[mm]) - np.mean(y[mm])) / max(np.mean(y[mm]), 1.0) * 100)
            loto[f"tp{int(tp)}"] = float(np.median(errs))

        # ---- final fit on everything + attribution
        coef, mult = fit_hw(X, y, hw_mask, fams, idle_cols)
        pred_full = predict(X, coef, mult, fams, idle_cols)
        X_nopre = design(zero_phase(d, "prefill"), hw, run_ids)
        X_nodec = design(zero_phase(d, "decode"), hw, run_ids)
        X_idle = design(zero_phase(d, "both"), hw, run_ids)
        p_nopre = predict(X_nopre, coef, mult, fams, idle_cols)
        p_nodec = predict(X_nodec, coef, mult, fams, idle_cols)
        p_idle = predict(X_idle, coef, mult, fams, idle_cols)

        att_rows = []
        for fi in np.unique(fams[hw_mask]):
            for tp in np.unique(d["tp"][hw_mask & (fams == fi)]):
                m = hw_mask & (fams == fi) & (d["tp"] == tp)
                E = float(np.sum(pred_full[m]))
                e_idle = float(np.sum(p_idle[m]))
                e_pre = float(np.sum(pred_full[m] - p_nopre[m]))
                e_dec = float(np.sum(pred_full[m] - p_nodec[m]))
                e_shared = E - e_idle - e_pre - e_dec
                att_rows.append(dict(
                    hardware=hw, family=fam_names[fi], tp=int(tp),
                    idle_pct=100 * e_idle / E, prefill_pct=100 * e_pre / E,
                    decode_pct=100 * e_dec / E, shared_dvfs_pct=100 * e_shared / E,
                ))
        att = pd.DataFrame(att_rows)
        att.to_csv(OUT / f"attribution_{hw}.csv", index=False)

        print(f"\n######## {hw} final model")
        print(f"  CV(5-fold, by run): R2={m1['r2']:.4f} RMSE={m1['rmse']:.1f}W MAPE={m1['mape']:.1f}%")
        for win, ma in aggs.items():
            print(f"  {win}s window:        R2={ma['r2']:.4f} RMSE={ma['rmse']:.1f}W MAPE={ma['mape']:.1f}%")
        print("  LOMO run-mean err: " + ", ".join(f"{k}={v:.1f}%" for k, v in lomo.items()))
        print("  LOTO run-mean err: " + ", ".join(f"{k}={v:.1f}%" for k, v in loto.items()))
        print("  coefficients:")
        for f, c in zip(FEATS, coef):
            if c > 0:
                print(f"    {FEATURES[f][0]:28s} = {c:.4g}")
        print("  family multipliers: " + ", ".join(
            f"{fam_names[k]}={v:.3f}" for k, v in sorted(mult.items())))
        print("  energy attribution (per family/TP):")
        print(att.round(1).to_string(index=False))

        coef_out[hw] = dict(
            features={f: dict(label=FEATURES[f][0], value=float(c))
                      for f, c in zip(FEATS, coef)},
            family_multipliers={fam_names[k]: v for k, v in mult.items()},
            lag="EMA(0.6)" if hw == "A100" else "MA(2)+EMA(0.7)",
        )
        summary.append(dict(hardware=hw, **{f"cv_{k}": v for k, v in m1.items()},
                            **{f"agg5_{k}": v for k, v in aggs[5].items()},
                            **{f"agg30_{k}": v for k, v in aggs[30].items()},
                            lomo_med=float(np.median(list(lomo.values()))),
                            loto_med=float(np.median(list(loto.values())))))

        # figures: pred-vs-measured run means + one example trace
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for fi in np.unique(fams[hw_mask]):
            mm = hw_mask & (fams == fi)
            rm_y = [float(np.mean(y[mm & (run_ids == r)])) for r in np.unique(run_ids[mm])]
            rm_p = [float(np.mean(pred_full[mm & (run_ids == r)])) for r in np.unique(run_ids[mm])]
            ax.scatter(rm_y, rm_p, s=14, alpha=0.7, label=fam_names[fi])
        lim = [0, ax.get_xlim()[1]]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Measured run-mean power (W)")
        ax.set_ylabel("Predicted run-mean power (W)")
        ax.set_title(f"{hw}: final model, run means")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"final_run_means_{hw}.png", dpi=140)
        plt.close(fig)

        big_runs = np.unique(run_ids[hw_mask & (d["tp"] == 8) & (d["rate"] == 2.0)])
        if big_runs.size:
            r = big_runs[0]
            mm = run_ids == r
            fig, ax = plt.subplots(figsize=(9, 3.2))
            t = np.arange(int(mm.sum()))
            ax.plot(t, y[mm], lw=0.8, label="measured")
            ax.plot(t, pred_full[mm], lw=1.2, label="predicted (mean model)")
            ax.set_xlabel("time (s)"); ax.set_ylabel("node power (W)")
            ax.set_title(f"{hw} example run (TP=8, rate=2)"); ax.legend()
            fig.tight_layout()
            fig.savefig(OUT / f"final_trace_{hw}.png", dpi=140)
            plt.close(fig)

    with open(OUT / "final_coefficients.json", "w") as f:
        json.dump(coef_out, f, indent=2)
    pd.DataFrame(summary).to_csv(OUT / "final_summary.csv", index=False)
    print(f"\nWrote {OUT}/final_coefficients.json, final_summary.csv, attribution_*.csv, figures")


if __name__ == "__main__":
    main()
