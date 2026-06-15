"""Fit and evaluate a ladder of first-principles power models on the ledger cache.

Each model variant is a non-negative linear combination of physically
interpretable work-rate features, fit per hardware platform. Evaluation:
  - grouped 5-fold CV by run (honest pooled metrics)
  - leave-one-model-family-out (LOMO)
  - leave-one-TP-out (LOTO)

Run from repo root:
    uv run python feature-test/fit_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("feature-test/results")


HBM_BW = {0: 2.0e12, 1: 3.35e12}  # A100 SXM, H100 SXM bytes/s per GPU
PEAK_FLOPS = {0: 312e12, 1: 990e12}  # bf16 dense per GPU


def ema_by_run(x, run_ids, alpha):
    """Causal EMA within each run (bins are stored contiguously per run)."""
    out = np.empty_like(x, dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(run_ids)) + 1
    starts = np.concatenate([[0], boundaries, [x.size]])
    for s, e in zip(starts[:-1], starts[1:]):
        if e <= s:
            continue
        seg = x[s:e]
        acc = np.empty_like(seg, dtype=np.float64)
        acc[0] = seg[0]
        for i in range(1, seg.size):
            acc[i] = alpha * seg[i] + (1 - alpha) * acc[i - 1]
        out[s:e] = acc
    return out


def load_cache(path="feature-test/ledger_cache.npz"):
    d = dict(np.load(path, allow_pickle=False))
    d["flops_pre"] = 2.0 * d["n_active"] * d["pre_tok"]
    d["flops_dec"] = 2.0 * d["n_active"] * d["dec_tok"]
    d["busy"] = ((d["pre_tok"] + d["dec_tok"]) > 0).astype(np.float64)
    bw = np.where(d["hw_idx"] == 0, HBM_BW[0], HBM_BW[1])
    pf = np.where(d["hw_idx"] == 0, PEAK_FLOPS[0], PEAK_FLOPS[1])
    # per-GPU utilizations in [0, ~1]
    d["util_mem"] = np.clip((d["w_read"] + d["kv_read"]) / (d["tp"] * bw), 0.0, 1.5)
    d["util_cmp"] = np.clip((d["flops_pre"] + d["flops_dec"]) / (d["tp"] * pf), 0.0, 1.5)
    return d


# ---------------------------------------------------------------- features
# Each feature: (name, fn(d) -> array). All coefficients constrained >= 0.

def F(name):
    return lambda d: d[name]


FEATURES = {
    "tp": ("p_idle [W/GPU]", F("tp")),
    "busy_tp": ("p_active_floor [W/GPU]", lambda d: d["tp"] * d["busy"]),
    "flops": ("e_f [J/FLOP]", lambda d: d["flops_pre"] + d["flops_dec"]),
    "flops_pre": ("e_f_prefill [J/FLOP]", F("flops_pre")),
    "flops_dec": ("e_f_decode [J/FLOP]", F("flops_dec")),
    "flops_pre_fp8": ("e_f_prefill_fp8 [J/FLOP]", lambda d: d["flops_pre"] * d["fp8"]),
    "flops_dec_fp8": ("e_f_decode_fp8 [J/FLOP]", lambda d: d["flops_dec"] * d["fp8"]),
    "flops_pre_bf16": ("e_f_prefill_bf16 [J/FLOP]", lambda d: d["flops_pre"] * (1 - d["fp8"])),
    "flops_dec_bf16": ("e_f_decode_bf16 [J/FLOP]", lambda d: d["flops_dec"] * (1 - d["fp8"])),
    "w_read": ("e_w [J/B]", F("w_read")),
    "kv_read": ("e_kv [J/B]", F("kv_read")),
    "comm": ("e_comm [J/B]", F("comm")),
    "w_read_pre": ("e_w_prefill [J/B]", F("w_read_pre")),
    "w_read_dec": ("e_w_decode [J/B]", F("w_read_dec")),
    "kv_write": ("e_kv_write [J/B]", F("kv_write")),
    # DVFS / clock-boost: concave per-GPU response to utilization, scaled by TP.
    "boost_mem": ("p_boost_mem [W/GPU]", lambda d: d["tp"] * np.sqrt(d["util_mem"])),
    "boost_cmp": ("p_boost_cmp [W/GPU]", lambda d: d["tp"] * np.sqrt(d["util_cmp"])),
    # Saturating frequency-residency bases: TP*(1-exp(-u/u0)).
    "sat_cmp_lo": ("p_sat_cmp_u0.03 [W/GPU]", lambda d: d["tp"] * (1 - np.exp(-d["util_cmp"] / 0.03))),
    "sat_cmp_hi": ("p_sat_cmp_u0.2 [W/GPU]", lambda d: d["tp"] * (1 - np.exp(-d["util_cmp"] / 0.2))),
    "sat_mem_lo": ("p_sat_mem_u0.1 [W/GPU]", lambda d: d["tp"] * (1 - np.exp(-d["util_mem"] / 0.1))),
    "sat_mem_hi": ("p_sat_mem_u0.4 [W/GPU]", lambda d: d["tp"] * (1 - np.exp(-d["util_mem"] / 0.4))),
    # NVLink/fabric standing power: per GPU, only when the model is sharded.
    "tp_link": ("p_link [W/GPU, TP>1]", lambda d: d["tp"] * (d["tp"] > 1)),
}

MODELS = {
    "M0_baseline": ["tp", "flops", "w_read", "kv_read"],
    "M1_comm": ["tp", "flops", "w_read", "kv_read", "comm"],
    "M2_phase_split": ["tp", "flops_pre", "flops_dec", "w_read", "kv_read", "comm"],
    "M3_active_floor": ["tp", "busy_tp", "flops_pre", "flops_dec", "w_read", "kv_read", "comm"],
    "M4_dtype": ["tp", "busy_tp", "flops_pre_bf16", "flops_dec_bf16",
                 "flops_pre_fp8", "flops_dec_fp8", "w_read", "kv_read", "comm"],
    "M5_dvfs": ["tp", "busy_tp", "boost_mem", "boost_cmp",
                "flops_pre", "flops_dec", "w_read", "kv_read", "comm"],
    "M6_phase_traffic": ["tp", "busy_tp", "flops_pre", "flops_dec",
                         "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm"],
    "M7_full": ["tp", "busy_tp", "sat_cmp_lo", "sat_cmp_hi", "sat_mem_lo", "sat_mem_hi",
                "flops_pre", "flops_dec", "w_read_pre", "w_read_dec",
                "kv_read", "kv_write", "comm"],
    "M8_link": ["tp", "tp_link", "busy_tp", "sat_cmp_lo", "sat_cmp_hi",
                "sat_mem_lo", "sat_mem_hi", "flops_pre", "flops_dec",
                "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm"],
}

# Variants of the best feature set with causal EMA lag applied to features
# (equivalent to lagging predicted power; models meter averaging + thermal lag).
LAG_BASE = "M8_link"
LAG_ALPHAS = {"lag_a0.8": 0.8, "lag_a0.7": 0.7, "lag_a0.6": 0.6, "lag_a0.5": 0.5}


def build_design_full(d, feats, alpha=None):
    """Full (n_bins, n_feat) design; optional causal EMA per run on each column."""
    cols = [np.asarray(FEATURES[f][1](d), dtype=np.float64) for f in feats]
    if alpha is not None:
        cols = [ema_by_run(c, d["run_id"], alpha) for c in cols]
    return np.column_stack(cols)


def metrics(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return dict(
        r2=1.0 - ss_res / max(ss_tot, 1e-12),
        rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
        mape=float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0))) * 100,
    )


def fit_predict(X, y, train_mask, test_mask):
    coef, _ = nnls(X[train_mask], y[train_mask])
    return coef, y[test_mask], X[test_mask] @ coef


def grouped_cv(X, y, hw_mask, run_ids, k=5, seed=0):
    rng = np.random.default_rng(seed)
    uruns = np.unique(run_ids[hw_mask])
    folds = rng.integers(0, k, size=uruns.size)
    fold_of = dict(zip(uruns.tolist(), folds.tolist()))
    fold_arr = np.array([fold_of.get(r, -1) for r in run_ids])
    ys, ps = [], []
    for f in range(k):
        tr = hw_mask & (fold_arr != f)
        te = hw_mask & (fold_arr == f)
        if te.sum() == 0 or tr.sum() == 0:
            continue
        _, yte, pred = fit_predict(X, y, tr, te)
        ys.append(yte)
        ps.append(pred)
    return metrics(np.concatenate(ys), np.concatenate(ps))


def run_mean_errors(X, y, train_mask, test_mask, run_ids):
    """Per-run mean-power |error| % on test runs."""
    _, yte, pred = fit_predict(X, y, train_mask, test_mask)
    rids = run_ids[test_mask]
    errs = []
    for r in np.unique(rids):
        m = rids == r
        meas, prd = float(np.mean(yte[m])), float(np.mean(pred[m]))
        errs.append(abs(prd - meas) / max(meas, 1.0) * 100)
    return np.asarray(errs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = load_cache()
    run_ids = d["run_id"]
    hw_names = [str(x) for x in d["hw_names"]]
    fam_names = [str(x) for x in d["family_names"]]
    rows = []

    variants = [(name, feats, None) for name, feats in MODELS.items()]
    variants += [(f"{LAG_BASE}+{ln}", MODELS[LAG_BASE], a) for ln, a in LAG_ALPHAS.items()]

    y = np.asarray(d["power"], dtype=np.float64)
    for hw_i, hw in enumerate(hw_names):
        hw_mask = d["hw_idx"] == hw_i
        if hw_mask.sum() == 0:
            continue
        fams_here = np.unique(d["family_idx"][hw_mask])
        tps_here = np.unique(d["tp"][hw_mask])
        print(f"\n################ {hw}: {int(hw_mask.sum())} bins, "
              f"families={[fam_names[i] for i in fams_here]}, TPs={tps_here.astype(int).tolist()}")

        for mname, feats, alpha in variants:
            X = build_design_full(d, feats, alpha=alpha)
            # pooled grouped CV
            cv = grouped_cv(X, y, hw_mask, run_ids)
            # in-sample fit for coefficients
            coef, y_all, pred_all = fit_predict(X, y, hw_mask, hw_mask)
            ins = metrics(y_all, pred_all)

            # LOMO
            lomo_meds = []
            for fi in fams_here:
                te = hw_mask & (d["family_idx"] == fi)
                tr = hw_mask & (d["family_idx"] != fi)
                if len(np.unique(d["family_idx"][tr])) < 2:
                    continue
                errs = run_mean_errors(X, y, tr, te, run_ids)
                lomo_meds.append(float(np.median(errs)))
                rows.append(dict(hardware=hw, model=mname, eval="lomo",
                                 held=fam_names[fi], metric="run_mean_err_med_pct",
                                 value=float(np.median(errs))))
            # LOTO
            loto_meds = []
            for tp in tps_here:
                te = hw_mask & (d["tp"] == tp)
                tr = hw_mask & (d["tp"] != tp)
                if tr.sum() == 0 or len(np.unique(d["tp"][tr])) < 2:
                    continue
                errs = run_mean_errors(X, y, tr, te, run_ids)
                loto_meds.append(float(np.median(errs)))
                rows.append(dict(hardware=hw, model=mname, eval="loto",
                                 held=f"tp{int(tp)}", metric="run_mean_err_med_pct",
                                 value=float(np.median(errs))))

            rows.extend([
                dict(hardware=hw, model=mname, eval="cv", held="", metric=k, value=v)
                for k, v in cv.items()
            ])
            print(f"  {mname:18s} CV: R2={cv['r2']:.4f} RMSE={cv['rmse']:6.1f}W "
                  f"MAPE={cv['mape']:5.1f}%  | LOMO med {np.mean(lomo_meds):5.1f}%"
                  f"  LOTO med {np.mean(loto_meds) if loto_meds else float('nan'):5.1f}%"
                  f"  (in-sample R2={ins['r2']:.4f})")
            coef_str = ", ".join(
                f"{FEATURES[f][0]}={c:.3g}" for f, c in zip(feats, coef)
            )
            print(f"      coefs: {coef_str}")
            for f, c in zip(feats, coef):
                rows.append(dict(hardware=hw, model=mname, eval="coef", held="",
                                 metric=FEATURES[f][0], value=float(c)))

    pd.DataFrame(rows).to_csv(OUT_DIR / "model_ladder_metrics.csv", index=False)
    print(f"\nWrote {OUT_DIR}/model_ladder_metrics.csv")


if __name__ == "__main__":
    main()
