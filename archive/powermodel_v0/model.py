"""The power model: assemble physics features, fit, predict, attribute.

Per hardware platform, node power is a non-negative sum of physically
interpretable terms, passed through a causal lag filter that models the
nvidia-smi meter's moving-average response:

    P_node(t) = p_idle*TP + p_link*TP*1[TP>1] + p_active*TP*1[busy]
              + eta_pre*( e_f_pre*FLOPs_proj_pre + e_f_attn*FLOPs_attn_pre )
              + e_w_pre*Wbytes_pre
              + eta_dec*( e_f_dec*FLOPs_dec ) + e_w_dec*Wbytes_dec
              + e_kv*KVread + e_comm*NVLink            (all per second)
    P_meas(t) = Lag(P_node)(t)

The feature columns come from ``arch.work_rates`` (the single physics source,
shared with the predict-time inference chain), so calibration and prediction can
never use different physics. Coefficients are fit by MAP with datasheet priors
(``estimate.py``); each carries a posterior sd + identifiability label and
predictions carry intervals.
"""

from __future__ import annotations

import json

import numpy as np

from powermodel import arch as A
from powermodel import estimate as E
from powermodel.priors import FEATS, LABELS

# Causal EMA smoothing alpha per platform (meter averaging + power dynamics).
LAG_ALPHA = {"A100": 0.6, "H100": 0.7}


def ema_by_run(x, run_ids, alpha):
    """Causal EMA within each run (bins stored contiguously per run)."""
    out = np.empty_like(x, dtype=np.float64)
    bnd = np.flatnonzero(np.diff(run_ids)) + 1
    starts = np.concatenate([[0], bnd, [x.size]])
    for s, e in zip(starts[:-1], starts[1:]):
        if e <= s:
            continue
        seg = x[s:e]
        acc = np.empty_like(seg)
        acc[0] = seg[0]
        for i in range(1, seg.size):
            acc[i] = alpha * seg[i] + (1 - alpha) * acc[i - 1]
        out[s:e] = acc
    return out


def state_features(arch, hw, tp, state):
    """Dynamic feature columns (pre-lag) for one run's per-bin state.

    Energy constants are merged to one-per-mechanism (see priors.py): ``flop``
    sums prefill-projection and decode compute (each scaled by its roofline
    ``eta``); ``hbm`` sums all HBM reads (prefill + decode weights + KV). Standing
    power is handled separately (anchored from idle), so it is NOT a column here.
    """
    arch = A.normalize_arch(arch)
    wr = A.work_rates(
        arch, hw, tp,
        pre_tok=state["pre_tok"], dec_tok=state["dec_tok"],
        decode_batch=state["decode_batch"], L_pre=state["L_pre"],
        ctx_dec=state["ctx_dec"],
        iters_pre=state.get("iters_pre"), iters_dec=state.get("iters_dec"),
    )
    busy = state.get("busy", ((state["pre_tok"] + state["dec_tok"]) > 0).astype(float))
    eta = wr["eta"]
    cols = {
        "active": tp * np.asarray(busy, dtype=np.float64),
        "flop": eta * (wr["flops_proj_pre"] + wr["flops_dec"]),
        "attn": eta * wr["flops_attn_pre"],
        "hbm": wr["w_read"] + wr["kv_read"],
        "comm": wr["nvlink"],
    }
    return cols


def build_design(ledger, hw_idx):
    """Full (n_bins, n_feat) lagged design for one hardware platform.

    ``ledger`` is the dict from ``ingest.load_ledger``. Returns (X, mask) where
    mask selects this platform's bins (rows of X are filled only for the mask,
    others are zero — keep alignment with run_ids for the lag).
    """
    hw_name = str(ledger["hw_names"][hw_idx])
    run_ids = ledger["run_id"]
    alpha = LAG_ALPHA.get(hw_name, 0.6)
    n = run_ids.size
    raw = {f: np.zeros(n) for f in FEATS}
    meta = ledger["meta"]
    arch_by_run = {m["run_id"]: ledger["arch"][i] for i, m in enumerate(meta)}
    tp_by_run = {m["run_id"]: m["tp"] for m in meta}
    for rid in np.unique(run_ids):
        sl = run_ids == rid
        if ledger["hw_idx"][sl][0] != hw_idx:
            continue
        state = {k: ledger[k][sl] for k in
                 ("pre_tok", "dec_tok", "decode_batch", "pre_active",
                  "L_pre", "ctx_dec", "iters_pre", "iters_dec", "busy")}
        cols = state_features(arch_by_run[int(rid)], hw_name, float(tp_by_run[int(rid)]), state)
        for f in FEATS:
            raw[f][sl] = cols[f]
    # causal lag per run on every column
    X = np.column_stack([ema_by_run(raw[f], run_ids, alpha) for f in FEATS])
    mask = ledger["hw_idx"] == hw_idx
    return X, mask


def metrics(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return dict(
        r2=1.0 - ss_res / max(ss_tot, 1e-12),
        rmse=float(np.sqrt(np.mean((y - pred) ** 2))),
        mape=float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0))) * 100,
    )


def standing_anchor(y, tp_vec, busy, mask):
    """Per-GPU standing power, measured from idle bins (not fit).

    Idle bins (busy==0) draw idle + NVLink-link power; their per-GPU mean is the
    standing constant. At a single TP idle and link are inseparable, so we anchor
    the COMBINED standing-per-GPU (exact for prediction at that TP). Returns the
    per-GPU watts; multiply by TP for the per-bin standing power.
    """
    idle = mask & (np.asarray(busy) == 0)
    if idle.sum() < 5:
        idle = mask  # fallback: use the low quantile of all bins
        per_gpu = np.percentile(y[idle] / np.maximum(tp_vec[idle], 1), 5)
    else:
        per_gpu = float(np.median(y[idle] / np.maximum(tp_vec[idle], 1)))
    return float(per_gpu)


def calibrate(ledger, hw_idx, train_mask=None):
    """Anchor standing power from idle, then MAP-fit the DYNAMIC terms on the
    residual. Removing the (collinear) standing terms from the fit lets the
    physical energy constants be identified instead of trading against idle."""
    X, hw_mask = build_design(ledger, hw_idx)
    y = ledger["power"].astype(np.float64)
    tp_vec = ledger["tp"].astype(np.float64)
    busy = ledger["busy"]
    fit_mask = hw_mask if train_mask is None else (hw_mask & train_mask)

    stand_pg = standing_anchor(y, tp_vec, busy, fit_mask)
    standing = stand_pg * tp_vec                      # per-bin standing power
    y_dyn = y - standing                              # dynamic residual

    theta, sigma = E.fit_two_stage(X[fit_mask], y_dyn[fit_mask])
    cov = E.laplace_cov(X[fit_mask], y_dyn[fit_mask], theta, sigma)
    return dict(theta=theta, cov=cov, sigma=sigma, hw=str(ledger["hw_names"][hw_idx]),
                X=X, hw_mask=hw_mask, standing_per_gpu=stand_pg, standing=standing)


def coefficients(theta):
    return {f: float(np.exp(theta[j])) for j, f in enumerate(FEATS)}


def predict_full(fit):
    """Predicted power = anchored standing + fitted dynamic (for a calibration fit)."""
    return fit["standing"] + fit["X"] @ np.exp(fit["theta"])


def predict_state(arch, hw, tp, state, theta, standing_per_gpu):
    """Predict the (lagged) power trace from a per-bin STATE dict.

    Predict-time entry point: ``state`` is produced by ``workload.py`` from
    provider-known inputs. Standing power = ``standing_per_gpu * tp``.
    """
    cols = state_features(arch, hw, tp, state)
    run_ids = np.zeros(np.asarray(state["pre_tok"]).size, dtype=np.int64)
    alpha = LAG_ALPHA.get(hw, 0.6)
    X = np.column_stack([ema_by_run(cols[f], run_ids, alpha) for f in FEATS])
    return standing_per_gpu * tp + X @ np.exp(theta)


def save_coefficients(path, by_hw):
    out = {}
    for hw, fit in by_hw.items():
        rows = E.identifiability(fit["theta"], fit["cov"])
        out[hw] = dict(
            standing_per_gpu_W=fit["standing_per_gpu"],
            coefficients={f: float(np.exp(fit["theta"][j])) for j, f in enumerate(FEATS)},
            labels=LABELS,
            identifiability=rows,
            lag=f"EMA({LAG_ALPHA.get(hw, 0.6)})",
        )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out
