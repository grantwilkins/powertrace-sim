"""Power-per-regime reformulation of the dynamic power model.

Instead of fitting abstract energy-per-unit coefficients (e_flop [J/FLOP],
e_hbm [J/B]) we fit a few directly-physical POWER LEVELS, and let the datasheet
roofline decide which regime a 1 s bin operates in. Per GPU, define datasheet
utilizations (clipped to [0, 1]):

    u_cmp = FLOPs_rate / (tp * peak_flops)   FLOPs_rate = proj_pre + attn_pre + dec
    u_mem = bytes_rate / (tp * hbm_bw)       bytes_rate = w_read + kv_read

Two variants share the same standing anchor (measured from idle, NOT fit) and
the same causal EMA lag as ``model.py``:

  ADDITIVE (priority):
    P_node = standing*tp
           + tp * P_cmp * min(u_cmp, 1)
           + tp * P_mem * min(u_mem, 1)
           + e_comm * nvlink

  BOTTLENECK (single binding resource per bin):
    P_node = standing*tp
           + tp * (P_cmp if u_cmp >= u_mem else P_mem) * max(u_cmp, u_mem)
           + e_comm * nvlink

Optionally a busy "active floor" column (tp * busy * P_active) can be added.
Fitting reuses ``estimate.py`` (MAP, Laplace cov, identifiability) via its new
optional ``priors`` argument; the model is LINEAR in the fitted power levels.
P_cmp / P_mem are W/GPU dynamic-power levels at full utilization (~O(100-400)).
"""

from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np

from powermodel import arch as A
from powermodel import estimate as E
from powermodel.model import ema_by_run, standing_anchor, metrics  # reuse, no fork

LAG_ALPHA = {"A100": 0.6, "H100": 0.7}

# Feature schemas + log-normal priors (mean, sd_log) for the regime variants.
# P_cmp / P_mem are dynamic power levels W/GPU at full utilization ~ O(100-400).
PRIORS_ADD = OrderedDict([
    ("cmp", (200.0, 0.60)),   # P_cmp  [W/GPU]  dynamic power when compute-bound
    ("mem", (200.0, 0.60)),   # P_mem  [W/GPU]  dynamic power when memory-bound
    ("comm", (2.0e-10, 0.20)),  # e_comm [J/B]  NVLink all-reduce
])
PRIORS_ADD_FLOOR = OrderedDict([
    ("active", (40.0, 0.50)),  # busy clock-boost floor [W/GPU]
    ("cmp", (200.0, 0.60)),
    ("mem", (200.0, 0.60)),
    ("comm", (2.0e-10, 0.20)),
])
# bottleneck uses one regime per bin; same priors as additive (no floor variant
# kept symmetric for comparison)
PRIORS_BOT = OrderedDict([
    ("regime", (200.0, 0.60)),  # binding-resource power level [W/GPU]
    # comm pinned TIGHT: with a single regime power level there is no TP-aware
    # term, so a loose e_comm runs away to ~1e-8 (drift +20 sigma) absorbing the
    # tp2-vs-tp1 contrast. Tight prior keeps it physical (~0.2 nJ/B).
    ("comm", (2.0e-10, 0.05)),
])

LABELS = {
    "active": "P_active [W/GPU]",
    "cmp": "P_cmp [W/GPU]",
    "mem": "P_mem [W/GPU]",
    "regime": "P_regime [W/GPU]",
    "comm": "e_comm [J/B]",
}


def _utilizations(arch, hw, tp, state):
    """Roofline utilizations (per GPU, clipped to [0,1]) + nvlink bytes/s."""
    arch = A.normalize_arch(arch)
    wr = A.work_rates(
        arch, hw, tp,
        pre_tok=state["pre_tok"], dec_tok=state["dec_tok"],
        decode_batch=state["decode_batch"], L_pre=state["L_pre"],
        ctx_dec=state["ctx_dec"],
        iters_pre=state.get("iters_pre"), iters_dec=state.get("iters_dec"),
    )
    peak = A.HW[hw]["peak_flops"]
    bw = A.HW[hw]["hbm_bw"]
    flops_rate = wr["flops_proj_pre"] + wr["flops_attn_pre"] + wr["flops_dec"]
    bytes_rate = wr["w_read"] + wr["kv_read"]
    u_cmp = np.clip(flops_rate / (tp * peak), 0.0, 1.0)
    u_mem = np.clip(bytes_rate / (tp * bw), 0.0, 1.0)
    busy = state.get("busy",
                      ((state["pre_tok"] + state["dec_tok"]) > 0).astype(float))
    return u_cmp, u_mem, wr["nvlink"], np.asarray(busy, dtype=np.float64)


def state_features(arch, hw, tp, state, variant="additive", floor=False):
    """Pre-lag feature columns for the regime model. Each column already carries
    its tp scaling so the fitted coefficient is a per-GPU power level."""
    u_cmp, u_mem, nvlink, busy = _utilizations(arch, hw, tp, state)
    if variant == "bottleneck":
        cmp_wins = u_cmp >= u_mem
        u_bind = np.where(cmp_wins, u_cmp, u_mem)
        cols = OrderedDict([
            ("regime", tp * u_bind),
            ("comm", nvlink),
        ])
        return cols
    # additive
    cols = OrderedDict()
    if floor:
        cols["active"] = tp * busy
    cols["cmp"] = tp * u_cmp
    cols["mem"] = tp * u_mem
    cols["comm"] = nvlink
    return cols


def priors_for(variant, floor):
    if variant == "bottleneck":
        return PRIORS_BOT
    return PRIORS_ADD_FLOOR if floor else PRIORS_ADD


def build_design(ledger, hw_idx, variant="additive", floor=False):
    """Full (n_bins, n_feat) lagged design for one platform. Returns (X, mask, feats)."""
    hw_name = str(ledger["hw_names"][hw_idx])
    run_ids = ledger["run_id"]
    alpha = LAG_ALPHA.get(hw_name, 0.6)
    n = run_ids.size
    feats = tuple(priors_for(variant, floor).keys())
    raw = {f: np.zeros(n) for f in feats}
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
        cols = state_features(arch_by_run[int(rid)], hw_name,
                              float(tp_by_run[int(rid)]), state,
                              variant=variant, floor=floor)
        for f in feats:
            raw[f][sl] = cols[f]
    X = np.column_stack([ema_by_run(raw[f], run_ids, alpha) for f in feats])
    mask = ledger["hw_idx"] == hw_idx
    return X, mask, feats


def calibrate(ledger, hw_idx, train_mask=None, variant="additive", floor=False):
    """Anchor standing from idle, MAP-fit the regime power levels on the residual."""
    X, hw_mask, feats = build_design(ledger, hw_idx, variant=variant, floor=floor)
    y = ledger["power"].astype(np.float64)
    tp_vec = ledger["tp"].astype(np.float64)
    busy = ledger["busy"]
    fit_mask = hw_mask if train_mask is None else (hw_mask & train_mask)
    priors = priors_for(variant, floor)

    stand_pg = standing_anchor(y, tp_vec, busy, fit_mask)
    standing = stand_pg * tp_vec
    y_dyn = y - standing

    theta, sigma = E.fit_two_stage(X[fit_mask], y_dyn[fit_mask], priors=priors)
    cov = E.laplace_cov(X[fit_mask], y_dyn[fit_mask], theta, sigma, priors=priors)
    return dict(theta=theta, cov=cov, sigma=sigma,
                hw=str(ledger["hw_names"][hw_idx]), X=X, hw_mask=hw_mask,
                standing_per_gpu=stand_pg, standing=standing,
                feats=feats, priors=priors, variant=variant, floor=floor)


def predict_full(fit):
    return fit["standing"] + fit["X"] @ np.exp(fit["theta"])
