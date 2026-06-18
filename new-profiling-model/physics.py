"""First-principles power features and the explainable model ladder.

These are the SAME physically-interpretable terms as feature-test/fit_models.py
(standing power, active floor, concave DVFS bases, phase-split compute/memory
traffic, fabric) -- reused deliberately so the model stays explainable and
comparable. The only change vs feature-test is *where the work rates come from*:
here they are computed from measured engine state (see dataio.work_rates), so
the features are clean.

A "point" is a dict from dataio (one probe level or one validate bin). We stack
points into a design matrix; coefficients are fit non-negative (every term is a
non-negative watt contribution), so each fitted constant has a physical sign and
unit.
"""

from __future__ import annotations

import numpy as np

# Per-GPU peak roofline constants (datasheet; used only to NORMALISE utilisation
# for the concave DVFS bases -- not fit).
HBM_BW = {"A100": 2.0e12, "H100": 3.35e12}        # bytes/s per GPU
PEAK_FLOPS = {"A100": 312e12, "H100": 990e12}      # bf16 dense FLOP/s per GPU


def _util(points):
    bw = np.array([HBM_BW[p["hw"]] for p in points])
    pf = np.array([PEAK_FLOPS[p["hw"]] for p in points])
    tp = np.array([p["tp"] for p in points])
    w_read = np.array([p["w_read"] for p in points])
    kv_read = np.array([p["kv_read"] for p in points])
    flops = np.array([p["flops_pre"] + p["flops_dec"] for p in points])
    util_mem = np.clip((w_read + kv_read) / (tp * bw), 0.0, 1.5)
    util_cmp = np.clip(flops / (tp * pf), 0.0, 1.5)
    return util_mem, util_cmp


def _col(points, key):
    return np.array([p[key] for p in points], dtype=float)


# name -> (label/unit, function(points) -> column). Coefficients all >= 0.
def feature_table(points):
    tp = _col(points, "tp")
    busy = _col(points, "busy")
    util_mem, util_cmp = _util(points)
    feats = {
        # standing power
        "tp":        ("p_idle [W/GPU]",            tp),
        "tp_link":   ("p_link [W/GPU, TP>1]",      tp * (tp > 1)),
        # busy floor (clock boost above idle the instant the GPU has work)
        "busy_tp":   ("p_active_floor [W/GPU]",    tp * busy),
        # concave DVFS bases: TP*(1-exp(-u/u0)); two knees per resource
        "sat_cmp_lo": ("p_sat_cmp_u0.03 [W/GPU]",  tp * (1 - np.exp(-util_cmp / 0.03))),
        "sat_cmp_hi": ("p_sat_cmp_u0.2 [W/GPU]",   tp * (1 - np.exp(-util_cmp / 0.2))),
        "sat_mem_lo": ("p_sat_mem_u0.1 [W/GPU]",   tp * (1 - np.exp(-util_mem / 0.1))),
        "sat_mem_hi": ("p_sat_mem_u0.4 [W/GPU]",   tp * (1 - np.exp(-util_mem / 0.4))),
        # compute energy (combined + phase split)
        "flops":     ("e_f [J/FLOP]",              _col(points, "flops_pre") + _col(points, "flops_dec")),
        "flops_pre": ("e_f_prefill [J/FLOP]",      _col(points, "flops_pre")),
        "flops_dec": ("e_f_decode [J/FLOP]",       _col(points, "flops_dec")),
        # memory traffic (combined + phase split)
        "w_read":    ("e_w [J/B]",                 _col(points, "w_read")),
        "w_read_pre": ("e_w_prefill [J/B]",        _col(points, "w_read_pre")),
        "w_read_dec": ("e_w_decode [J/B]",         _col(points, "w_read_dec")),
        "kv_read":   ("e_kv_read [J/B]",           _col(points, "kv_read")),
        "kv_write":  ("e_kv_write [J/B]",          _col(points, "kv_write")),
        # fabric
        "comm":      ("e_comm [J/B]",              _col(points, "comm")),
    }
    return feats


# Model ladder: increasingly physical, all explainable. On these A100s the SM
# clock is pinned at max boost (1410 MHz) across the whole staircase, so DVFS
# saturating terms carry no signal -- the physics is pure roofline-energy at
# fixed clock. We therefore center the ladder on energy-per-work coefficients.
MODELS = {
    # pure roofline energy: idle anchor + per-work energies. Every coefficient
    # is a J/B or J/FLOP with a physical sign. The "busy step" emerges from
    # weight streaming (you read the weights the instant you decode).
    "R0_roofline":   ["tp", "tp_link", "flops_dec", "w_read", "kv_read"],
    # phase-split roofline (prefill vs decode) + KV write + fabric
    "R1_phase":      ["tp", "tp_link", "flops_pre", "flops_dec",
                      "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm"],
    # roofline + explicit busy floor (fixed overhead the instant a kernel runs)
    "F0_floor":      ["tp", "tp_link", "busy_tp", "flops_dec", "w_read", "kv_read"],
    # floor + phase split
    "F1_phase":      ["tp", "tp_link", "busy_tp", "flops_pre", "flops_dec",
                      "w_read_pre", "w_read_dec", "kv_read", "kv_write", "comm"],
    # saturating-utilisation form: at fixed clock, dynamic power is the chip
    # filling up -> concave in memory/compute utilisation. Best curve shape.
    "S_sat":         ["tp", "tp_link", "busy_tp", "sat_mem_lo", "sat_mem_hi",
                      "sat_cmp_lo", "sat_cmp_hi"],
    # BEST: standing + busy floor + concave memory fill (decode is memory-bound)
    # + a compute-energy term so prefill / compute-heavy bins ride on top.
    "B_best":        ["tp", "tp_link", "busy_tp", "sat_mem_lo", "sat_mem_hi",
                      "flops_dec", "kv_read"],
}

IDLE_FEATS = {"tp", "tp_link"}  # the static (multiplier-exempt) part


def design(points, feats):
    table = feature_table(points)
    cols = [table[f][1] for f in feats]
    return np.column_stack(cols)


def labels(feats):
    # uses an empty-point-safe label lookup
    base = {
        "tp": "p_idle [W/GPU]", "tp_link": "p_link [W/GPU, TP>1]",
        "busy_tp": "p_active_floor [W/GPU]",
        "sat_cmp_lo": "p_sat_cmp_u0.03 [W/GPU]", "sat_cmp_hi": "p_sat_cmp_u0.2 [W/GPU]",
        "sat_mem_lo": "p_sat_mem_u0.1 [W/GPU]", "sat_mem_hi": "p_sat_mem_u0.4 [W/GPU]",
        "flops": "e_f [J/FLOP]",
        "flops_pre": "e_f_prefill [J/FLOP]", "flops_dec": "e_f_decode [J/FLOP]",
        "w_read": "e_w [J/B]",
        "w_read_pre": "e_w_prefill [J/B]", "w_read_dec": "e_w_decode [J/B]",
        "kv_read": "e_kv_read [J/B]", "kv_write": "e_kv_write [J/B]",
        "comm": "e_comm [J/B]",
    }
    return [base[f] for f in feats]
