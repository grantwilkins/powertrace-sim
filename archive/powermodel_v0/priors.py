"""Physical priors and the model's feature schema.

Each coefficient is ONE physically-irreducible constant, parameterized as
``c = exp(theta)`` with a log-normal prior centered on a datasheet value. We
deliberately do NOT split energy constants by phase: a FLOP costs the same energy
whether it is a prefill projection or a decode GEMV (the prefill-vs-decode
*efficiency* difference is carried by the roofline factor ``eta(AI)``, not by a
second coefficient), and an HBM byte costs the same whether it is a weight read
or a KV read. Collapsing these removes the collinearity that pushed the old
phase-split coefficients into degeneracy.

Standing power (idle + NVLink-link) is NOT fit here — it is anchored directly
from measured idle bins (``model.standing_anchor``), since at a single TP the
idle and link columns are perfectly collinear and only a TP=1 measurement could
separate them. So the fit estimates only the DYNAMIC terms below.

Dynamic feature columns (assembled in ``model.build_design`` from
``arch.work_rates``):

  active   p_active [W/GPU]   busy clock-boost floor          x TP*busy
  flop     e_flop   [J/FLOP]  compute energy (proj + decode)  x eta
  attn     e_attn   [J/FLOP]  attention-kernel energy         x eta_pre
  hbm      e_hbm    [J/B]     HBM read energy (weights + KV)
  comm     e_comm   [J/B]     NVLink all-reduce energy (tight prior)
"""

from __future__ import annotations

FEATS = ("active", "flop", "attn", "hbm", "comm")

LABELS = {
    "active": "p_active [W/GPU]",
    "flop": "e_flop [J/FLOP]",
    "attn": "e_attn [J/FLOP]",
    "hbm": "e_hbm [J/B]",
    "comm": "e_comm [J/B]",
}

# (prior mean, prior sd in log space). e_flop ~1 pJ/FLOP at datasheet MFU; HBM
# ~0.15 nJ/B; NVLink ~0.2 nJ/B (tight — prior-dominated until a TP-pair probe).
PRIORS = {
    # p_active is the idle->busy clock-boost floor: physically ~20-40 W/GPU and
    # PHASE-INDEPENDENT. Tight prior so it cannot balloon to absorb compute-bound
    # prefill power (that power belongs in e_flop*FLOPs, which is FLOP-proportional;
    # a flat floor cannot be both high for compute-bound prefill and low for
    # memory-bound decode).
    "active": (40.0, 0.50),
    "flop": (1.0e-12, 0.50),
    "attn": (1.0e-12, 0.70),
    # HBM read energy/byte is a hardware constant we have independently identified
    # at ~0.083 (A100) / 0.135 (H100) nJ/B (textbook) across many fits. Pin it
    # tight so the roofline eta term cannot steal from the memory term.
    "hbm": (1.1e-10, 0.22),
    # e_comm is small; tight so it does not proxy a TP-dependent effect (the tp2-vs
    # -tp4 contrast otherwise drives it to absurd values, as feature-test flagged).
    "comm": (2.0e-10, 0.20),
}
