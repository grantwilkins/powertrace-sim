"""powermodel: clean bottom-up, provider-configurable LLM inference power model.

A two-layer model:

* **Interface layer** (``workload.py``) — inputs a cloud provider knows or can
  infer: input/output token pairs, offered load, model architecture, TP degree,
  server config, hardware datasheet. It produces effective occupancy and the
  per-second work rates via a pure inference chain (no live telemetry needed).
* **Physics layer** (``arch.py``) — maps work to FLOPs / weight-bytes / KV-bytes
  / NVLink-bytes from the architecture descriptor, computes arithmetic intensity
  and the roofline efficiency ``eta(AI)``.

Power is a non-negative sum of physically interpretable terms (``model.py``),
fit by MAP with datasheet priors (``estimate.py``) so each coefficient carries a
posterior and an identifiability label, and predictions carry intervals.

Measured ``engine.csv`` / ``power.csv`` are used ONLY as calibration targets and
validation ground truth (``ingest.py``); they are never required model inputs.

See design plan: clean rewrite, validated on the gemma-4 staircase/validate/
agentic campaign AND the legacy ShareGPT campaign.
"""

from __future__ import annotations

__all__ = ["arch", "workload", "ingest", "priors", "estimate", "model"]
