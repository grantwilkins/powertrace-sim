"""Sample inter-turn (tool-execution) gaps from the per-tool-class gap model.

    gap_s ~ LogNormal(mu_t + b1_t * log(observation_tokens + 1), sigma_t)

``mu``/``b1``/``sigma`` are the fitted log-scale parameters in ``gap_params.json``
(``mu`` is used directly as the log-location — no mean-correction). Conditioning on
the observation size makes the gap explainable from the trace rather than noise:
token-heavy tool results take longer. Pure and deterministic given an ``rng``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import tool_classes

DEFAULT_PARAMS = Path(__file__).resolve().parent / "gap_params.json"


class GapSampler:
    def __init__(self, params: dict):
        self.classes = params["classes"]

    @classmethod
    def from_file(cls, path=DEFAULT_PARAMS) -> "GapSampler":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"gap params not found: {path}. Run openhands_gap_fit.py to fit "
                f"them, or point --gap-params at the checked-in priors.")
        return cls(json.loads(path.read_text()))

    def sample(self, tool_class: str, observation_tokens: int, rng) -> float:
        """Draw one gap (seconds). Unknown class falls back to ``bash`` (widest)."""
        p = self.classes.get(tool_class) or self.classes[tool_classes.BASH]
        mu = p["mu"] + p["b1"] * math.log(observation_tokens + 1)
        return float(rng.lognormal(mu, p["sigma"]))
