"""Phase-aware mixture-of-experts routing laws and weight traffic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAWS = REPO_ROOT / "results" / "moe_routing" / "routing-laws.json"


@dataclass(frozen=True, eq=False)
class RoutingLaw:
    model: str
    source: str
    top_k: int
    n_experts: int
    prefill_alpha: float
    decode_alpha: float
    prefill_touch_probability: np.ndarray
    decode_touch_probability: np.ndarray

    @lru_cache(maxsize=16384)
    def touched_fraction(
        self, decode_tokens: float, prefill_tokens: float,
        prefill_groups: float = 1.0,
    ) -> float:
        """Expected fraction of layer experts touched by one mixed iteration."""
        decode = max(float(decode_tokens), 0.0)
        prefill = max(float(prefill_tokens), 0.0)
        groups = max(float(prefill_groups), 1.0) if prefill > 0.0 else 0.0
        prefill_exponent = (
            groups * (prefill / groups) ** self.prefill_alpha
            if prefill > 0.0 else 0.0
        )
        decode_exponent = decode ** self.decode_alpha if decode > 0.0 else 0.0
        absent = (
            (1.0 - self.prefill_touch_probability) ** prefill_exponent
            * (1.0 - self.decode_touch_probability) ** decode_exponent
        )
        return float(np.mean(1.0 - absent))


def expected_iteration_weight_bytes(
    arch: Mapping[str, object], *, decode_tokens: float = 0.0,
    prefill_tokens: float = 0.0, prefill_groups: float = 1.0,
    routing_law: RoutingLaw | None = None,
) -> float:
    """Expected resident weights read once by a mixed engine iteration."""
    decode = max(float(decode_tokens), 0.0)
    prefill = max(float(prefill_tokens), 0.0)
    if decode + prefill == 0.0:
        return 0.0
    weights = float(arch["w_bytes"])
    moe_fraction = float(arch.get("moe_frac", 0.0) or 0.0)
    if moe_fraction <= 0.0:
        return weights
    if routing_law is None:
        experts = float(arch["n_experts"])
        draws = decode + prefill
        touched = 1.0 - (
            1.0 - float(arch["top_k"]) / experts
        ) ** draws
    else:
        if routing_law.n_experts != int(arch["n_experts"]):
            raise ValueError("Routing law expert count does not match architecture")
        if routing_law.top_k != int(arch["top_k"]):
            raise ValueError("Routing law top-k does not match architecture")
        touched = routing_law.touched_fraction(
            decode, prefill, prefill_groups)
    return weights * ((1.0 - moe_fraction) + moe_fraction * touched)


@lru_cache(maxsize=None)
def load_routing_laws(
    path: str | Path = DEFAULT_LAWS, source: str = "sharegpt",
) -> dict[str, RoutingLaw]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != "moe-routing-laws-v1":
        raise ValueError("Unknown MoE routing-law schema")
    laws = {}
    for model, model_record in payload["models"].items():
        record = model_record["sources"][source]
        laws[model] = RoutingLaw(
            model=model,
            source=source,
            top_k=int(model_record["top_k"]),
            n_experts=int(model_record["n_experts"]),
            prefill_alpha=float(record["prefill"]["alpha"]),
            decode_alpha=float(record["decode"]["alpha"]),
            prefill_touch_probability=np.asarray(
                record["prefill"]["touch_probability"], dtype=float),
            decode_touch_probability=np.asarray(
                record["decode"]["touch_probability"], dtype=float),
        )
    return laws

