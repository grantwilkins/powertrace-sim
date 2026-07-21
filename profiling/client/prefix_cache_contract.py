"""Validate observed prefix-cache counters against the declared treatment."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def validate_prefix_cache_mode(
    engine_csv: str | Path, prefix_cache: bool
) -> dict:
    with Path(engine_csv).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if (
            not reader.fieldnames
            or "prefix_cache_hits_total" not in reader.fieldnames
        ):
            raise ValueError("engine.csv is missing prefix_cache_hits_total")
        hits = [
            float(row["prefix_cache_hits_total"])
            for row in reader
            if row.get("prefix_cache_hits_total") not in (None, "")
        ]
    if len(hits) < 2 or not all(math.isfinite(value) for value in hits):
        raise ValueError("engine.csv requires finite prefix-cache hit counters")
    deltas = [right - left for left, right in zip(hits, hits[1:])]
    if any(delta < 0 for delta in deltas):
        raise ValueError("engine.csv prefix-cache hit counter decreased")
    increase = hits[-1] - hits[0]
    if not prefix_cache and increase > 0:
        raise ValueError(
            f"cache-off run observed {increase:.0f} prefix-cache hit tokens"
        )
    if prefix_cache and increase <= 0:
        raise ValueError("cache-on run observed no prefix-cache hit tokens")
    return {
        "declared_enabled": bool(prefix_cache),
        "first_hits": hits[0],
        "last_hits": hits[-1],
        "hit_token_increase": increase,
        "status": "validated",
    }
