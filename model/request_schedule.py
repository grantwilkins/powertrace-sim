"""Strict request-schedule contract for selected-model inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    number = int(value)
    if float(value) != number or number < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return number


def _sample_output(spec: Mapping[str, object], rng: np.random.Generator) -> int:
    values = np.asarray(spec.get("values"), dtype=float)
    probabilities = np.asarray(spec.get("probabilities"), dtype=float)
    if values.ndim != 1 or not values.size or probabilities.shape != values.shape:
        raise ValueError("output_tokens_distribution needs equal non-empty vectors")
    if not np.isfinite(values).all() or not np.isfinite(probabilities).all():
        raise ValueError("output token distribution must be finite")
    if np.any(values < 1) or not np.array_equal(values, np.floor(values)):
        raise ValueError("output token values must be positive integers")
    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("output token probabilities must be non-negative and sum to one")
    return int(rng.choice(values.astype(int), p=probabilities))


def realize_requests(
    rows: Sequence[Mapping[str, object]], *, seed: int | None = None,
) -> tuple[list[dict[str, object]], bool]:
    if not rows:
        raise ValueError("request schedule is empty")
    rng = np.random.default_rng(seed)
    realized = []
    sampled = False
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"request[{index}] must be an object")
        try:
            arrival = float(row["arrival_time"])
            total_input = _integer(row["input_tokens"], field="input_tokens", minimum=1)
        except KeyError as exc:
            raise ValueError(f"request[{index}] missing {exc.args[0]!r}") from exc
        if not np.isfinite(arrival) or arrival < 0:
            raise ValueError("arrival_time must be finite and non-negative")
        cached = _integer(
            row.get("cached_prefix_tokens", 0),
            field="cached_prefix_tokens", minimum=0,
        )
        if cached > total_input:
            raise ValueError("cached_prefix_tokens cannot exceed input_tokens")
        fixed = "output_tokens" in row
        distributed = "output_tokens_distribution" in row
        if fixed == distributed:
            raise ValueError("request needs exactly one output token specification")
        if fixed:
            output = _integer(row["output_tokens"], field="output_tokens", minimum=1)
        else:
            spec = row["output_tokens_distribution"]
            if not isinstance(spec, Mapping):
                raise ValueError("output_tokens_distribution must be an object")
            output = _sample_output(spec, rng)
            sampled = True
        realized.append({
            "request_id": str(row.get("request_id", index)),
            "arrival_time": arrival,
            "input_tokens": total_input,
            "cached_prefix_tokens": cached,
            "executed_input_tokens": total_input - cached,
            "output_tokens": output,
        })
    return realized, sampled


def load_requests(path: str | Path, *, seed: int | None = None):
    payload = json.loads(Path(path).read_text())
    rows = payload if isinstance(payload, list) else payload.get("requests")
    if not isinstance(rows, list):
        raise ValueError("requests JSON must be a list or contain a requests list")
    return realize_requests(rows, seed=seed)

