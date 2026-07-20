"""Normalize raw expert assignments into phase/source routing laws."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

PREFILL_SIZES = (1, 4, 16, 64, 256)
DECODE_SIZES = (1, 4, 16, 32, 64)
SOURCE_RANGES = {"sharegpt": (0, 64), "swe_smith": (64, 128)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def touch_probability(expert_ids: np.ndarray, n_experts: int) -> np.ndarray:
    """Per-token probability that each layer expert is among the top-k."""
    return np.stack([
        np.bincount(expert_ids[:, layer].ravel(), minlength=n_experts)
        / expert_ids.shape[0]
        for layer in range(expert_ids.shape[1])
    ])


def contiguous_distinct(
    expert_ids: np.ndarray, sizes: tuple[int, ...],
) -> np.ndarray:
    curves = []
    for size in sizes:
        groups = expert_ids[:expert_ids.shape[0] // size * size]
        groups = groups.reshape(-1, size, expert_ids.shape[1], expert_ids.shape[2])
        curves.append(np.stack([
            np.mean([np.unique(group[:, layer]).size for group in groups])
            for layer in range(expert_ids.shape[1])
        ]))
    return np.stack(curves, axis=1)


def decode_distinct(
    expert_ids: np.ndarray, sample_index: np.ndarray,
    phase: np.ndarray, samples: range, sizes: tuple[int, ...],
) -> np.ndarray:
    completions = [
        expert_ids[(sample_index == sample) & (phase == 1)]
        for sample in samples
    ]
    curves = []
    for size in sizes:
        groups = []
        for start in range(0, len(completions) - size + 1, size):
            block = completions[start:start + size]
            for position in range(min(len(sequence) for sequence in block)):
                tokens = np.stack([sequence[position] for sequence in block])
                groups.append([
                    np.unique(tokens[:, layer]).size
                    for layer in range(expert_ids.shape[1])
                ])
        curves.append(np.mean(groups, axis=0))
    return np.stack(curves, axis=1)


def fit_alpha(
    probability: np.ndarray, sizes: tuple[int, ...], observed: np.ndarray,
) -> tuple[float, float]:
    def loss(alpha: float) -> float:
        predicted = np.stack([
            np.sum(1.0 - (1.0 - probability) ** (size ** alpha), axis=1)
            for size in sizes
        ], axis=1)
        return float(np.mean((predicted - observed) ** 2))

    result = minimize_scalar(loss, bounds=(0.1, 1.5), method="bounded")
    return float(result.x), float(np.sqrt(result.fun))


def phase_record(
    ids: np.ndarray, sizes: tuple[int, ...], observed: np.ndarray,
    n_experts: int,
) -> dict:
    probability = touch_probability(ids, n_experts)
    alpha, rmse = fit_alpha(probability, sizes, observed)
    load = probability / probability.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(load > 0.0, load * np.log(load), 0.0), axis=1)
    return {
        "group_sizes": list(sizes),
        "observed_distinct_experts": observed.tolist(),
        "touch_probability": probability.tolist(),
        "alpha": alpha,
        "distinct_expert_rmse": rmse,
        "normalized_load_entropy": (entropy / np.log(n_experts)).tolist(),
    }


def build(captures: list[Path]) -> dict:
    output = {
        "schema_version": "moe-routing-laws-v1",
        "source_binding": (
            "sample indices 0:64=sharegpt and 64:128=swe_smith from the "
            "deterministic build_routing_samples.py append order"
        ),
        "models": {},
    }
    for path in captures:
        data = np.load(path)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        ids = data["expert_ids"]
        sample_index = data["sample_index"]
        phase = data["phase"]
        if ids.shape[0] != int(manifest["tokens"]):
            raise ValueError(f"{path} token count disagrees with its manifest")
        if ids.shape[1] != int(manifest["layers"]):
            raise ValueError(f"{path} layer count disagrees with its manifest")
        if ids.shape[2] != int(manifest["top_k"]):
            raise ValueError(f"{path} top-k disagrees with its manifest")
        if not np.array_equal(np.unique(sample_index), np.arange(128)):
            raise ValueError(
                f"{path} cannot satisfy the declared 64/64 source binding")
        if np.any(np.diff(np.sort(ids, axis=2), axis=2) == 0):
            raise ValueError(f"{path} contains duplicate experts within top-k")
        model = str(manifest["model"]).split("/")[-1]
        n_experts = int(ids.max()) + 1
        model_record = {
            "capture_sha256": sha256(path),
            "source_sha256": manifest["source_sha256"],
            "layers": int(ids.shape[1]),
            "n_experts": n_experts,
            "top_k": int(ids.shape[2]),
            "sources": {},
        }
        for source, (start, stop) in SOURCE_RANGES.items():
            selected = (sample_index >= start) & (sample_index < stop)
            prefill_ids = ids[selected & (phase == 0)]
            decode_ids = ids[selected & (phase == 1)]
            prefill_curve = contiguous_distinct(prefill_ids, PREFILL_SIZES)
            decode_curve = decode_distinct(
                ids, sample_index, phase, range(start, stop), DECODE_SIZES)
            model_record["sources"][source] = {
                "tokens": int(selected.sum()),
                "prefill": phase_record(
                    prefill_ids, PREFILL_SIZES, prefill_curve, n_experts),
                "decode": phase_record(
                    decode_ids, DECODE_SIZES, decode_curve, n_experts),
            }
        output["models"][model] = model_record
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-dir", default="results/moe_routing", type=Path)
    parser.add_argument(
        "--out", default="results/moe_routing/routing-laws.json", type=Path)
    args = parser.parse_args()
    captures = sorted(args.capture_dir.glob("gpt-oss-*-routing.npz"))
    if not captures:
        raise ValueError(f"No GPT-OSS routing captures under {args.capture_dir}")
    args.out.write_text(json.dumps(build(captures), indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
