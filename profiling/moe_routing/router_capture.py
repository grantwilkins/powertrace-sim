"""Capture and summarize token/layer MoE expert assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def topk_expert_ids(router_logits, top_k: int) -> np.ndarray:
    """Return expert IDs sorted by descending router score."""
    values = np.asarray(router_logits)
    if values.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    if not 0 < top_k <= values.shape[1]:
        raise ValueError("top_k must be within the expert dimension")
    partial = np.argpartition(values, -top_k, axis=1)[:, -top_k:]
    scores = np.take_along_axis(values, partial, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(partial, order, axis=1).astype(np.int32)


def distinct_expert_curve(expert_ids: np.ndarray, batch_sizes) -> np.ndarray:
    """Mean distinct experts per layer for contiguous token groups of size B."""
    ids = np.asarray(expert_ids)
    if ids.ndim != 3:
        raise ValueError("expert_ids must have shape [tokens, layers, top_k]")
    curves = []
    for batch in batch_sizes:
        batch = int(batch)
        if batch <= 0:
            raise ValueError("batch sizes must be positive")
        groups = []
        for start in range(0, ids.shape[0] - batch + 1, batch):
            block = ids[start:start + batch]
            groups.append([
                len(np.unique(block[:, layer, :]))
                for layer in range(ids.shape[1])
            ])
        if not groups:
            raise ValueError(f"batch size {batch} exceeds captured tokens")
        curves.append(np.mean(groups, axis=0))
    return np.asarray(curves, dtype=np.float64).T


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_samples(path: Path) -> list[dict]:
    samples = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "id" not in item or "prompt_text" not in item:
            raise ValueError(
                f"{path}:{line_number} requires id and prompt_text"
            )
        item.setdefault("completion_text", "")
        samples.append(item)
    if not samples:
        raise ValueError("routing input has no samples")
    return samples


def capture(model_name: str, input_jsonl: str, output_npz: str) -> None:
    """Teacher-force text and save real router assignments, never a proxy."""
    import torch
    import transformers

    source = Path(input_jsonl)
    samples = _load_samples(source)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    top_k = int(getattr(model.config, "num_experts_per_tok", 0))
    if top_k <= 0:
        raise ValueError("model config has no positive num_experts_per_tok")

    assignments, sample_index, token_index, phases = [], [], [], []
    with torch.inference_mode():
        for sample_i, item in enumerate(samples):
            prompt_ids = tokenizer(
                item["prompt_text"], add_special_tokens=True
            )["input_ids"]
            completion_ids = tokenizer(
                item["completion_text"], add_special_tokens=False
            )["input_ids"]
            input_ids = torch.tensor(
                [prompt_ids + completion_ids], dtype=torch.long,
                device=model.device,
            )
            outputs = model(
                input_ids=input_ids, use_cache=False, output_router_logits=True
            )
            router_logits = getattr(outputs, "router_logits", None)
            if not router_logits:
                raise ValueError(
                    "model did not return router_logits with "
                    "output_router_logits=True"
                )
            layers = [
                topk_expert_ids(logits.detach().float().cpu().reshape(
                    -1, logits.shape[-1]
                ).numpy(), top_k)
                for logits in router_logits
            ]
            token_count = len(prompt_ids) + len(completion_ids)
            if any(layer.shape[0] != token_count for layer in layers):
                raise ValueError("router-logit token axis does not match input IDs")
            assignments.append(np.stack(layers, axis=1))
            sample_index.extend([sample_i] * token_count)
            token_index.extend(range(token_count))
            phases.extend(
                [0] * len(prompt_ids) + [1] * len(completion_ids)
            )

    expert_ids = np.concatenate(assignments, axis=0)
    batches = [size for size in (1, 2, 4, 8, 16, 32, 64, 128)
               if size <= expert_ids.shape[0]]
    curve = distinct_expert_curve(expert_ids, batches)
    destination = Path(output_npz)
    np.savez_compressed(
        destination,
        expert_ids=expert_ids,
        sample_index=np.asarray(sample_index, dtype=np.int32),
        token_index=np.asarray(token_index, dtype=np.int32),
        phase=np.asarray(phases, dtype=np.int8),
        batch_sizes=np.asarray(batches, dtype=np.int32),
        distinct_experts=curve,
    )
    manifest = {
        "schema_version": 1,
        "model": model_name,
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "artifact": str(destination),
        "tokens": int(expert_ids.shape[0]),
        "layers": int(expert_ids.shape[1]),
        "top_k": int(expert_ids.shape[2]),
        "phase_codes": {"prefill": 0, "decode": 1},
        "batch_sizes": batches,
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args()
    capture(args.model, args.input_jsonl, args.output_npz)
    print(args.output_npz)


if __name__ == "__main__":
    main()
