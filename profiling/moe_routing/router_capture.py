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
    """Mean distinct experts per layer for contiguous prefill token groups."""
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


def decode_distinct_expert_curve(
    completion_assignments: list[np.ndarray], batch_sizes
) -> np.ndarray:
    """Mean distinct experts for one decode token from each active sequence."""
    if not completion_assignments:
        raise ValueError("decode curve requires completion assignments")
    layers = completion_assignments[0].shape[1]
    curves = []
    for batch in batch_sizes:
        batch = int(batch)
        if batch <= 0:
            raise ValueError("batch sizes must be positive")
        groups = []
        for start in range(0, len(completion_assignments) - batch + 1, batch):
            sequences = completion_assignments[start:start + batch]
            steps = min(sequence.shape[0] for sequence in sequences)
            for position in range(steps):
                token_group = np.stack(
                    [sequence[position] for sequence in sequences]
                )
                groups.append([
                    len(np.unique(token_group[:, layer, :]))
                    for layer in range(layers)
                ])
        if not groups:
            raise ValueError(f"batch size {batch} exceeds captured sequences")
        curves.append(np.mean(groups, axis=0))
    return np.asarray(curves, dtype=np.float64).T


def router_top_k(config) -> int:
    """Resolve routed experts/token for flat and nested model configs."""
    get_text_config = getattr(config, "get_text_config", None)
    text_config = (
        get_text_config() if callable(get_text_config)
        else getattr(config, "text_config", config)
    )
    for candidate in (text_config, config):
        for name in ("num_experts_per_tok", "top_k_experts"):
            value = int(getattr(candidate, name, 0) or 0)
            if value > 0:
                return value
    raise ValueError("model config has no positive routed-experts-per-token field")


def forward_memory_kwargs(config) -> dict:
    """Avoid materializing Gemma 4 full-vocabulary logits during routing capture."""
    if getattr(config, "model_type", "") == "gemma4":
        return {"logits_to_keep": 1}
    return {}


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


def tokenize_samples(tokenizer, samples: list[dict], max_tokens: int) -> list[tuple]:
    """Tokenize and bound every sample before loading model weights."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    prepared = []
    for item in samples:
        prompt_ids = tokenizer(
            item["prompt_text"], add_special_tokens=True
        )["input_ids"]
        completion_ids = tokenizer(
            item["completion_text"], add_special_tokens=False
        )["input_ids"]
        token_count = len(prompt_ids) + len(completion_ids)
        if token_count > max_tokens:
            raise ValueError(
                f"{item['id']} has {token_count} tokens, exceeds max_tokens "
                f"{max_tokens}"
            )
        prepared.append((item, prompt_ids, completion_ids))
    return prepared


def capture(
    model_name: str, input_jsonl: str, output_npz: str, *, max_tokens: int = 4096
) -> None:
    """Teacher-force text and save real router assignments, never a proxy."""
    import torch
    import transformers

    source = Path(input_jsonl)
    samples = _load_samples(source)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    prepared = tokenize_samples(tokenizer, samples, max_tokens)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    model.eval()
    top_k = router_top_k(model.config)

    assignments, sample_index, token_index, phases = [], [], [], []
    prompt_assignments, completion_assignments = [], []
    with torch.inference_mode():
        for sample_i, (item, prompt_ids, completion_ids) in enumerate(prepared):
            token_count = len(prompt_ids) + len(completion_ids)
            input_ids = torch.tensor(
                [prompt_ids + completion_ids], dtype=torch.long,
                device=model.device,
            )
            outputs = model(
                input_ids=input_ids, use_cache=False, output_router_logits=True,
                **forward_memory_kwargs(model.config),
            )
            router_logits = getattr(outputs, "router_logits", None)
            if router_logits is None or len(router_logits) == 0:
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
            if any(layer.shape[0] != token_count for layer in layers):
                raise ValueError("router-logit token axis does not match input IDs")
            assignment = np.stack(layers, axis=1)
            assignments.append(assignment)
            prompt_assignments.append(assignment[:len(prompt_ids)])
            if completion_ids:
                completion_assignments.append(assignment[len(prompt_ids):])
            sample_index.extend([sample_i] * token_count)
            token_index.extend(range(token_count))
            phases.extend(
                [0] * len(prompt_ids) + [1] * len(completion_ids)
            )

    expert_ids = np.concatenate(assignments, axis=0)
    prefill_ids = np.concatenate(prompt_assignments, axis=0)
    prefill_batches = [
        size for size in (1, 4, 16, 64, 256)
        if size <= prefill_ids.shape[0]
    ]
    prefill_curve = distinct_expert_curve(prefill_ids, prefill_batches)
    decode_batches = [
        size for size in (1, 4, 16, 64, 128)
        if size <= len(completion_assignments)
    ]
    decode_curve = (
        decode_distinct_expert_curve(completion_assignments, decode_batches)
        if decode_batches else np.empty((expert_ids.shape[1], 0))
    )
    destination = Path(output_npz)
    np.savez_compressed(
        destination,
        expert_ids=expert_ids,
        sample_index=np.asarray(sample_index, dtype=np.int32),
        token_index=np.asarray(token_index, dtype=np.int32),
        phase=np.asarray(phases, dtype=np.int8),
        prefill_group_sizes=np.asarray(prefill_batches, dtype=np.int32),
        prefill_distinct_experts=prefill_curve,
        decode_batch_sizes=np.asarray(decode_batches, dtype=np.int32),
        decode_distinct_experts=decode_curve,
    )
    manifest = {
        "schema_version": 2,
        "model": model_name,
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "artifact": str(destination),
        "tokens": int(expert_ids.shape[0]),
        "layers": int(expert_ids.shape[1]),
        "top_k": int(expert_ids.shape[2]),
        "max_tokens_per_sample": int(max_tokens),
        "phase_codes": {"prefill": 0, "decode": 1},
        "prefill_group_sizes": prefill_batches,
        "decode_batch_sizes": decode_batches,
        "curve_semantics": {
            "prefill": "contiguous teacher-forced tokens",
            "decode": "one completion token per active sequence and position",
        },
    }
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    capture(
        args.model, args.input_jsonl, args.output_npz,
        max_tokens=args.max_tokens,
    )
    print(args.output_npz)


if __name__ == "__main__":
    main()
