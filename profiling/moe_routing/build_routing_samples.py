"""Build balanced real-text ShareGPT and SWE-smith inputs for router capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SWE_SMITH_DATASET = "SWE-bench/SWE-smith-trajectories"
SWE_SMITH_SPLIT = "tool"


def _text(message: dict) -> str:
    content = message.get("content", message.get("value", ""))
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def _role(message: dict) -> str:
    role = message.get("role", message.get("from", ""))
    return {"human": "user", "gpt": "assistant"}.get(role, role)


def _assistant_text(message: dict) -> str:
    calls = message.get("tool_calls") or []
    action = json.dumps(
        [call.get("function", {}) for call in calls],
        sort_keys=True, separators=(",", ":"),
    ) if calls else ""
    return "\n".join(filter(None, (_text(message), action)))


def sample_from_messages(messages, *, sample_id: str, source: str) -> dict | None:
    if isinstance(messages, str):
        messages = json.loads(messages)
    assistant_index = next(
        (
            index for index, message in enumerate(messages)
            if index > 0 and _role(message) == "assistant"
            and _assistant_text(message)
        ),
        None,
    )
    if assistant_index is None:
        return None
    prompt = "\n".join(
        f"{_role(message)}: {_text(message)}"
        for message in messages[:assistant_index] if _text(message)
    )
    completion = _assistant_text(messages[assistant_index])
    if not prompt or not completion:
        return None
    return {
        "id": f"{source}:{sample_id}",
        "source": source,
        "prompt_text": prompt,
        "completion_text": completion,
    }


def select_samples(rows, *, source: str, n: int, seed: int) -> list[dict]:
    candidates = []
    for index, row in enumerate(rows):
        messages = row.get("messages", row.get("conversations"))
        if messages is None:
            continue
        sample_id = str(
            row.get("instance_id") or row.get("id") or row.get("conversation_id")
            or index
        )
        sample = sample_from_messages(
            messages, sample_id=sample_id, source=source
        )
        if sample is None:
            continue
        rank = hashlib.sha256(f"{seed}:{sample['id']}".encode()).hexdigest()
        candidates.append((rank, sample))
    selected = [sample for _, sample in sorted(candidates)[:n]]
    if len(selected) != n:
        raise ValueError(f"{source} has {len(selected)} usable samples, need {n}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharegpt", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--n-per-source", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    sharegpt = json.loads(Path(args.sharegpt).read_text())
    swe_smith = load_dataset(
        SWE_SMITH_DATASET, split=SWE_SMITH_SPLIT
    )
    samples = select_samples(
        sharegpt, source="sharegpt", n=args.n_per_source, seed=args.seed
    )
    samples += select_samples(
        swe_smith, source="swe_smith", n=args.n_per_source, seed=args.seed
    )
    destination = Path(args.output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples)
    )
    print(destination)


if __name__ == "__main__":
    main()
