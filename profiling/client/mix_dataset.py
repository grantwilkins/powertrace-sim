"""Build an equal random mix of real datasets in ShareGPT JSON form.

Streams small random portions of one or more HuggingFace datasets, extracts
(prompt, completion) pairs, takes an EQUAL count from each source, shuffles them
together, and writes a ShareGPT-format JSON. That file feeds the vendored
``benchmark_serving.py`` unchanged via ``--dataset-name sharegpt --dataset-path``
— the loader reads ``conversations[0]['value']`` / ``[1]['value']`` and derives
prompt/output lengths by tokenizing them.

Currently wired sources (no HF agreement required):
  allenai/WildChat            — real, current, multilingual chat
  nvidia/OpenCodeInstruct     — code instruction/response

    python -m profiling.client.mix_dataset --out mixed.json --per-source 150 \
        --source allenai/WildChat --source nvidia/OpenCodeInstruct

``to_sharegpt_record`` / ``equal_random_mix`` are pure (unit-tested); the HF
streaming in ``stream_pairs`` is the live layer.
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import islice


def to_sharegpt_record(prompt: str, completion: str, source: str) -> dict:
    """One ShareGPT entry (two turns). Extra keys are ignored by the loader."""
    return {
        "source": source,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": completion},
        ],
    }


def equal_random_mix(by_source: dict, seed: int = 0) -> list:
    """Truncate every source to the smallest count, concatenate, shuffle (pure)."""
    if not by_source:
        return []
    n = min(len(v) for v in by_source.values())
    rng = random.Random(seed)
    mixed = []
    for source in sorted(by_source):
        recs = list(by_source[source])
        rng.shuffle(recs)
        mixed.extend(recs[:n])
    rng.shuffle(mixed)
    return mixed


def _wildchat_pair(row):
    conv = row.get("conversation") or row.get("conversations") or []
    user = asst = None
    for t in conv:
        role = t.get("role") or t.get("from")
        content = (t.get("content") or t.get("value") or "")
        if role in ("user", "human") and user is None:
            user = content
        elif role in ("assistant", "gpt") and user is not None:
            asst = content
            break
    return user, asst


def _first_key(row, keys):
    for k in keys:
        if k in row and row[k]:
            return row[k]
    return None


def _opencode_pair(row):
    prompt = _first_key(row, ("input", "instruction", "problem", "question", "prompt"))
    completion = _first_key(row, ("output", "response", "solution", "completion", "answer"))
    return prompt, completion


EXTRACTORS = {
    "allenai/WildChat": _wildchat_pair,
    "allenai/WildChat-1M": _wildchat_pair,
    "nvidia/OpenCodeInstruct": _opencode_pair,
}


def _valid(prompt, completion) -> bool:
    return bool(prompt and completion
                and len(prompt.strip()) >= 8 and len(completion.strip()) >= 8)


def stream_pairs(path: str, n: int, seed: int = 0, split: str = "train",
                 buffer_size: int = 10000) -> list:
    """Stream a shuffled prefix of `path`, return `n` valid ShareGPT records."""
    from datasets import load_dataset

    extract = EXTRACTORS.get(path, _opencode_pair)
    ds = load_dataset(path, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    out = []
    for row in islice(ds, buffer_size):
        prompt, completion = extract(row)
        if _valid(prompt, completion):
            out.append(to_sharegpt_record(prompt.strip(), completion.strip(), path))
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f"{path}: only {len(out)}/{n} valid pairs in {buffer_size} streamed rows")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True,
                    help="HF dataset path; repeat for each source")
    ap.add_argument("--per-source", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_source = {}
    for src in args.source:
        recs = stream_pairs(src, args.per_source, seed=args.seed)
        by_source[src] = recs
        plen = sum(len(r["conversations"][0]["value"]) for r in recs) / len(recs)
        print(f"{src}: {len(recs)} pairs (mean prompt chars {plen:.0f})")

    mixed = equal_random_mix(by_source, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False)
    counts = {}
    for r in mixed:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(f"wrote {len(mixed)} records -> {args.out}  balance={counts}")


if __name__ == "__main__":
    main()
