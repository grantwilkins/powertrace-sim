"""
Claim:
Each routing sample is a real prefix paired with its immediately following
assistant completion, selected deterministically and exactly by source.

Plausible wrong implementations:
- Include the target assistant response in both prompt and completion.
- Pair a prompt with a later assistant response instead of the following one.
- Shuffle nondeterministically or emit duplicate/missing source samples.
- Accept fewer usable samples than the declared balanced stratum size.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_routing_samples import select_samples  # noqa: E402


def test_selects_immediately_following_assistant_without_leakage():
    rows = [{
        "id": "conversation-1",
        "conversations": [
            {"from": "human", "value": "first question"},
            {"from": "gpt", "value": "first answer"},
            {"from": "human", "value": "second question"},
            {"from": "gpt", "value": "second answer"},
        ],
    }]
    sample = select_samples(rows, source="sharegpt", n=1, seed=0)[0]
    assert sample["prompt_text"] == "user: first question"
    assert sample["completion_text"] == "first answer"
    assert "first answer" not in sample["prompt_text"]
    assert "second answer" not in sample["completion_text"]


def test_selection_is_deterministic_exact_and_source_distinct():
    rows = [
        {
            "instance_id": str(index),
            "messages": [
                {"role": "user", "content": f"task {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
        }
        for index in range(4)
    ]
    first = select_samples(rows, source="swe_smith", n=3, seed=7)
    second = select_samples(list(reversed(rows)), source="swe_smith", n=3, seed=7)
    assert first == second
    assert len({sample["id"] for sample in first}) == 3
    with pytest.raises(ValueError, match="need 5"):
        select_samples(rows, source="swe_smith", n=5, seed=7)
