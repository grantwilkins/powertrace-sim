"""Claim: routing evidence retains exact layer/token/expert assignments.

Plausible wrong implementations caught here: top-k along the token axis,
discarding the layer axis, grouping adjacent completion tokens as a decode
batch, missing Gemma's nested top-k field, or materializing full Gemma LM logits.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router_capture import (  # noqa: E402
    decode_distinct_expert_curve,
    distinct_expert_curve,
    forward_memory_kwargs,
    router_top_k,
    tokenize_samples,
    topk_expert_ids,
)


def test_topk_experts_are_per_token_and_score_ordered():
    logits = np.array([[0.1, 3.0, 2.0], [5.0, 1.0, 4.0]])
    np.testing.assert_array_equal(
        topk_expert_ids(logits, 2), [[1, 2], [0, 2]]
    )


def test_distinct_curve_preserves_layer_overlap():
    ids = np.array([
        [[0, 1], [0, 1]],
        [[0, 1], [2, 3]],
        [[2, 3], [0, 1]],
        [[2, 3], [2, 3]],
    ])
    curve = distinct_expert_curve(ids, [1, 2, 4])
    np.testing.assert_allclose(curve[:, 0], [2, 2])
    np.testing.assert_allclose(curve[:, 1], [2, 4])
    np.testing.assert_allclose(curve[:, 2], [4, 4])


def test_decode_curve_groups_across_sequences_at_the_same_position():
    sequences = [
        np.array([[[0]], [[1]]]),
        np.array([[[2]], [[2]]]),
    ]
    curve = decode_distinct_expert_curve(sequences, [1, 2])
    np.testing.assert_allclose(curve, [[1.0, 2.0]])


def test_gemma_nested_router_config_and_memory_bound_are_resolved():
    class TextConfig:
        top_k_experts = 8

    class GemmaConfig:
        model_type = "gemma4"
        text_config = TextConfig()

    config = GemmaConfig()
    assert router_top_k(config) == 8
    assert forward_memory_kwargs(config) == {"logits_to_keep": 1}


def test_flat_router_config_does_not_receive_gemma_only_forward_kwarg():
    class GptOssConfig:
        model_type = "gpt_oss"
        num_experts_per_tok = 4

    config = GptOssConfig()
    assert router_top_k(config) == 4
    assert forward_memory_kwargs(config) == {}


def test_sample_token_bound_is_checked_before_model_loading():
    class Tokenizer:
        def __call__(self, text, *, add_special_tokens):
            extra = 1 if add_special_tokens else 0
            return {"input_ids": list(range(len(text.split()) + extra))}

    samples = [{
        "id": "sample", "prompt_text": "one two",
        "completion_text": "three four",
    }]
    prepared = tokenize_samples(Tokenizer(), samples, max_tokens=5)
    assert len(prepared[0][1]) + len(prepared[0][2]) == 5
    with np.testing.assert_raises_regex(ValueError, "exceeds max_tokens 4"):
        tokenize_samples(Tokenizer(), samples, max_tokens=4)
