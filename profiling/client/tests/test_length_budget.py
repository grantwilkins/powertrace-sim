"""Loader length and prompt-accounting behavior.

Claim:
Pruning follows the configured context window, and synthetic requests report
the token count of the actual text sent to the server.

Plausible wrong implementations:
- Retain the historical 1024-token cap after configuring a long context.
- Report the sampled token-ID count even when decode/re-tokenize changes it.
- Count tokenizer-added special tokens that the server request does not add.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # profiling/client

import benchmark_dataset as bd  # noqa: E402


def _reset():
    bd._LENGTH_BUDGET["max_prompt_len"] = 1024
    bd._LENGTH_BUDGET["max_total_len"] = 2048


def test_default_caps_match_historical_behavior():
    _reset()
    assert bd.is_valid_sequence(800, 800)        # within 1024/2048
    assert not bd.is_valid_sequence(1500, 10)    # prompt > 1024 -> dropped (old behavior)


def test_configure_tracks_served_context_window():
    _reset()
    bd.configure_length_budget(32768)
    assert bd.is_valid_sequence(8000, 200)       # long prompt now kept
    assert bd.is_valid_sequence(30000, 1000)     # prompt+output <= 32768 kept
    assert not bd.is_valid_sequence(40000, 10)   # beyond the context -> still dropped
    _reset()


def test_configure_none_is_noop():
    _reset()
    bd.configure_length_budget(None)
    assert not bd.is_valid_sequence(1500, 10)    # unchanged from default
    _reset()


def test_random_dataset_reports_retokenized_prompt_length():
    class NonInvertibleTokenizer:
        vocab_size = 128

        def decode(self, token_ids):
            return "x" * len(token_ids)

        def __call__(self, prompt, *, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=list(range(len(prompt) - 3)))

    request = bd.RandomDataset().sample(
        NonInvertibleTokenizer(),
        num_requests=1,
        input_len=8,
        output_len=2,
        range_ratio=0.0,
    )[0]

    assert request.prompt_len == 5
