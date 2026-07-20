"""
Claim:
Expansion evaluation binds the recorded engine token/sequence limits and
charges only uncached prompt tokens while preserving cached context.

Plausible wrong implementations:
- Fall back to the legacy 2048-token budget for an 8192-token run.
- Subtract planned rather than measured cached tokens.
- Remove cached tokens from both prefill work and attention context.
- Reorder request metadata differently from arrival timestamps.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluate_expansion import engine_from_manifest, request_schedule  # noqa: E402


def test_engine_configuration_is_bound_from_manifest():
    engine = engine_from_manifest({
        "server": {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.85,
        }
    })
    assert engine.max_num_seqs == 128
    assert engine.chunk_budget_tokens == 8192
    assert engine.gpu_memory_utilization == 0.85


def test_cached_prompt_tokens_become_initial_context():
    record = SimpleNamespace(
        input_lens=np.asarray([100, 80]),
        output_lens=np.asarray([3, 4]),
        request_timestamps=np.asarray([12.0, 10.0]),
        request_table={
            "cached_prompt_tokens": np.asarray([64, 0], dtype=object),
        },
        provenance={"request_projection_indices": [0, 1]},
    )
    requests, order, origin = request_schedule(record)
    assert order.tolist() == [1, 0]
    assert origin.tolist() == [10.0]
    assert requests == [
        (0.0, 80, 4, 0),
        (2.0, 36, 3, 64),
    ]
