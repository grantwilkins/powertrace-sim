"""
Claim:
Expansion evaluation binds the recorded engine token/sequence limits and
charges only uncached prompt tokens while preserving cached context. It uses
the measured horizon, applies clean dense power to unseen dense arrivals, and
fails closed for an unsupported MoE architecture or sealed bundle.

Plausible wrong implementations:
- Fall back to the legacy 2048-token budget for an 8192-token run.
- Subtract planned rather than measured cached tokens.
- Remove cached tokens from both prefill work and attention context.
- Reorder request metadata differently from arrival timestamps.
- End power evaluation when the simulation drains early.
- Route an unseen MoE checkpoint through dense or GPT-OSS coefficients.
- Open sealed bundles during retrospective development.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluate_expansion import (  # noqa: E402
    _clean_prediction,
    _timing_summary,
    development_run_dirs,
    engine_from_manifest,
    measured_horizon_s,
    power_coverage,
    request_schedule,
)


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


def test_measured_bundle_horizon_does_not_depend_on_simulated_completion():
    record = SimpleNamespace(
        request_timestamps=np.asarray([100.0, 104.0]),
        ttfts=np.asarray([2.0, 1.0]),
        decode_times=np.asarray([8.0, 3.0]),
    )

    assert measured_horizon_s(record, 100.0) == 10.0


def test_zero_length_decode_is_excluded_from_relative_error():
    rows = [
        {
            "measured_ttft_s": 1.0, "predicted_ttft_s": 1.0,
            "measured_decode_s": 0.0, "predicted_decode_s": 0.1,
            "measured_e2e_s": 1.0, "predicted_e2e_s": 1.1,
        },
        {
            "measured_ttft_s": 2.0, "predicted_ttft_s": 2.0,
            "measured_decode_s": 1.0, "predicted_decode_s": 1.1,
            "measured_e2e_s": 3.0, "predicted_e2e_s": 3.1,
        },
    ]

    summary = _timing_summary(rows)

    assert np.isclose(summary["decode_s_medabs_pct"], 10.0)


def test_sparse_power_cadence_cannot_grade_temporal_metrics():
    measured = np.ones(64 * 4)
    measured.reshape(-1, 4)[::10] = np.nan

    coverage = power_coverage(measured, 0.25)

    assert np.isclose(coverage["missing_one_second_fraction"], 7 / 64)
    assert isinstance(coverage["temporal_supported"], bool)
    assert not coverage["temporal_supported"]


def test_long_power_gap_cannot_grade_temporal_metrics():
    measured = np.ones(100 * 4)
    measured[40:52] = np.nan

    coverage = power_coverage(measured, 0.25)

    assert coverage["missing_one_second_fraction"] == 0.03
    assert coverage["maximum_power_gap_s"] == 3.0
    assert not coverage["temporal_supported"]


def test_unseen_moe_checkpoint_fails_closed():
    record = SimpleNamespace(
        arch={"family": "moe-8b"},
        model="gemma-moe",
        hardware="A100",
    )
    artifact = {"moe": {"per_model": {"gpt-oss-20b": {}}}}

    prediction, surface, supported, reason = _clean_prediction(
        record, {}, artifact, 0.25
    )

    assert prediction is None
    assert surface == "moe"
    assert not supported
    assert "architecture-specific" in reason


def test_development_discovery_excludes_sealed_and_smoke(tmp_path):
    def write(campaign, run, payload):
        path = tmp_path / campaign / run
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps(payload))
        return path

    development = write(
        "arrival_shape", "dev", {"probe": {"validation_role": "development"}}
    )
    trace = write("agent_trace", "trace", {"probe": {"type": "trace_replay"}})
    write("sealed_model", "sealed", {"validation_role": "sealed"})
    write("trace_smoke", "smoke", {"probe": {"type": "trace_replay"}})

    assert development_run_dirs(tmp_path) == [trace, development]
