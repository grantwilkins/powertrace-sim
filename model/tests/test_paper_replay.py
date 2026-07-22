"""
Claim:
Paper replay applies the frozen release power law independently per measured
run and reports matched one-second values at the per-GPU level.

Plausible wrong implementations:
- Apply meter response across concatenated runs and leak one run into the next.
- Compare TP-summed node watts against per-GPU measurements.
- Average the wrong number of native 250 ms bins into a reported second.
- Silently score a run absent from the declared split.
"""

import numpy as np
import pytest

from model.paper_replay import (
    LEDGER_CHANNELS,
    one_second_per_gpu,
    predict_cache,
    score_cache,
)
from model.release import load_artifact


def _cache(run_ids=(0, 1), tp=1):
    bins = 4 * len(run_ids)
    ids = np.repeat(run_ids, 4)
    cache = {
        "run_id": ids,
        "power": np.full(bins, 100.0 * tp),
        "tp": np.full(bins, tp),
        "rate": np.ones(bins),
        "model_idx": np.zeros(bins, dtype=int),
        "model_names": np.asarray(["llama-3-8b"]),
        "hw_idx": np.zeros(bins, dtype=int),
        "hw_names": np.asarray(["A100"]),
        "family_idx": np.zeros(bins, dtype=int),
        "family_names": np.asarray(["dense-8b"]),
        "dt_s": np.asarray(0.25),
    }
    for name in LEDGER_CHANNELS:
        cache[name] = np.zeros(bins)
    cache["busy"][:4] = 1.0
    cache["prefill_gemm_flops_rate"][:4] = 1.0e15
    cache["gemm_flops_rate"][:4] = 1.0e15
    return cache


def test_prediction_resets_meter_response_at_run_boundary():
    artifact = load_artifact()
    cache = _cache()

    predicted, _ = predict_cache(cache, artifact)

    idle = artifact["power"]["dense"]["A100"]["coefficients"][0]
    assert predicted[4] == pytest.approx(idle)
    assert predicted[3] > idle


def test_idle_node_power_is_tp_sum_but_reported_trace_is_per_gpu():
    artifact = load_artifact()
    cache = _cache(run_ids=(0,), tp=2)
    cache["busy"][:] = 0.0
    cache["prefill_gemm_flops_rate"][:] = 0.0
    cache["gemm_flops_rate"][:] = 0.0

    predicted, _ = predict_cache(cache, artifact)
    observed, reported = one_second_per_gpu(
        np.full(4, 200.0), predicted, tp=2, dt_s=0.25,
    )

    idle = artifact["power"]["dense"]["A100"]["coefficients"][0]
    np.testing.assert_allclose(predicted, 2.0 * idle)
    np.testing.assert_allclose(observed, [100.0])
    np.testing.assert_allclose(reported, [idle])


def test_one_second_average_uses_four_native_bins():
    observed, predicted = one_second_per_gpu(
        np.asarray([2.0, 4.0, 6.0, 8.0]),
        np.asarray([8.0, 6.0, 4.0, 2.0]),
        tp=2,
        dt_s=0.25,
    )
    np.testing.assert_allclose(observed, [2.5])
    np.testing.assert_allclose(predicted, [2.5])


def test_score_requires_a_declared_role_for_every_run():
    with pytest.raises(ValueError, match="no role for run 1"):
        score_cache(_cache(), load_artifact(), {0: "train"})
