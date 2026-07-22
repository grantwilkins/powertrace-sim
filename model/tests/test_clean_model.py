"""
Claim:
The selected pipeline maps engine work to per-GPU power with the published
dense/MoE coordinates, scales it once to node power, enforces its support
boundary, and uses seeds only to realize uncertain output lengths.

Plausible wrong implementations:
- divide or multiply a utilization coordinate by TP at the wrong level;
- use memory utilization without the busy-duty square root;
- treat cached tokens as newly executed prompt work;
- silently run an unsupported MoE configuration;
- let a seed perturb an otherwise fixed deterministic request schedule.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from model.power.predictor import dense_design, moe_design, predict_power
from model.release import load_artifact
from model.request_schedule import realize_requests
from model.simulation import simulate
from model.timing.iteration import HARDWARE_PROFILES


def _ledger(n=1):
    keys = (
        "busy", "prefill_gemm_flops_rate", "decode_gemm_flops_rate",
        "prefill_attn_flops_rate", "decode_attn_flops_rate", "w_read",
        "prefill_attn_bytes_rate", "decode_attn_bytes_rate", "kv_read",
        "kv_write", "gemm_flops_rate", "attn_flops_rate",
        "engine_iterations_rate", "batch",
    )
    return {key: np.zeros(n) for key in keys}


def test_dense_coordinates_match_the_published_per_gpu_equation():
    profile = HARDWARE_PROFILES["A100"]
    ledger = _ledger()
    ledger["busy"][:] = 0.25
    ledger["prefill_gemm_flops_rate"][:] = 2 * profile["peak_flops_s"] * 0.2
    ledger["w_read"][:] = 2 * profile["hbm_bytes_s"] * 0.36
    arch = {"w_bytes": profile["hbm_capacity"], "family": "dense-test"}

    actual = dense_design(ledger, arch=arch, hardware="A100", tp=2)

    np.testing.assert_allclose(actual, [[1.0, 0.125, 0.2, 0.3]])


def test_dense_prediction_scales_per_gpu_power_to_node_once():
    ledger = _ledger()
    artifact = {
        "power": {"dense": {"A100": {
            "feature_names": [
                "idle_floor", "active_weight_fraction", "compute_util",
                "duty_sqrt_memory_util",
            ],
            "coefficients": [10.0, 20.0, 30.0, 40.0],
            "delay_s": 0.0,
        }}}
    }
    arch = {"w_bytes": 0.0, "family": "dense-test"}

    result = predict_power(
        ledger, arch=arch, model="test", hardware="A100", tp=4,
        artifact=artifact, dt_s=0.25,
    )

    np.testing.assert_allclose(result["mean_gpu_power_w"], [10.0])
    np.testing.assert_allclose(result["node_gpu_power_w"], [40.0])


def test_moe_coordinates_are_per_gpu_and_memory_is_lagged():
    profile = HARDWARE_PROFILES["A100"]
    ledger = _ledger(2)
    ledger["busy"][:] = 0.36
    ledger["w_read"][:] = [0.4, 0.8 * 2 * profile["hbm_bytes_s"]]
    ledger["gemm_flops_rate"][:] = 2 * profile["peak_flops_s"] * 0.25
    ledger["engine_iterations_rate"][:] = 500.0
    ledger["batch"][:] = 1.0

    actual = moe_design(
        ledger, hardware="A100", tp=2,
        feature_names=[
            "logical_memory_util_lag_250ms",
            "duty_sqrt_exact_compute_util",
            "engine_iterations_rate",
            "log_decode_batch",
        ],
    )

    np.testing.assert_allclose(actual[:, 0], [1e-13, 1e-13])
    np.testing.assert_allclose(actual[:, 1], [0.3, 0.3])
    np.testing.assert_allclose(actual[:, 2], [0.5, 0.5])
    np.testing.assert_allclose(actual[:, 3], np.log(2.0))


def test_output_length_sampling_is_seeded_but_fixed_schedules_are_not():
    uncertain = [{
        "arrival_time": 0.0,
        "input_tokens": 16,
        "output_tokens_distribution": {"values": [1, 9], "probabilities": [0.5, 0.5]},
    }]
    first, sampled = realize_requests(uncertain, seed=4)
    second, _ = realize_requests(uncertain, seed=4)
    assert sampled
    assert first == second

    fixed = [{"arrival_time": 0.0, "input_tokens": 16, "output_tokens": 4}]
    a = simulate(fixed, deployment="llama-3-8b-a100-tp1", seed=1)
    b = simulate(fixed, deployment="llama-3-8b-a100-tp1", seed=999)
    np.testing.assert_array_equal(a.power["node_gpu_power_w"], b.power["node_gpu_power_w"])
    assert a.requests == b.requests


def test_cached_prefix_is_context_not_executed_prompt_work():
    rows, _ = realize_requests([{
        "arrival_time": 0.0, "input_tokens": 32,
        "cached_prefix_tokens": 32, "output_tokens": 2,
    }])
    assert rows[0]["executed_input_tokens"] == 0
    result = simulate(rows, deployment="llama-3-8b-a100-tp1")
    assert result.requests[0]["ttft_s"] > 0.0


def test_unsupported_moe_configuration_requires_explicit_permission():
    artifact = load_artifact()
    custom = copy.deepcopy(artifact)
    custom["presets"]["bad-moe"] = {
        **custom["presets"]["gpt-oss-20b-a100-tp1"], "tp": 4,
    }
    request = [{"arrival_time": 0.0, "input_tokens": 8, "output_tokens": 2}]
    with pytest.raises(ValueError, match="MoE support excludes"):
        simulate(request, deployment="bad-moe", artifact=custom)
    result = simulate(
        request, deployment="bad-moe", artifact=custom,
        allow_unsupported=True,
    )
    assert result.support_status == "unsupported_extrapolation"

