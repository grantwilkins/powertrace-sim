"""
Claim:
Disaggregated inference executes prompt work plus one discarded token on the
prefiller, starts full-output decode only after that stage, and sums the power
of two independent role engines.

Plausible wrong implementations:
- Treat the two role GPUs as one collocated TP2 engine.
- Start decode at the client arrival or execute the prompt again on the decoder.
- Remove the prefiller token from the requested decoder output.
- Apply a shared idle change once instead of once per physical role GPU.
- Report a deployment trace that differs from the sum of its role traces.
- Simulate the confirmatory 8192-token prompt as one unsupported scheduler step.
"""
import numpy as np

from model.disaggregated import (
    apply_shared_idle_calibration,
    simulate_disaggregated,
)
from model.timing.ledger import NATIVE_DT_S


def test_role_ledgers_conserve_phase_work_and_respect_stage_order():
    result = simulate_disaggregated(
        [{"arrival_time": 0.25, "input_tokens": 4, "output_tokens": 3}],
        deployment="gpt-oss-20b-a100-tp1",
        horizon_s=1.0,
    )
    prefill = result.roles["prefill"]
    decode = result.roles["decode"]

    assert np.sum(prefill.ledger["pre_tok"]) * NATIVE_DT_S == 4
    assert np.sum(prefill.ledger["dec_tok"]) * NATIVE_DT_S == 0
    assert np.sum(decode.ledger["pre_tok"]) * NATIVE_DT_S == 0
    assert np.sum(decode.ledger["dec_tok"]) * NATIVE_DT_S == 3
    assert np.sum(prefill.ledger["logit_tokens_rate"]) * NATIVE_DT_S == 1

    source_arrival = result.requests[0]["arrival_time"]
    decode_arrival = decode.timed[0]["arrival_s"]
    assert decode_arrival > source_arrival
    assert result.requests[0]["ttft_s"] > result.requests[0]["prefill_s"]


def test_total_power_is_exact_role_sum():
    result = simulate_disaggregated(
        [{"arrival_time": 0.0, "input_tokens": 8, "output_tokens": 2}],
        deployment="gpt-oss-20b-a100-tp1",
        horizon_s=1.0,
    )
    expected = sum(
        np.asarray(role.power["node_gpu_power_w"])
        for role in result.roles.values()
    )
    np.testing.assert_allclose(result.node_gpu_power_w, expected)


def test_confirmatory_long_prompt_uses_four_supported_prefill_chunks():
    result = simulate_disaggregated(
        [{"arrival_time": 0.0, "input_tokens": 8192, "output_tokens": 64}],
        deployment="gpt-oss-20b-a100-tp1",
        horizon_s=2.0,
    )

    assert [row[4] for row in result.roles["prefill"].trace] == [2048] * 4


def test_shared_idle_calibration_shifts_each_role_gpu_once():
    result = simulate_disaggregated(
        [{"arrival_time": 0.0, "input_tokens": 8, "output_tokens": 2}],
        deployment="gpt-oss-20b-a100-tp1",
        horizon_s=1.0,
    )
    calibrated = apply_shared_idle_calibration(
        result, result.source_idle_w_per_gpu + 5.0
    )

    for role in ("prefill", "decode"):
        source = np.asarray(result.roles[role].power["node_gpu_power_w"])
        np.testing.assert_allclose(calibrated[role] - source, 5.0)
    np.testing.assert_allclose(
        calibrated["node_gpu_power_w"] - result.node_gpu_power_w, 10.0
    )
