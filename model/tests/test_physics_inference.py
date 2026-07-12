"""
Claim:
Arrivals-only physics inference converts each request into consecutive prefill
and decode work while conserving its token counts on the output time grid.

Plausible wrong implementations:
- Start decode at arrival and overlap it with prefill.
- Divide by bin width in the wrong direction and lose token conservation.
- Round an off-grid arrival to a bin edge instead of using overlap fractions.
- Count arrivals on the wrong side of a half-open bin boundary.
- Derive training and rollout work with subtly different arithmetic.
- Silently truncate unfinished request work at an explicit horizon.
- Label [t, t + dt) at t instead of t + dt.
"""

import json

import numpy as np
import pytest

from model.classifiers.physics import (
    FEATURE_ORDER,
    SCHEMA_VERSION,
    SELECTED_SCHEMA_VERSION,
    physics_feature_order,
)
from model.pipeline.physics_inference import (
    build_modeled_work_ledger,
    run_physics_inference,
)
from model.training_data.ledger_view import reconstruct_bins, schedule_work_rates


ARCH = {
    "n_layers": 2,
    "n_kv": 1,
    "head_dim": 2,
    "swa_window": 0,
    "moe_frac": 0,
    "top_k": 1,
    "n_experts": 1,
    "w_bytes": 100,
    "d_model": 4,
}
THROUGHPUT = {"lambda_prefill": 4.0, "lambda_decode": 2.0}


def _selected_artifact(timing_contract="conditional_timing", hardware="H100"):
    names = physics_feature_order("M0")
    return {
        "schema_version": SELECTED_SCHEMA_VERSION,
        "hardware": hardware,
        "dt_s": 1.0,
        "timing_contract": timing_contract,
        "architectures": {},
        "hardware_profile": {
            "hbm_bandwidth_bytes_s": 10.0,
            "compute_peak_flops_s": 100.0,
            "link_bandwidth_bytes_s": 20.0,
            "residence_bytes_per_gpu": 10.0,
        },
        "mode": {
            "candidate": "M0", "mean_kind": "M0", "residence": False,
            "coefficients": {name: 0.0 for name in names},
            "lag": {"moving_average_s": 1.0, "ema_alpha": 1.0},
            "state_filters": [], "cap_w_per_gpu": 1000.0,
        },
        "learned_scalar_count": len(names) + 2,
        "provenance": {
            "training_source_ids": ["train"],
            "selection_source_ids": ["dev"],
            "excluded_target_source_ids": ["test"],
        },
    }


def test_prefill_precedes_decode_on_exact_bins():
    ledger = build_modeled_work_ledger(
        [{"arrival_time": 0.0, "input_tokens": 4, "output_tokens": 2}],
        arch=ARCH,
        tp=1,
        throughput=THROUGHPUT,
        dt=1.0,
        T=2,
    )

    np.testing.assert_array_equal(ledger["pre_tok"], [4.0, 0.0])
    np.testing.assert_array_equal(ledger["dec_tok"], [0.0, 2.0])
    np.testing.assert_array_equal(ledger["pre_active"], [1.0, 0.0])
    np.testing.assert_array_equal(ledger["batch"], [0.0, 1.0])
    np.testing.assert_array_equal(ledger["arrivals"], [1.0, 0.0])
    np.testing.assert_array_equal(ledger["input_tokens_arriving"], [4.0, 0.0])
    np.testing.assert_array_equal(ledger["output_tokens_requested"], [2.0, 0.0])
    np.testing.assert_array_equal(ledger["A_t"], [1.0, 0.0])
    np.testing.assert_array_equal(ledger["delta_A_t"], [0.0, -1.0])
    np.testing.assert_array_equal(
        ledger["A_t"], ledger["running_requests"] + ledger["waiting_requests"]
    )


def test_off_grid_overlap_conserves_tokens():
    dt = 0.5
    ledger = build_modeled_work_ledger(
        [{"arrival_time": 0.25, "input_tokens": 4, "output_tokens": 2}],
        arch=ARCH,
        tp=1,
        throughput=THROUGHPUT,
        dt=dt,
    )

    assert np.sum(ledger["pre_tok"]) * dt == 4.0
    assert np.sum(ledger["dec_tok"]) * dt == 2.0
    assert np.sum(ledger["pre_active"]) * dt == 1.0
    assert np.sum(ledger["batch"]) * dt == 1.0


def test_arrival_boundary_and_waiting_state_are_hand_checkable():
    ledger = schedule_work_rates(
        arrivals=[0.0, 1.0],
        prefill_starts=[1.5, 1.0],
        prefill_ends=[2.0, 1.5],
        decode_ends=[3.0, 2.0],
        input_tokens=[2.0, 2.0],
        output_tokens=[2.0, 1.0],
        edges=[0.0, 1.0, 2.0, 3.0],
        arch=ARCH,
        tp=1,
    )

    np.testing.assert_array_equal(ledger["arrivals"], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(ledger["A_t"], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(ledger["waiting_requests"], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(ledger["running_requests"], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(ledger["delta_A_t"], [0.0, 0.0, -1.0])


def test_explicit_horizon_cannot_discard_request_tokens():
    request = [{"arrival_time": 0.25, "input_tokens": 4, "output_tokens": 2}]
    with pytest.raises(ValueError, match="truncates modeled request work"):
        build_modeled_work_ledger(
            request, arch=ARCH, tp=1, throughput=THROUGHPUT, dt=0.5, T=3
        )


def test_retrospective_and_rollout_use_identical_schedule_arithmetic():
    epoch = 1_780_000_000.0
    req = {
        "request_timestamps": np.asarray([epoch]),
        "ttfts": np.asarray([1.0]),
        "decode_times": np.asarray([11.0]),
        "input_lens": np.asarray([4.0]),
        "output_lens": np.asarray([22.0]),
        "has_timestamps": True,
    }
    pw = {
        "timestamps": epoch + np.arange(13, dtype=np.float64),
        "power": np.full(13, 100.0),
    }
    retrospective = reconstruct_bins(
        req, pw, ARCH, 1, lambda_prefill=4.0, dt=1.0, trim_s=0.0,
        arrival_alignment="exact_epoch",
    )
    rollout = build_modeled_work_ledger(
        [{"arrival_time": 0.0, "input_tokens": 4, "output_tokens": 22}],
        arch=ARCH, tp=1,
        throughput={"lambda_prefill": 4.0, "lambda_decode": 2.0},
        dt=1.0, T=12,
    )

    assert retrospective is not None
    for key, expected in rollout.items():
        np.testing.assert_array_equal(retrospective[key], expected, err_msg=key)


def test_standalone_physics_inference_uses_declared_artifacts(tmp_path):
    requests = tmp_path / "requests.json"
    requests.write_text(
        json.dumps(
            {
                "requests": [
                    {"arrival_time": 0.0, "input_tokens": 4, "output_tokens": 2}
                ]
            }
        )
    )
    throughput = tmp_path / "throughput.json"
    throughput.write_text(
        json.dumps(
            {
                "configs": {
                    "org/toy-model_H100_tp1": {
                        "prefill_rate_median_toks_per_s": 4,
                        "decode_rate_median_toks_per_s": 2,
                    }
                }
            }
        )
    )
    coefficients = {key: 0.0 for key in FEATURE_ORDER}
    coefficients["tp"] = 10.0
    artifact = tmp_path / "physics.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "dt_s": 1.0,
                "feature_order": list(FEATURE_ORDER),
                "architectures": {
                    "org/toy-model": {
                        **ARCH,
                        "family": "unit",
                        "n_active": 10.0,
                        "fp8": 0,
                    }
                },
                "hardware": {
                    "H100": {
                        "coefficients": coefficients,
                        "hbm_bandwidth_bytes_s": 1e12,
                        "family_multipliers": {},
                        "lag": {"moving_average_bins": 1, "ema_alpha": 1.0},
                        "cap_w_per_gpu": 1000,
                    }
                },
            }
        )
    )
    output = tmp_path / "power.csv"

    result = run_physics_inference(
        config_id="org/toy-model_H100_tp1",
        requests_json=str(requests),
        physics_artifact=str(artifact),
        throughput_db=str(throughput),
        out_csv=str(output),
    )

    values = np.genfromtxt(output, delimiter=",", names=True)
    np.testing.assert_array_equal(values["power_w"], [10.0, 10.0])
    np.testing.assert_array_equal(values["time_s"], [1.0, 2.0])
    assert result["generation_mode"] == "physics_modeled_mean"
    assert (tmp_path / "power.csv.manifest.json").is_file()


def test_arrival_only_inference_rejects_conditional_selected_artifact(tmp_path):
    artifact = tmp_path / "selected.json"
    artifact.write_text(json.dumps(_selected_artifact()))
    with pytest.raises(ValueError, match="Conditional-timing"):
        run_physics_inference(
            config_id="toy_H100_tp1", requests_json="missing-requests.json",
            physics_artifact=str(artifact), throughput_db="missing-throughput.json",
            out_csv=str(tmp_path / "power.csv"),
        )


def test_selected_artifact_hardware_must_match_config(tmp_path):
    artifact = tmp_path / "selected.json"
    artifact.write_text(json.dumps(_selected_artifact("arrival_only_validated", "A100")))
    with pytest.raises(ValueError, match="not 'H100'"):
        run_physics_inference(
            config_id="toy_H100_tp1", requests_json="missing-requests.json",
            physics_artifact=str(artifact), throughput_db="missing-throughput.json",
            out_csv=str(tmp_path / "power.csv"),
        )
