"""
Claim:
Arrivals-only physics inference converts each request into consecutive prefill
and decode work while conserving its token counts on the output time grid.

Plausible wrong implementations:
- Start decode at arrival and overlap it with prefill.
- Divide by bin width in the wrong direction and lose token conservation.
- Round an off-grid arrival to a bin edge instead of using overlap fractions.
- Count a prefill or decode interval when its token count is zero.
"""

import json

import numpy as np

from model.classifiers.physics import FEATURE_ORDER, SCHEMA_VERSION
from model.pipeline.physics_inference import (
    build_modeled_work_ledger,
    run_physics_inference,
)


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
    assert result["generation_mode"] == "physics_modeled_mean"
    assert (tmp_path / "power.csv.manifest.json").is_file()
