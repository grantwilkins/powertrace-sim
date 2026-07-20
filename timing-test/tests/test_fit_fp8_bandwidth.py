"""
Claim:
FP8 streaming calibration uses only the explicitly amended source rows and
cannot be affected by the remaining repeat/rate holdouts.

Plausible wrong implementations:
- Fit all rows from the 405B checkpoint.
- Treat repeat 2 as calibration because its model and rate match.
- Apply the fitted scalar to a different hardware silently.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from fit_fp8_bandwidth import calibrate  # noqa: E402


def _data(held_itl):
    return {
        "run_model": np.asarray(["llama-3-405b", "llama-3-405b"]),
        "run_repeat": np.asarray([0, 2]),
        "run_rate": np.asarray([1.0, 1.0]),
        "run_tp": np.asarray([8, 8]),
        "run_hardware": np.asarray(["H100", "H100"]),
        "req_run_id": np.asarray([0, 1]),
        "arrival_time_s": np.asarray([0.0, 0.0]),
        "ttft_s": np.asarray([0.1, 0.1]),
        "decode_duration_s": np.asarray([1.0, 1.0]),
        "output_tokens": np.asarray([16, 16]),
        "input_tokens": np.asarray([128, 128]),
        "itl_offsets": np.asarray([0, 15, 30]),
        "itl_values": np.asarray([0.08] * 15 + [held_itl] * 15),
    }


def _fit():
    return {
        "H100": {
            "eff_flops": 0.6,
            "eff_bw": 0.9,
            "base_overhead_s": 0.0,
            "per_message_s": {"8": 0.00004},
            "per_token_sample_s": 0.0,
        }
    }


def test_noncalibration_repeat_cannot_change_fp8_scale():
    manifest = {"roles": {"0": "holdout_model", "1": "holdout_model"}}
    fit_a, amended_a, _ = calibrate(_data(0.01), manifest, _fit())
    fit_b, amended_b, _ = calibrate(_data(1.0), manifest, _fit())
    assert fit_a["H100"]["fp8_stream_scale"] == (
        fit_b["H100"]["fp8_stream_scale"]
    )
    assert amended_a["roles"] == amended_b["roles"] == {
        "0": "dtype_calibration",
        "1": "holdout_model",
    }
