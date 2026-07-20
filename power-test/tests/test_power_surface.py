"""
Claim:
Power depends on model-loaded/fabric floors, active duty, monotone one-hinge
compute and memory utilization, and iteration rate. A power limit is applied
only when the run binds it.

Plausible wrong implementations:
- Retain phase or token-budget features that encode an engine configuration.
- Reverse the pre/post-hinge slopes and violate concavity.
- Apply an assumed board cap when no operating limit was recorded.
- Apply FP8 scaling to memory traffic instead of compute work.
- Confuse resident-weight bytes with executed HBM byte rate.
- Apply the transformer FP8 fraction to a BF16 output head.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from power_surface import dtype_scale, predict, surface_design


def _bin(**overrides):
    d = {"tp": [2.0], "busy": [0.5], "n_active": [2.08e11], "pre_tok": [100.0],
         "dec_tok": [50.0], "w_read": [3e11], "kv_read": [0.5e11],
         "kv_write": [0.5e11], "w_bytes": [80e9],
         "engine_iterations_rate": [4.0],
         "engine_tokens_per_iteration": [37.5], "prefill_duty": [0.25],
         "decode_duty": [0.5], "fp8": [0.0]}
    d.update(overrides)
    return {key: np.asarray(value, float) for key, value in d.items()}


def test_hand_computed_design_row():
    # u_compute = 2 * 2.08e11 * 150 / (2 * 312e12) = 0.1
    # u_memory = (3e11 + 0.5e11 + 0.5e11) / (2 * 2e12) = 0.1
    design, names = surface_design(_bin(), "A100")
    expected = {"tp": 2.0, "tp_link": 2.0, "resident_weights": 1.0,
                "busy_tp": 1.0,
                "compute_linear": 0.2, "compute_hinge_0.4": 0.2,
                "memory_linear": 0.2, "memory_hinge_0.4": 0.2,
                "iter_rate": 0.008}
    assert names == list(expected)
    np.testing.assert_allclose(design[0], list(expected.values()), rtol=1e-12)


def test_hinge_saturates_but_linear_tail_remains():
    # tp = 1, n_active = 312e12, one token: u_compute = 2.
    d = _bin(tp=[1.0], n_active=[312e12], pre_tok=[1.0], dec_tok=[0.0])
    design, names = surface_design(d, "A100")
    row = dict(zip(names, design[0]))
    assert row["compute_linear"] == 2.0
    assert row["compute_hinge_0.4"] == 0.4
    assert row["tp_link"] == 0.0  # tp == 1 has no fabric floor


def test_fractional_fp8_scale():
    np.testing.assert_allclose(
        dtype_scale({"fp8_flop_frac": [0.0, 0.6, 1.0, 2.0, -1.0]}),
        [1.0, 0.7, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(dtype_scale({"fp8": [0.0, 1.0]}), [1.0, 0.5])


def test_component_compute_scales_transformer_but_not_output_head():
    base = _bin(
        tp=[1.0], pre_tok=[0.0], dec_tok=[1.0],
        transformer_active_params=[100.0], output_head_params=[50.0],
        logit_tokens_rate=[1.0], fp8=[1.0], fp8_flop_frac=[1.0],
    )
    design, names = surface_design(base, "A100")
    compute = dict(zip(names, design[0]))["compute_linear"]
    assert compute == (2.0 * 0.5 * 100.0 + 2.0 * 50.0) / 312e12


def test_predict_requires_an_explicit_power_limit_to_cap():
    design = np.array([[1.0], [1.0]])
    np.testing.assert_allclose(
        predict(design, [1000.0], np.array([2.0, 2.0]), "A100"),
        [1000.0, 1000.0],
    )
    np.testing.assert_allclose(
        predict(
            design, [1000.0], np.array([2.0, 2.0]), "A100",
            power_limit_w=np.array([400.0, 500.0]),
        ),
        [800.0, 1000.0],
    )


def test_deployment_idle_delta_is_per_gpu_and_precedes_cap():
    design = np.array([[1.0], [1.0]])
    np.testing.assert_allclose(
        predict(
            design, [100.0], np.array([1.0, 4.0]), "A100",
            loaded_idle_delta_w_per_gpu=10.0,
        ),
        [110.0, 140.0],
    )
    np.testing.assert_allclose(
        predict(
            np.array([[1.0]]), [1000.0], np.array([2.0]), "A100",
            loaded_idle_delta_w_per_gpu=100.0,
            power_limit_w=np.array([500.0]),
        ),
        [1000.0],
    )
