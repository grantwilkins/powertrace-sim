"""
Claim:
The dense surface has a hardware idle floor, charges model footprint only while
active, applies nonlinear memory response at active duty, fits the reported
one-second timescale without crossing runs, and selects response delay using
that same objective.

Plausible wrong implementations:
- Let checkpoint size raise idle power.
- Compute sqrt of duty-averaged memory, filling partially active valleys.
- Pool transient or short idle bins into the hardware floor.
- Merge adjacent runs while constructing one-second bins.
- Fit raw 250 ms samples or hard-code zero response delay.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clean_dense_surface import (  # noqa: E402
    DENSE_FEATURES,
    filter_design,
    fit_dense_hardware,
    one_second_view,
    raw_design,
    sustained_idle_floor,
    tail_balanced_fit,
)
from power_surface import HARDWARE  # noqa: E402


def _sub(busy, compute_util, memory_util, resident=0.5, hardware="A100"):
    busy = np.asarray(busy, float)
    compute_util = np.asarray(compute_util, float)
    memory_util = np.asarray(memory_util, float)
    tp = np.ones(busy.size)
    return {
        "tp": tp,
        "busy": busy,
        "w_bytes": np.full(busy.size, resident * HARDWARE[hardware]["hbm_capacity_bytes"]),
        "w_read": memory_util * HARDWARE[hardware]["hbm_bandwidth_bytes_s"],
        "prefill_gemm_flops_rate": compute_util * HARDWARE[hardware]["compute_peak_flops_s"],
        "decode_gemm_flops_rate": np.zeros(busy.size),
        "prefill_attn_flops_rate": np.zeros(busy.size),
        "decode_attn_flops_rate": np.zeros(busy.size),
        "prefill_attn_bytes_rate": np.zeros(busy.size),
        "decode_attn_bytes_rate": np.zeros(busy.size),
    }


def test_idle_is_checkpoint_invariant_and_active_footprint_is_gated():
    design = raw_design(_sub([0.0, 1.0], [0.0, 0.0], [0.0, 0.0]), "A100")

    assert DENSE_FEATURES[1] == "active_weight_fraction"
    np.testing.assert_allclose(design[:, 1], [0.0, 0.5])


def test_memory_response_averages_instantaneous_response_over_duty():
    design = raw_design(_sub([0.25], [0.0], [0.25]), "A100")

    assert DENSE_FEATURES[-1] == "duty_sqrt_memory_util"
    np.testing.assert_allclose(design[:, -1], [0.25])
    assert not np.isclose(design[0, -1], np.sqrt(0.25))


def test_idle_floor_uses_only_settled_portion_of_sustained_gaps():
    run_id = np.repeat([0, 1], 20)
    busy = np.tile(np.r_[np.zeros(16), np.ones(4)], 2)
    power = np.r_[np.full(8, 100.0), np.full(8, 70.0), np.full(4, 300.0),
                  np.full(8, 90.0), np.full(8, 72.0), np.full(4, 300.0)]

    floor, source_runs = sustained_idle_floor(power, busy, run_id, {0, 1}, 0.25)

    assert floor == 71.0
    assert source_runs == 2


def test_one_second_bins_never_cross_run_boundaries():
    design = np.arange(12.0)[:, None]
    power = np.arange(12.0)
    run_id = np.r_[np.zeros(6, dtype=int), np.ones(6, dtype=int)]

    x, y, ids = one_second_view(design, power, run_id, {0, 1}, 0.25)

    np.testing.assert_allclose(x[:, 0], [1.5, 7.5])
    np.testing.assert_allclose(y, [1.5, 7.5])
    np.testing.assert_array_equal(ids, [0, 1])


def test_tail_balance_does_not_ignore_rare_high_power_seconds():
    design = np.asarray([[1.0, 0.0]] * 8 + [[1.0, 1.0]] * 2)
    target = np.asarray([100.0] * 9 + [200.0])
    runs = np.zeros(10, dtype=int)

    coefficients, _ = tail_balanced_fit(design, target, runs, {0}, 100.0)

    assert coefficients[1] > 0.0
    assert (design @ coefficients)[-1] > 100.0


def test_h100_delay_is_selected_under_one_second_objective():
    n = 96
    busy = np.r_[np.zeros(20), np.tile([0.0, 1.0, 1.0, 0.0], 19)]
    util = busy * np.resize([0.0, 0.2, 0.8, 0.0, 0.0, 0.9, 0.3, 0.0], n)
    sub = _sub(busy, util, np.zeros(n), resident=0.0, hardware="H100")
    sub["run_id"] = np.zeros(n, dtype=int)
    raw = raw_design(sub, "H100")
    target = filter_design(raw, sub["run_id"], 0.25, "H100", 0.25) @ np.asarray(
        [120.0, 0.0, 500.0, 0.0]
    )
    cache = sub | {
        "power": target,
        "hw_idx": np.zeros(n, dtype=int),
        "hw_names": np.asarray(["H100"]),
        "dt_s": np.asarray(0.25),
    }

    fit, _, _ = fit_dense_hardware(cache, "H100", {0})

    assert fit["delay_s"] == 0.25
    assert fit["fit_timestep_s"] == 1.0
