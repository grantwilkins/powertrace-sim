"""
Claim:
The figure compares the same measured run with the best supported model path,
uses a median repetition rather than the visually best trace, and reports
per-GPU watts consistently.

Plausible wrong implementations:
- Extrapolate the GPT-OSS-20B MoE surface to unsupported GPT-OSS-120B TP4/8.
- Select the repetition with minimum energy error instead of the metric medoid.
- Mix runs from another split, request rate, model, or TP cell.
- Divide measured power by TP but leave predicted node power unscaled.
"""
import numpy as np
import pytest

from plot_best_moe_curves import (
    candidate_run_ids,
    median_repetition,
    model_kind,
    per_gpu,
)


def test_model_paths_do_not_extrapolate_the_moe_surface_to_120b():
    assert model_kind("gpt-oss-20b") == "MoE v3"
    assert model_kind("gpt-oss-120b") == "Frozen dense comparator"
    with pytest.raises(ValueError, match="No validated"):
        model_kind("gpt-oss-200b")


def test_representative_is_metric_medoid_not_lowest_error():
    rows = [
        {"run_id": 1, "energy_error_pct": 0.0, "acf_mae": 0.0,
         "nrmse_range": 0.0},
        {"run_id": 2, "energy_error_pct": 2.0, "acf_mae": 2.0,
         "nrmse_range": 2.0},
        {"run_id": 3, "energy_error_pct": 100.0, "acf_mae": 100.0,
         "nrmse_range": 100.0},
    ]
    assert median_repetition(rows)["run_id"] == 2


def test_candidate_selection_respects_the_exact_experimental_cell():
    cache = {
        "model_names": np.asarray(["gpt-oss-20b", "gpt-oss-120b"]),
        "model_idx": np.asarray([0, 0, 0, 0, 1]),
        "role_names": np.asarray(["holdout_rate", "train"]),
        "role_idx": np.asarray([0, 0, 0, 1, 0]),
        "tp": np.asarray([1, 1, 2, 1, 1]),
        "rate": np.asarray([4.0, 2.0, 4.0, 4.0, 4.0]),
        "run_id": np.asarray([10, 11, 12, 13, 14]),
    }
    assert candidate_run_ids(
        cache, "gpt-oss-20b", 1, 4.0, "holdout_rate") == [10]


def test_per_gpu_scaling_applies_the_same_node_to_device_conversion():
    np.testing.assert_array_equal(per_gpu([400.0, 800.0], 4), [100.0, 200.0])
