"""
Claim:
The four trace panels use the user-fixed dense configuration, a central
repetition at each rate, exactly 600 seconds, and consistent per-GPU power from
the clean artifact.

Plausible wrong implementations:
- Optimize energy and DTW downward but accidentally optimize ACF R² downward.
- Admit a configuration that does not cover every selection rate.
- Cherry-pick the lowest-error repetition instead of a metric medoid.
- Crop 600 bins instead of 600 seconds at 250 ms resolution.
- Divide measured node power by TP but leave predicted power at node scale.
- Reuse the legacy dense artifact after the clean candidate has been selected.
- Average only measured power or use shifted/overlapping prediction windows.
"""
import numpy as np
import pytest

from plot_best_rate_traces import (
    add_alpha_gradient_line,
    crop_ten_minutes,
    metric_medoid,
    plot_series,
    select_configuration,
    target_configuration_rows,
)


def _rows(config, metrics, rates=(1.0, 2.0)):
    hardware, model, tp = config
    return [{
        "hardware": hardware,
        "model": model,
        "tp": tp,
        "rate": rate,
        "dense": True,
        "energy_error_pct": metrics[0],
        "acf_r2": metrics[1],
        "soft_dtw_divergence": metrics[2],
    } for rate in rates]


def test_joint_selection_uses_metric_directions_and_full_rate_coverage():
    rows = (
        _rows(("H100", "two-metric-winner", 2), (1.0, 0.8, 0.1))
        + _rows(("H100", "acf-winner", 8), (2.0, 0.9, 0.2))
        + _rows(("A100", "missing-rate", 1), (0.0, 1.0, 0.0), rates=(1.0,))
    )

    selected = select_configuration(rows, rates=(1.0, 2.0))

    assert selected["model"] == "two-metric-winner"
    assert selected["ranks"] == {
        "energy_error_pct": 1,
        "acf_r2": 2,
        "soft_dtw_divergence": 1,
    }


def test_showcase_constraint_excludes_other_models_and_tensor_parallelism():
    rows = (
        _rows(("H100", "llama-3-8b", 1), (3.0, 0.8, 0.3))
        + _rows(("A100", "llama-3-8b", 1), (0.0, 1.0, 0.0))
        + _rows(("H100", "llama-3-8b", 4), (0.0, 1.0, 0.0))
        + _rows(("H100", "other-model", 1), (0.0, 1.0, 0.0))
    )

    selected = target_configuration_rows(rows)

    assert {(row["hardware"], row["model"], row["tp"]) for row in selected} == {
        ("H100", "llama-3-8b", 1)
    }


def test_representative_trace_is_metric_medoid_not_best_run():
    rows = [
        {"run_id": 1, "energy_error_pct": 0.0, "acf_r2": 1.0,
         "soft_dtw_divergence": 0.0},
        {"run_id": 2, "energy_error_pct": 2.0, "acf_r2": 0.8,
         "soft_dtw_divergence": 2.0},
        {"run_id": 3, "energy_error_pct": 100.0, "acf_r2": -1.0,
         "soft_dtw_divergence": 100.0},
    ]

    assert metric_medoid(rows)["run_id"] == 2


def test_ten_minute_crop_uses_timestep_and_fails_when_short():
    values = np.arange(2401, dtype=float)
    cropped = crop_ten_minutes(values, 0.25)

    assert len(cropped) == 2400
    assert cropped[-1] == 2399
    with pytest.raises(ValueError, match="needs 600s"):
        crop_ten_minutes(values[:2399], 0.25)


def test_plot_series_averages_matched_one_second_per_gpu_blocks():
    measured = np.full(2400, 20.0)
    predicted = np.full(2400, 40.0)
    measured[:4] = [2.0, 4.0, 6.0, 8.0]
    predicted[:4] = [10.0, 12.0, 14.0, 16.0]
    time_min, measured_1s, predicted_1s = plot_series({
        "tp": 2,
        "dt_s": 0.25,
        "measured": measured,
        "predicted": predicted,
    })

    assert len(measured_1s) == len(predicted_1s) == 600
    assert measured_1s[0] == 2.5
    assert predicted_1s[0] == 6.5
    assert time_min[1] == pytest.approx(1.0 / 60.0)


def test_line_alpha_increases_without_changing_trace_coordinates():
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots()
    collection = add_alpha_gradient_line(
        axis, [0.0, 1.0, 2.0], [3.0, 4.0, 5.0], color="black",
        linewidth=1.0, alpha_range=(0.25, 0.75), label="trace",
    )

    np.testing.assert_allclose(
        collection.get_segments(),
        [[[0.0, 3.0], [1.0, 4.0]], [[1.0, 4.0], [2.0, 5.0]]],
    )
    np.testing.assert_allclose(collection.get_colors()[:, 3], [0.25, 0.75])
    plt.close(fig)
